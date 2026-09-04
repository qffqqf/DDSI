"""HAVOK identification for the high-current portion of a stud-welding sound.

The script reports three deliberately different quantities:

1. ``observed-forcing reconstruction``: a HAVOK reconstruction for which the
   forcing coordinates are measured from the target waveform. This is useful
   for judging the identified linear state equation, but it is not a blind
   forecast.
2. ``one-step prediction``: retained only as a JSON debugging diagnostic and
   omitted from the result figure.
3. ``free forecast``: after the train/test split no measured test samples are
   used. This is the strictest prediction reported by the script.

The original WAV sample rate is retained by default. This matters for these
recordings because their strongest spectral line is close to 25 kHz.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Literal

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.io import wavfile
from scipy.linalg import svd
from scipy.signal import resample_poly, welch


EPS = np.finfo(float).eps

# File-specific defaults requested for the comparatively stationary,
# high-amplitude acoustic part of each welding event.
DEFAULT_STEADY_SEGMENTS_MS = {
    "Taguan_003.2.wav": (70.0, 90.0),
    "Taguan_004.2.wav": (80.0, 100.0),
}


@dataclass(frozen=True)
class HavokConfig:
    """Hyperparameters selected without looking at the final test interval."""

    n_delays: int
    delay_interval: int
    rank: int
    ridge_alpha: float
    n_forcing: int = 0
    retained_energy: float = math.nan


@dataclass
class SearchChoice:
    config: HavokConfig
    validation_metrics: dict[str, float]


def _safe_stem(path: str | Path) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", Path(path).stem).strip("_")


def _to_builtin(value):
    """Recursively convert NumPy values so they can be written as JSON."""

    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def load_wav_mono(audio_path: str | Path) -> tuple[float, np.ndarray]:
    """Load a WAV file as an uncalibrated, floating-point mono waveform."""

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    fs, data = wavfile.read(audio_path)
    if data.ndim == 2:
        data = np.mean(data.astype(np.float64), axis=1)
    else:
        data = data.astype(np.float64)
    if data.ndim != 1 or len(data) < 16:
        raise ValueError("The WAV file must contain a non-empty mono time series.")
    return float(fs), data


def resample_waveform(
    data: np.ndarray, fs: float, target_fs: float | None
) -> tuple[np.ndarray, float]:
    """Resample the complete recording, avoiding filter transients at the segment."""

    if target_fs is None or target_fs <= 0 or np.isclose(target_fs, fs):
        return data.copy(), fs
    if target_fs > fs:
        raise ValueError("target_fs must not exceed the native WAV sample rate.")

    ratio = Fraction(float(target_fs) / float(fs)).limit_denominator(2000)
    result = resample_poly(data, ratio.numerator, ratio.denominator)
    actual_fs = fs * ratio.numerator / ratio.denominator
    return np.asarray(result, dtype=np.float64), float(actual_fs)


def segment_bounds(
    n_samples: int,
    fs: float,
    start_ratio: float,
    end_ratio: float,
    start_ms: float | None,
    end_ms: float | None,
) -> tuple[int, int]:
    """Resolve either explicit millisecond bounds or recording-length ratios."""

    using_ms = start_ms is not None or end_ms is not None
    if using_ms and (start_ms is None or end_ms is None):
        raise ValueError("segment_start_ms and segment_end_ms must be given together.")
    if using_ms:
        start = int(round(start_ms * 1e-3 * fs))
        end = int(round(end_ms * 1e-3 * fs))
    else:
        start = int(math.floor(start_ratio * n_samples))
        end = int(math.floor(end_ratio * n_samples))

    if not 0 <= start < end <= n_samples:
        raise ValueError(
            f"Invalid segment [{start}, {end}) for a recording of {n_samples} samples."
        )
    return start, end


def resolve_segment_specification(
    audio_path: str | Path,
    start_ratio: float | None,
    end_ratio: float | None,
    start_ms: float | None,
    end_ms: float | None,
) -> tuple[float, float, float | None, float | None, str]:
    """Resolve explicit bounds before applying file-specific steady defaults."""

    using_ms = start_ms is not None or end_ms is not None
    using_ratio = start_ratio is not None or end_ratio is not None
    if using_ms and using_ratio:
        raise ValueError("Use millisecond bounds or ratio bounds, not both.")
    if using_ms:
        if start_ms is None or end_ms is None:
            raise ValueError(
                "segment_start_ms and segment_end_ms must be given together."
            )
        return 0.0, 1.0, start_ms, end_ms, "explicit_milliseconds"
    if using_ratio:
        if start_ratio is None or end_ratio is None:
            raise ValueError("start_ratio and end_ratio must be given together.")
        return start_ratio, end_ratio, None, None, "explicit_ratios"

    steady_bounds = DEFAULT_STEADY_SEGMENTS_MS.get(Path(audio_path).name)
    if steady_bounds is not None:
        return 0.0, 1.0, steady_bounds[0], steady_bounds[1], "file_steady_default"
    return 0.30, 0.50, None, None, "fallback_30_to_50_percent"


def spectral_energy_above(x: np.ndarray, fs: float, cutoff_hz: float) -> float:
    """Return the fraction of Welch spectral energy above ``cutoff_hz``."""

    nperseg = min(len(x), 4096)
    frequencies, spectrum = welch(x - np.mean(x), fs=fs, nperseg=nperseg)
    total = float(np.sum(spectrum))
    if total <= EPS:
        return 0.0
    return float(np.sum(spectrum[frequencies > cutoff_hz]) / total)


def causal_hankel(
    x: np.ndarray, n_delays: int, delay_interval: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Create rows ``[x(t), x(t-d), ..., x(t-n_delays*d)]``.

    The returned sample indices explicitly map each Hankel row to ``x``. This
    prevents the offset ambiguity that occurred with ``pykoopman.TimeDelay``.
    """

    if n_delays < 1 or delay_interval < 1:
        raise ValueError("n_delays and delay_interval must be positive integers.")
    consumed = n_delays * delay_interval
    if len(x) <= consumed:
        raise ValueError(
            f"Need more than {consumed} samples for this delay embedding; got {len(x)}."
        )

    if delay_interval == 1:
        hankel = sliding_window_view(x, n_delays + 1)[:, ::-1]
        sample_indices = np.arange(n_delays, len(x), dtype=int)
    else:
        sample_indices = np.arange(consumed, len(x), dtype=int)
        offsets = delay_interval * np.arange(n_delays + 1, dtype=int)
        hankel = x[sample_indices[:, None] - offsets[None, :]]
    return np.asarray(hankel), sample_indices


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Metrics in the same amplitude scale as the supplied arrays."""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("Metric inputs must be non-empty arrays with equal shape.")

    error = y_true - y_pred
    mse = float(np.mean(error**2))
    variance = float(np.var(y_true))
    true_std = math.sqrt(max(variance, EPS))
    correlation = (
        float(np.corrcoef(y_true, y_pred)[0, 1])
        if np.std(y_pred) > EPS and np.std(y_true) > EPS
        else 0.0
    )
    return {
        "rmse": math.sqrt(mse),
        "mae": float(np.mean(np.abs(error))),
        "nrmse_std": math.sqrt(mse) / true_std,
        "correlation": correlation,
        "r2": 1.0 - mse / max(variance, EPS),
    }


def spectral_rmse_db(y_true: np.ndarray, y_pred: np.ndarray, fs: float) -> float:
    """RMSE between Welch spectra in dB; lower is better."""

    nperseg = min(len(y_true), 512)
    _, p_true = welch(y_true, fs=fs, nperseg=nperseg)
    _, p_pred = welch(y_pred, fs=fs, nperseg=nperseg)
    floor = max(float(np.max(p_true)), EPS) * 1e-12
    true_db = 10.0 * np.log10(np.maximum(p_true, floor))
    pred_db = 10.0 * np.log10(np.maximum(p_pred, floor))
    return float(np.sqrt(np.mean((true_db - pred_db) ** 2)))


def relative_ridge(X: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    """Fit ``X @ W = Y`` with a scale-independent ridge parameter."""

    if alpha < 0:
        raise ValueError("ridge alpha must be non-negative.")
    if alpha == 0:
        return np.linalg.lstsq(X, Y, rcond=None)[0]

    gram = X.T @ X
    scale = max(float(np.trace(gram)) / max(gram.shape[0], 1), EPS)
    regularized = gram + alpha * scale * np.eye(gram.shape[0])
    return np.linalg.solve(regularized, X.T @ Y)


def stabilize_row_operator(
    operator: np.ndarray, max_radius: float = 0.99995
) -> tuple[np.ndarray, float, float]:
    """Uniformly contract an autonomous row operator only when it is unstable."""

    radius_before = float(np.max(np.abs(np.linalg.eigvals(operator))))
    if not np.isfinite(radius_before):
        raise FloatingPointError("Non-finite eigenvalue in the fitted operator.")
    factor = 1.0
    if radius_before > max_radius:
        factor = max_radius / radius_before
        operator = operator * factor
    return operator, radius_before, factor


def stabilize_forced_operator(
    operator: np.ndarray, n_state: int, max_radius: float = 0.99995
) -> tuple[np.ndarray, float, float]:
    """Stabilize only the state block of ``[state, forcing] @ W``."""

    result = operator.copy()
    state_block, radius, factor = stabilize_row_operator(
        result[:n_state, :], max_radius=max_radius
    )
    result[:n_state, :] = state_block
    return result, radius, factor


def simulate_autonomous(q_initial: np.ndarray, operator: np.ndarray, steps: int) -> np.ndarray:
    result = np.empty((steps, len(q_initial)), dtype=float)
    state = q_initial.copy()
    for index in range(steps):
        state = state @ operator
        result[index] = state
    return result


def simulate_observed_forcing(
    state_initial: np.ndarray,
    latent: np.ndarray,
    operator: np.ndarray,
    row_before_first_target: int,
    steps: int,
    n_state: int,
) -> np.ndarray:
    """Simulate HAVOK states while supplying measured forcing coordinates."""

    result = np.empty((steps, latent.shape[1]), dtype=float)
    state = state_initial.copy()
    for offset in range(steps):
        previous_row = row_before_first_target + offset
        forcing_previous = latent[previous_row, n_state:]
        state = np.concatenate((state, forcing_previous)) @ operator
        forcing_target = latent[previous_row + 1, n_state:]
        result[offset] = np.concatenate((state, forcing_target))
    return result


def candidate_ranks(singular_values: np.ndarray, maximum_rank: int) -> list[int]:
    """Combine energy-based ranks with a small deterministic search grid."""

    available = min(len(singular_values), maximum_rank)
    cumulative = np.cumsum(singular_values**2) / np.sum(singular_values**2)
    candidates = {2, 4, 8, 12, 16, 24, 32, 48, 64, 96}
    for threshold in (0.90, 0.95, 0.98, 0.99, 0.995, 0.999):
        candidates.add(int(np.searchsorted(cumulative, threshold) + 1))
    return sorted(rank for rank in candidates if 2 <= rank <= available)


def _finite_score(metrics: dict[str, float]) -> float:
    score = metrics["nrmse_std"]
    return score if np.isfinite(score) else math.inf


def search_havok_configs(
    x: np.ndarray,
    train_end: int,
    validation_fraction: float = 0.25,
    delay_candidates: Iterable[int] = (64, 128, 256),
    delay_interval: int = 1,
    alpha_candidates: Iterable[float] = (0.0, 1e-8, 1e-6, 1e-4, 1e-2),
    maximum_rank: int = 96,
    maximum_forcing: int = 16,
    minimum_autonomous_energy: float = 0.90,
) -> tuple[SearchChoice, SearchChoice, SearchChoice]:
    """Select reconstruction and forecast models using only training data."""

    fit_end = int(round(train_end * (1.0 - validation_fraction)))
    if not 0.5 <= 1.0 - validation_fraction < 1.0:
        raise ValueError("validation_fraction must leave 50%-99% for fitting.")

    best_forced: SearchChoice | None = None
    best_auto: SearchChoice | None = None
    best_one_step: SearchChoice | None = None
    best_forced_score = math.inf
    best_auto_score = math.inf
    best_one_step_score = math.inf

    for n_delays in sorted(set(int(value) for value in delay_candidates)):
        consumed = n_delays * delay_interval
        if consumed + 4 >= fit_end:
            continue

        hankel, _ = causal_hankel(x[:train_end], n_delays, delay_interval)
        n_fit_rows = fit_end - consumed
        h_fit = hankel[:n_fit_rows]
        _, singular_values, right_vectors_t = svd(
            h_fit, full_matrices=False, lapack_driver="gesdd"
        )
        total_energy = float(np.sum(singular_values**2))

        for rank in candidate_ranks(singular_values, maximum_rank):
            basis = right_vectors_t[:rank].T
            latent = (hankel @ basis) / singular_values[:rank]
            decoder = singular_values[:rank] * right_vectors_t[:rank, 0]
            retained = float(np.sum(singular_values[:rank] ** 2) / total_energy)
            fit_inputs = latent[: n_fit_rows - 1]
            validation_steps = train_end - fit_end

            for alpha in alpha_candidates:
                try:
                    raw_auto_operator = relative_ridge(
                        fit_inputs, latent[1:n_fit_rows], float(alpha)
                    )
                    one_step_prediction = (
                        latent[n_fit_rows - 1 : -1] @ raw_auto_operator
                    ) @ decoder
                    one_step_metrics = regression_metrics(
                        x[fit_end:train_end], one_step_prediction
                    )
                    one_step_score = _finite_score(one_step_metrics)
                    if one_step_score < best_one_step_score:
                        config = HavokConfig(
                            n_delays=n_delays,
                            delay_interval=delay_interval,
                            rank=rank,
                            ridge_alpha=float(alpha),
                            n_forcing=0,
                            retained_energy=retained,
                        )
                        best_one_step = SearchChoice(config, one_step_metrics)
                        best_one_step_score = one_step_score

                    auto_operator = raw_auto_operator.copy()
                    auto_operator, _, _ = stabilize_row_operator(auto_operator)
                    auto_latent = simulate_autonomous(
                        latent[n_fit_rows - 1], auto_operator, validation_steps
                    )
                    auto_prediction = auto_latent @ decoder
                    auto_metrics = regression_metrics(
                        x[fit_end:train_end], auto_prediction
                    )
                    auto_score = _finite_score(auto_metrics)
                    if (
                        retained >= minimum_autonomous_energy
                        and auto_score < best_auto_score
                    ):
                        config = HavokConfig(
                            n_delays=n_delays,
                            delay_interval=delay_interval,
                            rank=rank,
                            ridge_alpha=float(alpha),
                            n_forcing=0,
                            retained_energy=retained,
                        )
                        best_auto = SearchChoice(config, auto_metrics)
                        best_auto_score = auto_score
                except (np.linalg.LinAlgError, FloatingPointError, ValueError):
                    pass

                for n_forcing in (1, 2, 4, 8, 16):
                    if n_forcing > maximum_forcing or 2 * n_forcing > rank:
                        continue
                    n_state = rank - n_forcing
                    try:
                        forced_operator = relative_ridge(
                            fit_inputs,
                            latent[1:n_fit_rows, :n_state],
                            float(alpha),
                        )
                        forced_operator, _, _ = stabilize_forced_operator(
                            forced_operator, n_state
                        )
                        forced_latent = simulate_observed_forcing(
                            latent[n_fit_rows - 1, :n_state],
                            latent,
                            forced_operator,
                            n_fit_rows - 1,
                            validation_steps,
                            n_state,
                        )
                        forced_prediction = forced_latent @ decoder
                        forced_metrics = regression_metrics(
                            x[fit_end:train_end], forced_prediction
                        )
                        forced_score = _finite_score(forced_metrics)
                        if forced_score < best_forced_score:
                            config = HavokConfig(
                                n_delays=n_delays,
                                delay_interval=delay_interval,
                                rank=rank,
                                ridge_alpha=float(alpha),
                                n_forcing=n_forcing,
                                retained_energy=retained,
                            )
                            best_forced = SearchChoice(config, forced_metrics)
                            best_forced_score = forced_score
                    except (np.linalg.LinAlgError, FloatingPointError, ValueError):
                        pass

    if best_forced is None or best_auto is None or best_one_step is None:
        raise RuntimeError("No valid HAVOK configuration was found.")
    return best_forced, best_auto, best_one_step


def fit_and_predict(
    x: np.ndarray,
    train_end: int,
    config: HavokConfig,
    mode: Literal["observed_forcing", "autonomous"],
) -> dict:
    """Refit one selected configuration on all training data and predict test data."""

    hankel, sample_indices = causal_hankel(
        x, config.n_delays, config.delay_interval
    )
    consumed = config.n_delays * config.delay_interval
    n_train_rows = train_end - consumed
    if n_train_rows < config.rank + 2:
        raise ValueError("Too few training Hankel rows for the selected rank.")

    h_train = hankel[:n_train_rows]
    _, singular_values, right_vectors_t = svd(
        h_train, full_matrices=False, lapack_driver="gesdd"
    )
    rank = min(config.rank, len(singular_values))
    basis = right_vectors_t[:rank].T
    latent = (hankel @ basis) / singular_values[:rank]
    decoder = singular_values[:rank] * right_vectors_t[:rank, 0]
    oracle = latent @ decoder
    steps = len(x) - train_end
    row_before_test = n_train_rows - 1

    result = {
        "sample_indices": sample_indices,
        "singular_values": singular_values,
        "cumulative_energy": np.cumsum(singular_values**2)
        / np.sum(singular_values**2),
        "oracle": oracle,
        "decoder": decoder,
        "latent": latent,
    }

    if mode == "autonomous":
        raw_operator = relative_ridge(
            latent[: n_train_rows - 1],
            latent[1:n_train_rows],
            config.ridge_alpha,
        )
        one_step_latent = latent[:-1] @ raw_operator
        one_step = one_step_latent @ decoder
        one_step_samples = sample_indices[1:]
        one_step_test_start = int(np.searchsorted(one_step_samples, train_end))

        operator = raw_operator.copy()
        operator, radius_before, contraction = stabilize_row_operator(operator)
        predicted_latent = simulate_autonomous(
            latent[row_before_test], operator, steps
        )
        prediction = predicted_latent @ decoder

        result.update(
            {
                "prediction": prediction,
                "operator": operator,
                "spectral_radius_before": radius_before,
                "stability_contraction": contraction,
                "one_step": one_step[one_step_test_start:],
            }
        )
    elif mode == "observed_forcing":
        n_state = rank - config.n_forcing
        operator = relative_ridge(
            latent[: n_train_rows - 1],
            latent[1:n_train_rows, :n_state],
            config.ridge_alpha,
        )
        operator, radius_before, contraction = stabilize_forced_operator(
            operator, n_state
        )
        predicted_latent = simulate_observed_forcing(
            latent[row_before_test, :n_state],
            latent,
            operator,
            row_before_test,
            steps,
            n_state,
        )
        prediction = predicted_latent @ decoder
        result.update(
            {
                "prediction": prediction,
                "operator": operator,
                "spectral_radius_before": radius_before,
                "stability_contraction": contraction,
                "forcing_norm": np.linalg.norm(latent[:, n_state:], axis=1),
            }
        )
    else:
        raise ValueError(f"Unknown prediction mode: {mode}")
    return result


def _add_spectral_metric(
    metrics: dict[str, float], truth: np.ndarray, prediction: np.ndarray, fs: float
) -> dict[str, float]:
    result = dict(metrics)
    result["spectral_rmse_db"] = spectral_rmse_db(truth, prediction, fs)
    return result


def plot_results(
    output_path: Path,
    audio_name: str,
    fs: float,
    segment_start_s: float,
    full_x: np.ndarray,
    x: np.ndarray,
    train_end: int,
    forced: dict,
    autonomous: dict,
    metrics: dict[str, dict[str, float]],
    show: bool,
) -> None:
    """Plot observed-forcing reconstruction and strict free prediction."""

    full_time_ms = np.arange(len(full_x)) / fs * 1000.0
    time_ms = (segment_start_s + np.arange(len(x)) / fs) * 1000.0
    test_time_ms = time_ms[train_end:]
    truth = x[train_end:]
    forced_prediction = forced["prediction"]
    free_prediction = autonomous["prediction"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    fig.suptitle(
        f"Discrete HAVOK identification - {audio_name}\n",
        fontsize=15,
    )

    ax = axes[0, 0]
    ax.plot(
        full_time_ms,
        full_x,
        color="0.2",
        linewidth=0.55,
        label="Measured sound pressure",
    )
    ax.axvspan(
        time_ms[0],
        time_ms[train_end],
        color="tab:blue",
        alpha=0.10,
        label="Training data",
    )
    ax.axvspan(
        time_ms[train_end],
        time_ms[-1],
        color="tab:orange",
        alpha=0.16,
        label="Test data",
    )
    ax.axvline(
        time_ms[-1],
        color="tab:green",
        linestyle="--",
        linewidth=0.9,
    )
    ax.axvline(
        time_ms[train_end],
        color="tab:blue",
        linestyle=":",
    )
    ax.set(
        title="Measured data with train/test split",
        xlabel="Recording time (ms)",
        ylabel="Normalized amplitude",
    )
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)

    zoom_samples = min(len(truth), max(32, int(round(0.002 * fs))))
    ax = axes[0, 1]
    zoom = slice(0, zoom_samples)
    ax.plot(test_time_ms, truth, color="black", linewidth=1.1, label="Measured")
    ax.plot(
        test_time_ms,
        forced_prediction,
        color="tab:green",
        linestyle="-",
        linewidth=1.0,
        label="Observed-forcing reconstruction",
    )
    ax.set(
        title="Test data and predictions",
        xlabel="Recording time (ms)",
        ylabel="Normalized amplitude",
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    ax.plot(
        test_time_ms,
        2* np.abs(truth - forced_prediction)/(np.abs(truth) + np.max(np.abs(truth))),
        color="tab:green",
        linewidth=0.7,
        label="Observed forcing",
    )
    ax.set(
        title="Relative test error",
        xlabel="Recording time (ms)",
        ylabel="Relative error",
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    forcing_time_ms = (
        segment_start_s + forced["sample_indices"] / fs
    ) * 1000.0
    forcing_magnitude = forced["forcing_norm"]
    forcing_train_mask = forced["sample_indices"] < train_end
    forcing_scale = math.sqrt(
        float(np.mean(forcing_magnitude[forcing_train_mask] ** 2))
    )
    normalized_forcing = forcing_magnitude / max(forcing_scale, EPS)
    ax.plot(
        forcing_time_ms,
        normalized_forcing,
        color="tab:green",
        linewidth=0.8,
        label=r"$\|\mathbf{v}(t)\|_2$",
    )
    ax.axvline(
        time_ms[train_end],
        color="tab:blue",
        linestyle=":",
        label="Train/test split",
    )
    ax.axvspan(forcing_time_ms[0], time_ms[train_end], color="tab:blue", alpha=0.05)
    ax.axvspan(time_ms[train_end], forcing_time_ms[-1], color="tab:orange", alpha=0.08)
    ax.set(
        title="Identified external-forcing magnitude",
        xlabel="Recording time (ms)",
        ylabel="Forcing magnitude / training RMS",
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0.075, 1, 0.955))
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def run_havok_analysis(
    audio_path: str | Path = "Taguan_003.2.wav",
    target_fs: float | None = None,
    start_ratio: float | None = None,
    end_ratio: float | None = None,
    segment_start_ms: float | None = None,
    segment_end_ms: float | None = None,
    train_fraction: float = 0.80,
    validation_fraction: float = 0.25,
    output_dir: str | Path = ".",
    search_profile: Literal["quick", "balanced", "thorough"] = "balanced",
    show: bool = False,
) -> dict:
    """Run leakage-aware HAVOK identification and write a figure plus JSON report."""

    native_fs, native_data = load_wav_mono(audio_path)
    (
        resolved_start_ratio,
        resolved_end_ratio,
        resolved_start_ms,
        resolved_end_ms,
        segment_source,
    ) = resolve_segment_specification(
        audio_path,
        start_ratio,
        end_ratio,
        segment_start_ms,
        segment_end_ms,
    )
    native_start, native_end = segment_bounds(
        len(native_data),
        native_fs,
        resolved_start_ratio,
        resolved_end_ratio,
        resolved_start_ms,
        resolved_end_ms,
    )
    native_segment = native_data[native_start:native_end]

    data, fs = resample_waveform(native_data, native_fs, target_fs)
    start_seconds = native_start / native_fs
    end_seconds = native_end / native_fs
    start = int(round(start_seconds * fs))
    end = min(len(data), int(round(end_seconds * fs)))
    segment = np.asarray(data[start:end], dtype=float)

    if not 0.55 <= train_fraction <= 0.95:
        raise ValueError("train_fraction must lie between 0.55 and 0.95.")
    train_end = int(round(len(segment) * train_fraction))
    if train_end < 256 or len(segment) - train_end < 32:
        raise ValueError("The selected audio segment is too short for train/test analysis.")

    train_mean = float(np.mean(segment[:train_end]))
    train_scale = float(np.std(segment[:train_end]))
    if train_scale <= EPS:
        raise ValueError("The training interval has zero variance.")
    x = (segment - train_mean) / train_scale
    full_x = (data - train_mean) / train_scale

    if search_profile == "quick":
        delays = (64, 128)
        alphas = (0.0, 1e-4, 1e-2)
        maximum_rank = 64
        maximum_forcing = 8
    elif search_profile == "thorough":
        delays = (32, 64, 128, 256, 384)
        alphas = (0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2)
        maximum_rank = 128
        maximum_forcing = 16
    else:
        delays = (32, 64, 128, 256, 384)
        alphas = (0.0, 1e-8, 1e-6, 1e-4, 1e-2)
        maximum_rank = 128
        maximum_forcing = 16

    print(f"\nReading: {audio_path}")
    print(
        f"Native fs={native_fs:.0f} Hz, analysis fs={fs:.0f} Hz, "
        f"segment={start_seconds * 1000:.3f}-{end_seconds * 1000:.3f} ms "
        f"({len(segment)} samples)"
    )
    if fs < native_fs:
        removed_energy = spectral_energy_above(native_segment, native_fs, fs / 2.0)
        print(
            f"WARNING: resampling removes approximately {100 * removed_energy:.1f}% "
            "of native-band spectral energy above the new Nyquist frequency."
        )
    else:
        removed_energy = 0.0

    print("Selecting HAVOK hyperparameters using the training-only validation block...")
    forced_choice, auto_choice, one_step_choice = search_havok_configs(
        x,
        train_end,
        validation_fraction=validation_fraction,
        delay_candidates=delays,
        alpha_candidates=alphas,
        maximum_rank=maximum_rank,
        maximum_forcing=maximum_forcing,
    )
    print(f"Observed-forcing choice: {forced_choice.config}")
    print(f"Autonomous choice:       {auto_choice.config}")

    forced = fit_and_predict(x, train_end, forced_choice.config, "observed_forcing")
    autonomous = fit_and_predict(x, train_end, auto_choice.config, "autonomous")
    one_step_model = fit_and_predict(
        x, train_end, one_step_choice.config, "autonomous"
    )
    truth = x[train_end:]
    metrics = {
        "observed_forcing": _add_spectral_metric(
            regression_metrics(truth, forced["prediction"]), truth, forced["prediction"], fs
        ),
        "one_step": _add_spectral_metric(
            regression_metrics(truth, one_step_model["one_step"]),
            truth,
            one_step_model["one_step"],
            fs,
        ),
        "free_forecast": _add_spectral_metric(
            regression_metrics(truth, autonomous["prediction"]), truth, autonomous["prediction"], fs
        ),
    }
    for values in metrics.values():
        values["rmse_pcm_counts"] = values["rmse"] * train_scale
        values["mae_pcm_counts"] = values["mae"] * train_scale

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(audio_path)
    figure_path = output_dir / f"havok_audio_{stem}_improved.png"
    report_path = output_dir / f"havok_audio_{stem}_metrics.json"
    plot_results(
        figure_path,
        Path(audio_path).name,
        fs,
        start_seconds,
        full_x,
        x,
        train_end,
        forced,
        autonomous,
        metrics,
        show,
    )

    report = {
        "audio_path": str(Path(audio_path)),
        "native_sample_rate_hz": native_fs,
        "analysis_sample_rate_hz": fs,
        "segment_start_ms": start_seconds * 1000.0,
        "segment_end_ms": end_seconds * 1000.0,
        "segment_source": segment_source,
        "segment_samples": len(segment),
        "train_fraction": train_fraction,
        "train_samples": train_end,
        "test_samples": len(segment) - train_end,
        "normalization": {
            "training_mean_pcm_counts": train_mean,
            "training_std_pcm_counts": train_scale,
            "note": "WAV amplitudes are uncalibrated PCM counts, not pascals.",
        },
        "estimated_energy_removed_by_resampling": removed_energy,
        "observed_forcing": {
            "interpretation": "reconstruction using measured test forcing; not a blind forecast",
            "config": asdict(forced_choice.config),
            "validation_metrics": forced_choice.validation_metrics,
            "test_metrics": metrics["observed_forcing"],
            "spectral_radius_before_stabilization": forced["spectral_radius_before"],
            "stability_contraction": forced["stability_contraction"],
        },
        "autonomous": {
            "interpretation": "free forecast after the split; no measured test samples used",
            "config": asdict(auto_choice.config),
            "validation_metrics": auto_choice.validation_metrics,
            "free_forecast_test_metrics": metrics["free_forecast"],
            "spectral_radius_before_stabilization": autonomous["spectral_radius_before"],
            "stability_contraction": autonomous["stability_contraction"],
        },
        "debug_one_step": {
            "interpretation": (
                "next-sample prediction using measured samples only through "
                "the previous instant"
            ),
            "config": asdict(one_step_choice.config),
            "validation_metrics": one_step_choice.validation_metrics,
            "test_metrics": metrics["one_step"],
            "spectral_radius_before_stabilization": one_step_model[
                "spectral_radius_before"
            ],
        },
        "outputs": {"figure": str(figure_path), "metrics": str(report_path)},
    }
    report_path.write_text(
        json.dumps(_to_builtin(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for label in ("observed_forcing", "free_forecast"):
        values = metrics[label]
        print(
            f"{label:>18}: NRMSE={values['nrmse_std']:.4f}, "
            f"corr={values['correlation']:.4f}, R2={values['r2']:.4f}, "
            f"spectral RMSE={values['spectral_rmse_db']:.2f} dB"
        )
    print(f"Figure: {figure_path}")
    print(f"Metrics: {report_path}")
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Leakage-aware discrete HAVOK analysis of a welding sound."
    )
    parser.add_argument(
        "--audio",
        nargs="+",
        default=["Taguan_003.2.wav"],
        help="One or more WAV files (default: Taguan_003.2.wav).",
    )
    parser.add_argument(
        "--target-fs",
        type=float,
        default=None,
        help="Optional analysis sample rate. Native rate is retained by default.",
    )
    parser.add_argument(
        "--start-ratio",
        type=float,
        default=None,
        help="Override the file-specific steady interval with a start ratio.",
    )
    parser.add_argument(
        "--end-ratio",
        type=float,
        default=None,
        help="Override the file-specific steady interval with an end ratio.",
    )
    parser.add_argument("--segment-start-ms", type=float, default=None)
    parser.add_argument("--segment-end-ms", type=float, default=None)
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument(
        "--search-profile",
        choices=("quick", "balanced", "thorough"),
        default="balanced",
    )
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--show", action="store_true")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    for audio_path in args.audio:
        run_havok_analysis(
            audio_path=audio_path,
            target_fs=args.target_fs,
            start_ratio=args.start_ratio,
            end_ratio=args.end_ratio,
            segment_start_ms=args.segment_start_ms,
            segment_end_ms=args.segment_end_ms,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            output_dir=args.output_dir,
            search_profile=args.search_profile,
            show=args.show,
        )


if __name__ == "__main__":
    main()
