import unittest

import numpy as np

import havok_sound as havok


class HavokSoundTests(unittest.TestCase):
    def test_file_specific_steady_intervals(self):
        for filename, expected in (
            ("Taguan_003.2.wav", (80.0, 100.0)),
            ("Taguan_004.2.wav", (80.0, 100.0)),
        ):
            _, _, start_ms, end_ms, source = havok.resolve_segment_specification(
                filename, None, None, None, None
            )
            self.assertEqual((start_ms, end_ms), expected)
            self.assertEqual(source, "file_steady_default")

    def test_causal_hankel_has_explicit_alignment(self):
        signal = np.arange(10.0)
        matrix, indices = havok.causal_hankel(signal, n_delays=2, delay_interval=2)

        np.testing.assert_array_equal(indices, np.arange(4, 10))
        np.testing.assert_array_equal(matrix[0], [4.0, 2.0, 0.0])
        np.testing.assert_array_equal(matrix[-1], [9.0, 7.0, 5.0])

    def test_metrics_are_exact_for_identical_signals(self):
        signal = np.array([-1.0, 0.5, 2.0, -0.25])
        metrics = havok.regression_metrics(signal, signal.copy())

        self.assertEqual(metrics["rmse"], 0.0)
        self.assertEqual(metrics["mae"], 0.0)
        self.assertEqual(metrics["nrmse_std"], 0.0)
        self.assertAlmostEqual(metrics["correlation"], 1.0)
        self.assertAlmostEqual(metrics["r2"], 1.0)

    def test_autonomous_havok_forecasts_a_clean_tone(self):
        sample = np.arange(1400)
        signal = np.sin(2.0 * np.pi * 0.137 * sample)
        train_end = 1000
        config = havok.HavokConfig(
            n_delays=32,
            delay_interval=1,
            rank=2,
            ridge_alpha=0.0,
            retained_energy=1.0,
        )

        result = havok.fit_and_predict(signal, train_end, config, "autonomous")
        metrics = havok.regression_metrics(signal[train_end:], result["prediction"])

        self.assertLess(metrics["nrmse_std"], 0.05)
        self.assertGreater(metrics["correlation"], 0.99)


if __name__ == "__main__":
    unittest.main()
