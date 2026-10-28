import numpy as np

from eyehand.timebase import FrameClock, drift_ppm, phase_ms

PERIOD = 1000 / 30


def feed(clock, times):
    return [clock.add(t) for t in times]


def test_dejitter_recovers_period_and_clean_times():
    rng = np.random.default_rng(1)
    true_t = 3_000_000.0 + PERIOD * np.arange(300)
    noisy = true_t + rng.uniform(-2, 2, true_t.size)  # measured kernel jitter is about +/-2 ms
    clock = FrameClock(PERIOD)
    out = feed(clock, noisy)
    assert abs(clock.period_ms - PERIOD) < 0.01
    fit_err = np.array([f.t_fit_ms for f in out[-100:]]) - true_t[-100:]
    assert np.abs(fit_err).max() < 0.5          # much cleaner than the raw +/-2 ms
    assert 0.9 < clock.residual_rms_ms() < 1.3  # uniform(-2,2) has RMS 1.15


def test_dropped_frames_are_counted_and_numbered():
    t = 1000.0 + PERIOD * np.array([0, 1, 2, 3, 6, 7, 8, 10])
    clock = FrameClock(PERIOD)
    out = feed(clock, t)
    assert [f.n for f in out] == [0, 1, 2, 3, 6, 7, 8, 10]
    assert clock.dropped == 3
    assert out[4].dropped_before == 2


def test_phase_and_drift_between_cameras():
    a, b = FrameClock(PERIOD), FrameClock(PERIOD)
    offset = 7.0
    period_b = PERIOD * (1 + 500e-6)  # b runs 500 ppm slow
    feed(a, 10_000 + PERIOD * np.arange(200))
    feed(b, 10_000 + offset + period_b * np.arange(200))
    assert abs(drift_ppm(a, b) - 500) < 5
    # Near the start the grids are offset by `offset`.
    assert abs(phase_ms(a, b, 10_000 + 5 * PERIOD) - (offset + 5 * (period_b - PERIOD))) < 0.05


def test_phase_wraps_into_half_period():
    a, b = FrameClock(PERIOD), FrameClock(PERIOD)
    feed(a, PERIOD * np.arange(50))
    feed(b, PERIOD * np.arange(50) + PERIOD - 3.0)  # same as -3 ms
    assert abs(phase_ms(a, b, 20 * PERIOD) + 3.0) < 1e-6
