"""Regression tests fuer Monitoring-History-Downsampling."""

from kiara.web.api_monitoring import _downsample_rows


def test_downsample_rows_preserves_first_and_last_point():
    rows = [{"ts": i, "cpu_percent": i} for i in range(101)]

    sampled = _downsample_rows(rows, 10)

    assert len(sampled) == 10
    assert sampled[0] == rows[0]
    assert sampled[-1] == rows[-1]
