"""Unit tests for rowing.analysis.pieces (pure piece-analysis compute, offline)."""

from rowing.analysis import loaders, pieces, splits


class _Upload:
    def __init__(self, name, data):
        self.name = name
        self._data = data

    def read(self):
        return self._data


def test_get_crossing_times_for_cam(cam_gpx):
    with open(cam_gpx, "rb") as f:
        positions = loaders.parse_gpx(f)
    landmarks = splits.load_location_landmarks("cam")

    crossings = pieces.get_crossing_times({"cam": positions}, locations=landmarks)

    # Empty results are dropped, so "cam" present means crossings were found.
    assert "cam" in crossings
    assert len(crossings["cam"]) > 0


def test_interpolate_power_for_powerline(powerline_txt):
    upload = _Upload("powerline.txt", powerline_txt.read_bytes())
    telemetry_data = loaders.parse_telemetry_text([upload])

    interpolated = pieces.interpolate_power(telemetry_data, dists=0.01)

    assert "powerline" in interpolated
    frame = interpolated["powerline"]
    assert len(frame) > 0
    # resampled onto a regular distance index named "distance"
    assert frame.index.name == "distance"
