"""Unit tests for rowing.analysis.loaders (pure file parsing, offline)."""

from rowing.analysis import loaders


class _Upload:
    """Minimal stand-in for a Streamlit UploadedFile (has .name and .read())."""

    def __init__(self, name, data):
        self.name = name
        self._data = data

    def read(self):
        return self._data


def test_parse_gpx(cam_gpx):
    with open(cam_gpx, "rb") as f:
        positions = loaders.parse_gpx(f)
    assert len(positions) > 0
    assert {"latitude", "longitude", "distance"}.issubset(positions.columns)


def test_parse_telemetry_text(powerline_txt):
    upload = _Upload("powerline.txt", powerline_txt.read_bytes())
    data = loaders.parse_telemetry_text([upload])
    assert "powerline" in data
    assert "positions" in data["powerline"]
    assert "power" in data["powerline"]
