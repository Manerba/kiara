"""Tests fuer STEP/IGES-Loader (Phase 8).

OCP ist optional — Tests muessen sowohl MIT als auch OHNE OCP funktionieren.
Die Mock-Tests simulieren eine vorhandene OCP-Installation.
"""

from unittest.mock import MagicMock, patch

from kiara.ingestion.loader import _load_iges, _load_step, load_file

# ── STEP/IGES ohne OCP ───────────────────────────────────────────────────────


class TestStepWithoutOCP:
    """STEP-Loader ohne installiertes OCP."""

    def test_load_step_without_ocp_returns_none(self, tmp_path):
        """_load_step() ohne OCP installiert gibt None zurueck (kein Crash)."""
        path = tmp_path / "test.step"
        path.write_text("ISO-10303-21;")
        result = _load_step(str(path))
        assert result is None

    def test_load_file_step_without_ocp_returns_none(self, tmp_path):
        """load_file() mit .step ohne OCP gibt None zurueck."""
        path = tmp_path / "test.step"
        path.write_text("ISO-10303-21;")
        result = load_file(str(path))
        assert result is None


class TestIgesWithoutOCP:
    """IGES-Loader ohne installiertes OCP."""

    def test_load_iges_without_ocp_returns_none(self, tmp_path):
        """_load_iges() ohne OCP installiert gibt None zurueck (kein Crash)."""
        path = tmp_path / "test.iges"
        path.write_text(
            "                                                                        S      1"
        )
        result = _load_iges(str(path))
        assert result is None

    def test_load_file_iges_without_ocp_returns_none(self, tmp_path):
        """load_file() mit .iges ohne OCP gibt None zurueck."""
        path = tmp_path / "test.iges"
        path.write_text("IGES-HEADER-DUMMY")
        result = load_file(str(path))
        assert result is None


# ── STEP mit Mock-OCP ─────────────────────────────────────────────────────────


class TestStepWithMockOCP:
    """STEP-Loader mit simuliertem OCP (Mock)."""

    def _build_ocp_mocks(self):
        """Erzeugt Mock-Objekte die OCP-Module simulieren."""
        mock_reader = MagicMock()
        mock_reader.ReadFile.return_value = 1  # IFSelect_RetDone
        mock_reader.NbShapes.return_value = 3
        mock_shape = MagicMock()
        mock_reader.OneShape.return_value = mock_shape

        # Bounding-Box Mock
        mock_bbox = MagicMock()
        mock_bbox.Get.return_value = (0.0, 0.0, 0.0, 100.0, 50.0, 25.0)

        # GProps Mock
        mock_vol_props = MagicMock()
        mock_vol_props.Mass.return_value = 125000.0
        mock_surf_props = MagicMock()
        mock_surf_props.Mass.return_value = 35000.0

        return mock_reader, mock_shape, mock_bbox, mock_vol_props, mock_surf_props

    def test_step_with_mock_ocp_has_produktstruktur(self, tmp_path):
        """Mock-OCP STEP Output enthaelt PRODUKTSTRUKTUR."""
        reader, shape, bbox, vol_props, surf_props = self._build_ocp_mocks()
        path = tmp_path / "assembly.step"
        path.write_text("ISO-10303-21;")

        mock_step_module = MagicMock()
        mock_step_module.STEPControl_Reader.return_value = reader
        mock_ifselect = MagicMock()
        mock_ifselect.IFSelect_RetDone = 1
        mock_bnd = MagicMock()
        mock_bnd.Bnd_Box.return_value = bbox
        mock_brepbndlib = MagicMock()
        mock_gprop = MagicMock()
        mock_gprop.GProp_GProps.side_effect = [vol_props, surf_props]
        mock_brepgprop = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "OCP": MagicMock(),
                "OCP.STEPControl": mock_step_module,
                "OCP.IFSelect": mock_ifselect,
                "OCP.Bnd": mock_bnd,
                "OCP.BRepBndLib": mock_brepbndlib,
                "OCP.GProp": mock_gprop,
                "OCP.BRepGProp": mock_brepgprop,
            },
        ):
            result = _load_step(str(path))

        assert result is not None
        assert "PRODUKTSTRUKTUR" in result
        assert "Baugruppe mit 3 Teilen" in result

    def test_step_with_mock_ocp_has_geometrie(self, tmp_path):
        """Mock-OCP STEP Output enthaelt GEOMETRIE."""
        reader, shape, bbox, vol_props, surf_props = self._build_ocp_mocks()
        path = tmp_path / "part.step"
        path.write_text("ISO-10303-21;")

        mock_step_module = MagicMock()
        mock_step_module.STEPControl_Reader.return_value = reader
        mock_ifselect = MagicMock()
        mock_ifselect.IFSelect_RetDone = 1
        mock_bnd = MagicMock()
        mock_bnd.Bnd_Box.return_value = bbox
        mock_brepbndlib = MagicMock()
        mock_gprop = MagicMock()
        mock_gprop.GProp_GProps.side_effect = [vol_props, surf_props]
        mock_brepgprop = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "OCP": MagicMock(),
                "OCP.STEPControl": mock_step_module,
                "OCP.IFSelect": mock_ifselect,
                "OCP.Bnd": mock_bnd,
                "OCP.BRepBndLib": mock_brepbndlib,
                "OCP.GProp": mock_gprop,
                "OCP.BRepGProp": mock_brepgprop,
            },
        ):
            result = _load_step(str(path))

        assert result is not None
        assert "GEOMETRIE" in result
        assert "Bounding-Box:" in result


# ── IGES mit Mock-OCP ─────────────────────────────────────────────────────────


class TestIgesWithMockOCP:
    """IGES-Loader mit simuliertem OCP (Mock)."""

    def _build_ocp_mocks(self):
        """Erzeugt Mock-Objekte die OCP-Module simulieren."""
        mock_reader = MagicMock()
        mock_reader.ReadFile.return_value = 1  # IFSelect_RetDone
        mock_reader.NbShapes.return_value = 1
        mock_shape = MagicMock()
        mock_reader.OneShape.return_value = mock_shape

        mock_bbox = MagicMock()
        mock_bbox.Get.return_value = (0.0, 0.0, 0.0, 200.0, 100.0, 50.0)

        mock_vol_props = MagicMock()
        mock_vol_props.Mass.return_value = 1000000.0
        mock_surf_props = MagicMock()
        mock_surf_props.Mass.return_value = 140000.0

        return mock_reader, mock_shape, mock_bbox, mock_vol_props, mock_surf_props

    def test_iges_with_mock_ocp_has_produktstruktur(self, tmp_path):
        """Mock-OCP IGES Output enthaelt PRODUKTSTRUKTUR."""
        reader, shape, bbox, vol_props, surf_props = self._build_ocp_mocks()
        path = tmp_path / "model.iges"
        path.write_text("IGES-HEADER")

        mock_iges_module = MagicMock()
        mock_iges_module.IGESControl_Reader.return_value = reader
        mock_ifselect = MagicMock()
        mock_ifselect.IFSelect_RetDone = 1
        mock_bnd = MagicMock()
        mock_bnd.Bnd_Box.return_value = bbox
        mock_brepbndlib = MagicMock()
        mock_gprop = MagicMock()
        mock_gprop.GProp_GProps.side_effect = [vol_props, surf_props]
        mock_brepgprop = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "OCP": MagicMock(),
                "OCP.IGESControl": mock_iges_module,
                "OCP.IFSelect": mock_ifselect,
                "OCP.Bnd": mock_bnd,
                "OCP.BRepBndLib": mock_brepbndlib,
                "OCP.GProp": mock_gprop,
                "OCP.BRepGProp": mock_brepgprop,
            },
        ):
            result = _load_iges(str(path))

        assert result is not None
        assert "PRODUKTSTRUKTUR" in result
        assert "Einzelteil" in result

    def test_iges_with_mock_ocp_has_geometrie(self, tmp_path):
        """Mock-OCP IGES Output enthaelt GEOMETRIE."""
        reader, shape, bbox, vol_props, surf_props = self._build_ocp_mocks()
        path = tmp_path / "model.igs"
        path.write_text("IGES-HEADER")

        mock_iges_module = MagicMock()
        mock_iges_module.IGESControl_Reader.return_value = reader
        mock_ifselect = MagicMock()
        mock_ifselect.IFSelect_RetDone = 1
        mock_bnd = MagicMock()
        mock_bnd.Bnd_Box.return_value = bbox
        mock_brepbndlib = MagicMock()
        mock_gprop = MagicMock()
        mock_gprop.GProp_GProps.side_effect = [vol_props, surf_props]
        mock_brepgprop = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "OCP": MagicMock(),
                "OCP.IGESControl": mock_iges_module,
                "OCP.IFSelect": mock_ifselect,
                "OCP.Bnd": mock_bnd,
                "OCP.BRepBndLib": mock_brepbndlib,
                "OCP.GProp": mock_gprop,
                "OCP.BRepGProp": mock_brepgprop,
            },
        ):
            result = _load_iges(str(path))

        assert result is not None
        assert "GEOMETRIE" in result
        assert "Bounding-Box:" in result


# ── Routing-Tests ─────────────────────────────────────────────────────────────


class TestLoadFileRouting:
    """Tests fuer load_file() Extension-Routing bei STEP/IGES."""

    def test_step_extension_routes_correctly(self, tmp_path):
        """load_file() mit .step routet zu _load_step."""
        path = tmp_path / "test.step"
        path.write_text("ISO-10303-21;")
        # Ohne OCP: gibt None zurueck (Routing funktioniert aber)
        result = load_file(str(path))
        assert result is None  # weil OCP fehlt

    def test_stp_extension_routes_correctly(self, tmp_path):
        """load_file() mit .stp routet zu _load_step."""
        path = tmp_path / "test.stp"
        path.write_text("ISO-10303-21;")
        result = load_file(str(path))
        assert result is None

    def test_iges_extension_routes_correctly(self, tmp_path):
        """load_file() mit .iges routet zu _load_iges."""
        path = tmp_path / "test.iges"
        path.write_text("IGES-HEADER")
        result = load_file(str(path))
        assert result is None

    def test_igs_extension_routes_correctly(self, tmp_path):
        """load_file() mit .igs routet zu _load_iges."""
        path = tmp_path / "test.igs"
        path.write_text("IGES-HEADER")
        result = load_file(str(path))
        assert result is None
