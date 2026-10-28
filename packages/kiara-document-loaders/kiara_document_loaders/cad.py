"""CAD-Loader: DXF, STEP, IGES."""

import concurrent.futures
import logging
import re
from pathlib import Path

_OCP_READ_TIMEOUT_SECS = 300
_DXF_READ_TIMEOUT_SECS = 300

logger = logging.getLogger(__name__)


# Schriftfeld-Tags fuer DXF INSERT-Attribute (case-insensitive Match)
_TITLE_BLOCK_TAGS = {
    # Artikeldaten
    "ARTIKELNR",
    "ARTIKELNUMMER",
    "TEILENR",
    "TEILENUMMER",
    "PART_NO",
    "PART_NUMBER",
    "DRAWING_NO",
    "ZEICHNUNGSNR",
    # Benennung
    "BENENNUNG",
    "BEZEICHNUNG",
    "DESCRIPTION",
    "NAME",
    "TITLE",
    # Material
    "MATERIAL",
    "WERKSTOFF",
    "MAT",
    # Gewicht
    "GEWICHT",
    "WEIGHT",
    "MASS",
    # Bearbeiter
    "BEARBEITER",
    "ERSTELLT",
    "DRAWN",
    "DRAWN_BY",
    "GEZEICHNET",
    # Pruefer
    "GEPRÜFT",
    "PRUEFER",
    "CHECKED",
    "CHECKED_BY",
    # Datum
    "DATUM",
    "DATE",
    "ERSTELLT_AM",
    # Revision
    "REVISION",
    "REV",
    "AENDERUNG",
    "ÄNDERUNG",
    "INDEX",
    # Massstab
    "MASSSTAB",
    "MASSTAB",
    "SCALE",
    # Norm/Toleranz
    "NORM",
    "TOLERANZ",
    "TOLERANCE",
    # Blatt
    "BLATT",
    "SHEET",
    "SEITE",
    # Firma
    "FIRMA",
    "COMPANY",
}

_BOM_TAGS = {
    "POS",
    "POSITION",
    "MENGE",
    "QTY",
    "QUANTITY",
    "BEZEICHNUNG",
    "DESCRIPTION",
    "BENENNUNG",
    "MATERIAL",
    "WERKSTOFF",
    "NORM",
    "BEMERKUNG",
}

_MTEXT_FORMAT_RE = re.compile(
    r"\\[fFHWTQAaCcpPS][^;]*;"  # format codes like \fArial|...; \H2.5;
    r"|\\[\\{}]"  # escaped chars (~ handled by replace)
    r"|\{|\}"  # grouping braces
    r"|\\[OoLlKk]"  # strike/underline toggles
)


def _dxf_clean_mtext(raw: str) -> str:
    """Entferne DXF-Formatcodes aus MTEXT-Strings."""
    text = _MTEXT_FORMAT_RE.sub("", raw)
    text = text.replace("\\P", "\n").replace("\\~", " ")
    return text.strip()


def _dxf_extract_title_block(doc) -> list[tuple[str, str]]:
    """Extrahiere Schriftfeld-Attribute aus INSERT-Entities.

    Ueberspringt INSERT-Bloecke die BOM-typische Tags wie POS/MENGE enthalten,
    um Verwechslungen mit Stuecklisten-Zeilen zu vermeiden.
    """
    # Tags die eindeutig auf BOM-Zeilen hindeuten (nicht im Schriftfeld)
    _BOM_ONLY_TAGS = {"POS", "POSITION", "MENGE", "QTY", "QUANTITY"}

    results = []
    seen = set()

    def _scan_layout(layout):
        for entity in layout:
            if entity.dxftype() == "INSERT":
                try:
                    attribs = list(entity.attribs)
                except Exception as exc:
                    logger.warning("DXF INSERT-Block uebersprungen (attribs-Fehler): %s", exc)
                    continue
                # Prüfe ob dieser INSERT-Block BOM-typische Tags hat → skip
                attrib_tags = {a.dxf.tag.upper() for a in attribs}
                if attrib_tags & _BOM_ONLY_TAGS:
                    continue
                for attrib in attribs:
                    tag = attrib.dxf.tag.upper()
                    val = attrib.dxf.text.strip()
                    if tag in _TITLE_BLOCK_TAGS and val and tag not in seen:
                        results.append((tag, val))
                        seen.add(tag)

    # Modelspace
    _scan_layout(doc.modelspace())
    # Alle Layouts (Papierraum)
    for layout in doc.layouts:
        if layout.name != "Model":
            _scan_layout(layout)
    return results


def _dxf_extract_bom(doc) -> list[dict]:
    """Extrahiere Stueckliste aus TABLE oder INSERT-Patterns."""
    rows = []
    all_entities = list(doc.modelspace())
    for layout in doc.layouts:
        if layout.name != "Model":
            all_entities.extend(layout)

    # Methode 1: TABLE-Entities (selten, aber moeglich)
    for entity in all_entities:
        if entity.dxftype() == "ACAD_TABLE":
            try:
                table = entity
                headers = []
                for col in range(table.dxf.columns):
                    cell = table.cell(0, col)
                    headers.append(cell.text.strip().upper() if cell.text else "")
                bom_match = sum(1 for h in headers if h in _BOM_TAGS)
                if bom_match >= 3:
                    for r in range(1, table.dxf.rows):
                        row_data = {}
                        for c in range(table.dxf.columns):
                            cell = table.cell(r, c)
                            row_data[headers[c]] = cell.text.strip() if cell.text else ""
                        if any(row_data.values()):
                            rows.append(row_data)
            except Exception:
                continue

    if rows:
        return rows

    # Methode 2: Wiederholte INSERT-Bloecke mit BOM-Attributen
    block_rows = {}
    seen_bom_rows = set()
    for entity in all_entities:
        if entity.dxftype() == "INSERT":
            try:
                attribs = list(entity.attribs)
            except Exception:
                continue
            attrib_dict = {}
            for attrib in attribs:
                tag = attrib.dxf.tag.upper()
                val = attrib.dxf.text.strip()
                if tag in _BOM_TAGS and val:
                    attrib_dict[tag] = val
            if len(attrib_dict) >= 2:
                attrib_key = tuple(sorted(attrib_dict.items()))
                if attrib_key in seen_bom_rows:
                    continue
                seen_bom_rows.add(attrib_key)
                block_name = entity.dxf.name
                if block_name not in block_rows:
                    block_rows[block_name] = []
                block_rows[block_name].append(attrib_dict)

    # Finde den Block mit den meisten Zeilen (= wahrscheinlich BOM)
    for _block_name, block_data in sorted(block_rows.items(), key=lambda x: -len(x[1])):
        if len(block_data) >= 2:
            return block_data

    return []


def _dxf_extract_texts_by_layer(doc) -> dict[str, list[str]]:
    """Sammle alle TEXT/MTEXT-Entities, gruppiert nach Layer."""
    layers = {}
    seen_texts = set()
    all_entities = list(doc.modelspace())
    for layout in doc.layouts:
        if layout.name != "Model":
            all_entities.extend(layout)
    for entity in all_entities:
        dtype = entity.dxftype()
        if dtype == "TEXT":
            layer = entity.dxf.layer
            text = entity.dxf.text.strip()
            insert = getattr(entity.dxf, "insert", None)
            y = insert[1] if insert is not None and len(insert) >= 2 else 0
            if text and (layer, text) not in seen_texts:
                seen_texts.add((layer, text))
                layers.setdefault(layer, []).append((y, text))
        elif dtype == "MTEXT":
            layer = entity.dxf.layer
            try:
                text = entity.plain_text().strip()
            except Exception:
                text = _dxf_clean_mtext(entity.text or "")
            insert = getattr(entity.dxf, "insert", None)
            y = insert[1] if insert is not None and len(insert) >= 2 else 0
            if text and (layer, text) not in seen_texts:
                seen_texts.add((layer, text))
                layers.setdefault(layer, []).append((y, text))

    # Sortiere innerhalb jedes Layers nach Y-Koordinate (oben → unten = absteigend)
    result = {}
    for layer, items in layers.items():
        items.sort(key=lambda x: -x[0])
        result[layer] = [text for _, text in items]
    return result


def _dxf_extract_dimensions(doc) -> list[str]:
    """Extrahiere Bemassungstexte."""
    import math

    dims = []
    seen_dims = set()
    all_entities = list(doc.modelspace())
    for layout in doc.layouts:
        if layout.name != "Model":
            all_entities.extend(layout)
    for entity in all_entities:
        if entity.dxftype() == "DIMENSION":
            text = entity.dxf.text if hasattr(entity.dxf, "text") else ""
            if not text or "<>" in text:
                # measurement-Attribut oder manuelle Berechnung aus Definitionspunkten
                try:
                    val = entity.measurement
                except Exception:
                    try:
                        p2 = entity.dxf.defpoint2
                        p3 = entity.dxf.defpoint3
                        if len(p2) < 3 or len(p3) < 3:
                            continue
                        val = math.sqrt(
                            (p3[0] - p2[0]) ** 2 + (p3[1] - p2[1]) ** 2 + (p3[2] - p2[2]) ** 2
                        )
                        if not math.isfinite(val):
                            continue
                    except Exception:
                        continue
                measurement = f"{val:.2f}"
                text = text.replace("<>", measurement) if text else measurement
            if text and text.strip() not in seen_dims:
                seen_dims.add(text.strip())
                dims.append(text.strip())
    return dims


def _dxf_extract_leaders(doc) -> list[str]:
    """Extrahiere Leader/Multileader-Texte (Fertigungshinweise)."""
    texts = []
    seen_texts = set()
    all_entities = list(doc.modelspace())
    for layout in doc.layouts:
        if layout.name != "Model":
            all_entities.extend(layout)
    for entity in all_entities:
        dtype = entity.dxftype()
        if dtype == "LEADER":
            # LEADER hat oft zugehoerigen MTEXT ueber annotation_handle
            try:
                handle = entity.dxf.annotation_handle
                annotation = doc.entitydb.get(handle)
                if annotation and annotation.dxftype() == "MTEXT":
                    try:
                        text = annotation.plain_text().strip()
                    except Exception:
                        text = _dxf_clean_mtext(annotation.text or "")
                    if text and text not in seen_texts:
                        seen_texts.add(text)
                        texts.append(text)
            except Exception as exc:
                logger.debug("LEADER annotation_handle nicht lesbar: %s", exc)
        elif dtype == "MULTILEADER":
            try:
                ctx = entity.context
                if ctx.mtext:
                    try:
                        text = ctx.mtext.plain_text().strip()
                    except Exception:
                        text = _dxf_clean_mtext(ctx.mtext.text or "") if ctx.mtext.text else ""
                    if text and text not in seen_texts:
                        seen_texts.add(text)
                        texts.append(text)
            except Exception as exc:
                logger.debug("MULTILEADER context nicht lesbar: %s", exc)
    return texts


def _load_dxf(path: str) -> str:
    """Load .dxf CAD files — extract title block, BOM, annotations, dimensions."""
    try:
        import ezdxf
    except ImportError:
        logger.warning("ezdxf nicht installiert — DXF-Datei uebersprungen: %s", path)
        return None

    # Hard-Timeout fuer pathologische DXF-Dateien (Millionen Entities,
    # manipulierte Dateien). Pattern analog _load_ocp_model.
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        try:
            doc = pool.submit(ezdxf.readfile, path).result(
                timeout=_DXF_READ_TIMEOUT_SECS,
            )
        except concurrent.futures.TimeoutError:
            logger.warning(
                "Timeout (%ds) reading DXF file %s",
                _DXF_READ_TIMEOUT_SECS,
                path,
            )
            return None
        except Exception as exc:
            logger.warning("DXF-Datei nicht lesbar: %s — %s", path, exc)
            return None
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    filename = Path(path).name
    sections = [f"=== CAD-ZEICHNUNG: {filename} ==="]

    # 1. Schriftfeld
    title_block = _dxf_extract_title_block(doc)
    if title_block:
        sections.append("\n--- SCHRIFTFELD ---")
        for tag, val in title_block:
            sections.append(f"{tag}: {val}")

    # 2. Stueckliste
    bom = _dxf_extract_bom(doc)
    if bom:
        sections.append("\n--- STUECKLISTE ---")
        all_keys = []
        for row in bom:
            for k in row:
                if k not in all_keys:
                    all_keys.append(k)
        sections.append(" | ".join(all_keys))
        for row in bom:
            sections.append(" | ".join(row.get(k, "") for k in all_keys))

    # 3. Annotationen nach Layer
    texts_by_layer = _dxf_extract_texts_by_layer(doc)
    for layer, texts in sorted(texts_by_layer.items()):
        sections.append(f"\n--- ANNOTATIONEN (Layer: {layer}) ---")
        sections.extend(texts)

    # 4. Bemassungen
    dims = _dxf_extract_dimensions(doc)
    if dims:
        sections.append("\n--- BEMASSUNGEN ---")
        sections.extend(dims)

    # 5. Leader/Multileader
    leaders = _dxf_extract_leaders(doc)
    if leaders:
        sections.append("\n--- FERTIGUNGSHINWEISE ---")
        sections.extend(leaders)

    return "\n".join(sections)


def _load_ocp_model(path: str, format_name: str, reader_factory) -> str:
    """Gemeinsame Lade-Logik fuer OCP-basierte CAD-Formate (STEP, IGES).

    Args:
        path: Dateipfad.
        format_name: Format-Bezeichnung fuer Log-Meldungen (z.B. "STEP", "IGES").
        reader_factory: Callable das einen OCP-Reader erzeugt (z.B. STEPControl_Reader).
    """
    try:
        from OCP.Bnd import Bnd_Box
        from OCP.BRepBndLib import BRepBndLib
        from OCP.BRepGProp import BRepGProp
        from OCP.GProp import GProp_GProps
        from OCP.IFSelect import IFSelect_RetDone
    except ImportError:
        logger.warning("OCP (pythonOCC-core) not installed — skipping %s", path)
        return None

    reader = reader_factory()
    # Kein Context-Manager: pool.shutdown(wait=True) beim Verlassen wuerde
    # auf blockierte OCP-C-Threads warten und den Haupt-Worker haengen lassen.
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        try:
            status = pool.submit(reader.ReadFile, path).result(
                timeout=_OCP_READ_TIMEOUT_SECS,
            )
        except concurrent.futures.TimeoutError:
            logger.warning(
                "Timeout (%ds) reading %s file %s",
                _OCP_READ_TIMEOUT_SECS,
                format_name,
                path,
            )
            return None
    finally:
        # C-Extension-Threads koennen nicht abgebrochen werden; wait=False
        # laesst sie im Hintergrund ausklingen, unterbricht aber das Warten.
        pool.shutdown(wait=False, cancel_futures=True)
    if status != IFSelect_RetDone:
        logger.warning("Could not read %s file %s", format_name, path)
        return None

    reader.TransferRoots()
    shape = reader.OneShape()
    filename = Path(path).name

    sections = [f"=== CAD-MODELL: {filename} ==="]

    # Produktstruktur aus Transfer
    sections.append("\n--- PRODUKTSTRUKTUR ---")
    n_shapes = reader.NbShapes()
    if n_shapes > 1:
        sections.append(f"Baugruppe mit {n_shapes} Teilen")
        for i in range(1, n_shapes + 1):
            sections.append(f"  Teil {i}")
    else:
        sections.append("Einzelteil")

    # Bounding-Box
    try:
        bbox = Bnd_Box()
        BRepBndLib.Add_(shape, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        dx = abs(xmax - xmin)
        dy = abs(ymax - ymin)
        dz = abs(zmax - zmin)
        sections.append("\n--- GEOMETRIE ---")
        sections.append(f"Bounding-Box: {dx:.1f} x {dy:.1f} x {dz:.1f} mm")
    except Exception as exc:
        logger.debug("Bounding-Box-Extraktion fehlgeschlagen: %s", exc)

    # Volumen und Oberflaeche
    try:
        props = GProp_GProps()
        BRepGProp.VolumeProperties_(shape, props)
        volume = props.Mass()
        sprops = GProp_GProps()
        BRepGProp.SurfaceProperties_(shape, sprops)
        area = sprops.Mass()
        sections.append(f"Volumen: {volume:.1f} mm³")
        sections.append(f"Oberflaeche: {area:.1f} mm²")
    except Exception as exc:
        logger.debug("Volumen/Oberflaeche-Berechnung fehlgeschlagen: %s", exc)

    return "\n".join(sections)


def _load_step(path: str) -> str:
    """Load .step/.stp 3D CAD files — extract product structure and geometry."""
    try:
        from OCP.STEPControl import STEPControl_Reader
    except ImportError:
        logger.warning("OCP (pythonOCC-core) not installed — skipping %s", path)
        return None
    return _load_ocp_model(path, "STEP", STEPControl_Reader)


def _load_iges(path: str) -> str:
    """Load .iges/.igs CAD files — extract geometry summary."""
    try:
        from OCP.IGESControl import IGESControl_Reader
    except ImportError:
        logger.warning("OCP (pythonOCC-core) not installed — skipping %s", path)
        return None
    return _load_ocp_model(path, "IGES", IGESControl_Reader)
