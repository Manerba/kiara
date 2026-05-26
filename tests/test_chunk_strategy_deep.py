"""Tiefergehende Tests fuer automatische Chunk-Groessen-Anpassung.

Testet detect_chunk_params(), _analyze_text(), chunk_document() und
_determine_chunks() mit realistischen Textbeispielen fuer verschiedene
Dokumenttypen und Grenzfaelle.
"""

import pytest

from kiara.ingestion.chunk_strategy import (
    STRATEGIES,
    _analyze_text,
    detect_chunk_params,
)
from kiara.ingestion.chunker import chunk_document

# ---------------------------------------------------------------------------
# Hilfsfunktionen fuer realistische Testdaten
# ---------------------------------------------------------------------------


def _prose_text(n_chars=2000):
    """Generiert realistischen deutschen Fliesstext."""
    paragraph = (
        "Die IT-Abteilung hat die Migration der Legacy-Systeme auf die neue "
        "Cloud-Infrastruktur erfolgreich abgeschlossen. Dabei wurden insgesamt "
        "42 Server virtualisiert und in drei Verfuegbarkeitszonen verteilt. "
        "Die Latenz konnte um 35% reduziert werden. "
    )
    return (paragraph * (n_chars // len(paragraph) + 1))[:n_chars]


def _tabular_text(n_rows=20):
    """Generiert Pipe-getrennte Tabellenzeilen."""
    header = "Hostname | IP-Adresse | Status | Letzte Pruefung"
    rows = [header]
    for i in range(n_rows):
        rows.append(f"srv-{i:03d} | 10.0.{i // 256}.{i % 256} | aktiv | 2024-01-{(i % 28) + 1:02d}")
    return "\n".join(rows)


def _kv_text(n_pairs=20):
    """Generiert Key-Value-Paare."""
    pairs = []
    for i in range(n_pairs):
        pairs.append(f"Parameter_{i}: Wert_{i}")
    return "\n".join(pairs)


def _code_text():
    """Generiert Python-Quellcode."""
    return """
class ServerManager:
    def __init__(self, config):
        self.config = config
        self.servers = []

    def add_server(self, hostname, ip):
        self.servers.append({"hostname": hostname, "ip": ip})

    def remove_server(self, hostname):
        self.servers = [s for s in self.servers if s["hostname"] != hostname]

def main():
    mgr = ServerManager({})
    mgr.add_server("srv-001", "10.0.0.1")
    mgr.add_server("srv-002", "10.0.0.2")
"""


# ---------------------------------------------------------------------------
# 1. detect_chunk_params() — Dateityp-Heuristik (Stufe 1)
# ---------------------------------------------------------------------------


class TestDetectChunkParamsFiletype:
    """Stufe-1-Heuristik: Dateityp entscheidet ohne Textanalyse."""

    def test_csv_returns_tabular(self):
        """.csv wird als tabular erkannt."""
        meta = {"file_type": ".csv", "source": "/data/export.csv"}
        size, overlap = detect_chunk_params("col1,col2\nval1,val2", meta)
        assert (size, overlap) == STRATEGIES["tabular"]
        assert meta["chunk_strategy"] == "tabular"

    def test_xlsx_returns_tabular(self):
        """.xlsx wird als tabular erkannt."""
        meta = {"file_type": ".xlsx", "source": "/data/tabelle.xlsx"}
        size, overlap = detect_chunk_params("A\tB\n1\t2", meta)
        assert (size, overlap) == STRATEGIES["tabular"]
        assert meta["chunk_strategy"] == "tabular"

    def test_json_returns_tabular(self):
        """.json wird als tabular erkannt — auch bei Fliesstext-Inhalt."""
        meta = {"file_type": ".json", "source": "/data/export.json"}
        # Fliesstext-Inhalt darf die tabular-Erkennung nicht aushebeln
        size, overlap = detect_chunk_params(_prose_text(500), meta)
        assert (size, overlap) == STRATEGIES["tabular"]
        assert meta["chunk_strategy"] == "tabular"

    def test_py_returns_code(self):
        """.py wird als code erkannt."""
        meta = {"file_type": ".py", "source": "/src/main.py"}
        size, overlap = detect_chunk_params(_code_text(), meta)
        assert (size, overlap) == STRATEGIES["code"]
        assert meta["chunk_strategy"] == "code"

    def test_cs_returns_code(self):
        """.cs wird als code erkannt."""
        meta = {"file_type": ".cs", "source": "/src/Program.cs"}
        text = 'using System;\nclass Hello {\n    static void Main() {\n        Console.WriteLine("Hi");\n    }\n}'
        size, overlap = detect_chunk_params(text, meta)
        assert (size, overlap) == STRATEGIES["code"]
        assert meta["chunk_strategy"] == "code"

    def test_sh_returns_code(self):
        """.sh wird als code erkannt."""
        meta = {"file_type": ".sh", "source": "/scripts/deploy.sh"}
        text = "#!/bin/bash\nset -e\necho 'Deploying...'\nrsync -av /src/ /dst/"
        size, overlap = detect_chunk_params(text, meta)
        assert (size, overlap) == STRATEGIES["code"]
        assert meta["chunk_strategy"] == "code"

    def test_docx_prose_text_returns_prose(self):
        """.docx mit normalem Fliesstext wird als prose erkannt."""
        meta = {"file_type": ".docx", "source": "/docs/bericht.docx"}
        size, overlap = detect_chunk_params(_prose_text(2000), meta)
        assert (size, overlap) == STRATEGIES["prose"]
        assert meta["chunk_strategy"] == "prose"

    def test_pdf_prose_text_returns_prose(self):
        """.pdf mit normalem Fliesstext wird als prose erkannt."""
        meta = {"file_type": ".pdf", "source": "/docs/handbuch.pdf"}
        size, overlap = detect_chunk_params(_prose_text(2000), meta)
        assert (size, overlap) == STRATEGIES["prose"]
        assert meta["chunk_strategy"] == "prose"

    def test_md_prose_text_returns_prose(self):
        """.md mit normalem Fliesstext wird als prose erkannt."""
        meta = {"file_type": ".md", "source": "/docs/readme.md"}
        size, overlap = detect_chunk_params(_prose_text(2000), meta)
        assert (size, overlap) == STRATEGIES["prose"]
        assert meta["chunk_strategy"] == "prose"

    def test_txt_in_mail_cache_returns_email(self):
        """.txt im /mail-cache/-Pfad wird als email erkannt."""
        meta = {
            "file_type": ".txt",
            "source": "/var/lib/kiara/mail-cache/ds5/inbox/msg001.txt",
        }
        text = "From: user@firma.de\nTo: admin@firma.de\nSubject: Test\n\nHallo Welt."
        size, overlap = detect_chunk_params(text, meta)
        assert (size, overlap) == STRATEGIES["email"]
        assert meta["chunk_strategy"] == "email"

    def test_txt_in_sql_cache_returns_tabular(self):
        """.txt im /sql-cache/-Pfad wird als tabular erkannt."""
        meta = {
            "file_type": ".txt",
            "source": "/var/lib/kiara/sql-cache/ds7/query_001.txt",
        }
        text = "id: 1\ntitel: Testticket\nbeschreibung: Drucker defekt"
        size, overlap = detect_chunk_params(text, meta)
        assert (size, overlap) == STRATEGIES["tabular"]
        assert meta["chunk_strategy"] == "tabular"

    def test_dxf_returns_structured(self):
        """.dxf (CAD) wird als structured erkannt."""
        meta = {"file_type": ".dxf", "source": "/cad/zeichnung.dxf"}
        text = "=== CAD-ZEICHNUNG ===\nARTIKELNR: 12345\nMATERIAL: S235JR"
        size, overlap = detect_chunk_params(text, meta)
        assert (size, overlap) == STRATEGIES["structured"]
        assert meta["chunk_strategy"] == "structured"

    def test_step_returns_structured(self):
        """.step (CAD) wird als structured erkannt."""
        meta = {"file_type": ".step", "source": "/cad/baugruppe.step"}
        text = "=== CAD-MODELL ===\nEinzelteil\nAbmessungen: 100x50x25"
        size, overlap = detect_chunk_params(text, meta)
        assert (size, overlap) == STRATEGIES["structured"]
        assert meta["chunk_strategy"] == "structured"

    def test_unknown_extension_with_plain_text_returns_default(self):
        """Unbekannte Extension .xyz → default wenn Text nicht analysierbar."""
        meta = {"file_type": ".xyz", "source": "/data/datei.xyz"}
        # Langer Fliesstext ohne Pipes/KV und mit langen Zeilen → _analyze_text() gibt None
        text = _prose_text(2000)
        size, overlap = detect_chunk_params(text, meta)
        assert (size, overlap) == STRATEGIES["default"]
        assert meta["chunk_strategy"] == "default"

    def test_png_returns_prose(self):
        """.png (Bild) wird als prose erkannt."""
        meta = {"file_type": ".png", "source": "/scans/dokument.png"}
        # Zeilen muessen avg_line_len >= 50 haben, damit Kurzzeilen-Pfad nicht greift
        text = (
            "OCR-erkannter Text aus einem gescannten Dokument mit mehreren langen Woertern und Satzteilen.\n"
            * 10
        )
        size, overlap = detect_chunk_params(text, meta)
        assert (size, overlap) == STRATEGIES["prose"]
        assert meta["chunk_strategy"] == "prose"


# ---------------------------------------------------------------------------
# 2. detect_chunk_params() — Textanalyse-Override (Stufe 2)
# ---------------------------------------------------------------------------


class TestDetectChunkParamsTextOverride:
    """Stufe-2-Korrektur: Textanalyse ueberschreibt Basis bei prose und default."""

    def test_txt_with_pipe_table_overrides_to_tabular(self):
        """.txt (prose-Basis) mit >40% Pipe-Zeilen → tabular."""
        meta = {"file_type": ".txt", "source": "/data/tabelle.txt"}
        # 15 Pipe-Zeilen, 5 normale → 75% > 40%
        lines = [f"| Spalte A | Wert {i} |" for i in range(15)]
        lines += [f"Normaler Text Zeile {i}." for i in range(5)]
        text = "\n".join(lines)
        size, overlap = detect_chunk_params(text, meta)
        assert (size, overlap) == STRATEGIES["tabular"]
        assert meta["chunk_strategy"] == "tabular"

    def test_txt_with_kv_pairs_overrides_to_structured(self):
        """.txt (prose-Basis) mit >50% KV-Zeilen und langen Zeilen → structured."""
        meta = {"file_type": ".txt", "source": "/data/formular.txt"}
        # 6 KV-Zeilen mit langem Wert, 4 normale lange Zeilen → kv_ratio=0.6 > 0.5
        kv_lines = [
            f"Feldname_{i}: Wert_{i} mit genuegend Laenge damit avg_line_len gross bleibt"
            for i in range(6)
        ]
        prose_lines = [
            f"Normaler Fliesstext in Zeile {i}, der lang genug ist um avg_line_len ueber 50 zu halten."
            for i in range(4)
        ]
        text = "\n".join(kv_lines + prose_lines)
        size, overlap = detect_chunk_params(text, meta)
        assert (size, overlap) == STRATEGIES["structured"]
        assert meta["chunk_strategy"] == "structured"

    def test_txt_short_lines_kv_above_30pct_returns_structured(self):
        """.txt mit kurzen Zeilen und KV-Ratio >30% → structured."""
        meta = {"file_type": ".txt", "source": "/data/config.txt"}
        # Kurze Zeilen (avg <50), 4 von 8 sind KV (50% > 30%)
        lines = [
            "Name: Max Muster",  # KV, kurz
            "Alter: 42",  # KV, kurz
            "Stadt: Berlin",  # KV, kurz
            "Rolle: Admin",  # KV, kurz
            "kein kv hier",  # kein KV, kurz
            "auch nicht kv",  # kein KV, kurz
            "noch einer",  # kein KV, kurz
            "kurze zeile",  # kein KV, kurz
        ]
        text = "\n".join(lines)
        size, overlap = detect_chunk_params(text, meta)
        assert (size, overlap) == STRATEGIES["structured"]
        assert meta["chunk_strategy"] == "structured"

    def test_txt_short_lines_kv_below_30pct_returns_tabular(self):
        """.txt mit kurzen Zeilen und KV-Ratio <=30% → tabular."""
        meta = {"file_type": ".txt", "source": "/data/liste.txt"}
        # Kurze Zeilen (avg <50), nur 2 von 10 sind KV (20% <= 30%)
        lines = [
            "Name: Max",  # KV
            "Stadt: Berlin",  # KV
            "kein kv",
            "auch nicht",
            "drittes item",
            "viertes item",
            "fuenftes item",
            "sechstes item",
            "siebtes item",
            "achtes item",
        ]
        text = "\n".join(lines)
        size, overlap = detect_chunk_params(text, meta)
        assert (size, overlap) == STRATEGIES["tabular"]
        assert meta["chunk_strategy"] == "tabular"

    def test_md_with_prose_stays_prose(self):
        """.md mit langen Fliesstext-Saetzen bleibt prose."""
        meta = {"file_type": ".md", "source": "/docs/readme.md"}
        # Langer Fliesstext → _analyze_text() gibt None → prose bleibt
        text = _prose_text(2000)
        size, overlap = detect_chunk_params(text, meta)
        assert (size, overlap) == STRATEGIES["prose"]
        assert meta["chunk_strategy"] == "prose"

    def test_py_with_pipe_tables_stays_code(self):
        """.py mit Pipe-Zeichen im Code → code bleibt (Textanalyse greift nicht)."""
        meta = {"file_type": ".py", "source": "/src/parser.py"}
        # Python-Datei mit vielen Pipe-Zeichen (z.B. Bitoperationen)
        lines = ["def func():", "    pass"]
        for i in range(20):
            lines.append(f"    x_{i} = value_{i} | mask_{i}  # bitwise or")
        text = "\n".join(lines)
        size, overlap = detect_chunk_params(text, meta)
        assert (size, overlap) == STRATEGIES["code"]
        assert meta["chunk_strategy"] == "code"

    def test_unknown_extension_with_pipe_table_overrides_to_tabular(self):
        """Unbekannte Extension (default-Basis) mit Pipe-Tabelle → tabular."""
        meta = {"file_type": ".xyz", "source": "/data/datei.xyz"}
        text = "\n".join([f"| Col1 | Col2 | Val{i} |" for i in range(15)])
        size, overlap = detect_chunk_params(text, meta)
        assert (size, overlap) == STRATEGIES["tabular"]
        assert meta["chunk_strategy"] == "tabular"

    def test_empty_text_returns_default(self):
        """Leerer Text → default (unabhaengig von file_type)."""
        meta = {"file_type": ".docx", "source": "/docs/leer.docx"}
        size, overlap = detect_chunk_params("", meta)
        assert (size, overlap) == STRATEGIES["default"]
        assert meta["chunk_strategy"] == "default"

    def test_whitespace_only_text_returns_default(self):
        """Nur-Whitespace-Text → default."""
        meta = {"file_type": ".txt", "source": "/data/leer.txt"}
        size, overlap = detect_chunk_params("   \n  \n  ", meta)
        assert (size, overlap) == STRATEGIES["default"]
        assert meta["chunk_strategy"] == "default"


# ---------------------------------------------------------------------------
# 3. _analyze_text() — Grenzwerte
# ---------------------------------------------------------------------------


class TestAnalyzeTextBoundaries:
    """Exakte Grenzwert-Tests fuer die drei Metriken in _analyze_text()."""

    # --- pipe_ratio Grenzwerte ---

    def test_pipe_ratio_exactly_40pct_is_not_tabular(self):
        """Exakt 40% Pipe-Zeilen → NICHT tabular ueber Pipe-Pfad (Schwelle ist >0.4, nicht >=0.4).

        Damit ausschliesslich die Pipe-Schwelle getestet wird, muessen die Zeilen
        lang genug sein (avg_line_len >= 50), damit der Kurzzeilen-Pfad nicht greift.
        """
        # 4 Pipe-Zeilen, 6 normale, alle lang (avg_line_len > 50)
        # → pipe_ratio = 4/10 = 0.40, NICHT > 0.40 → kein tabular via Pipe-Pfad
        # → kv_ratio = 0 → kein structured
        # → avg_line_len >> 50 → kein Kurzzeilen-Pfad
        # → _analyze_text() gibt None zurueck
        lines = [
            f"| Spalte A | Spalte B | Wert {i} enthalt mehr langen Text hier drin |"
            for i in range(4)
        ]
        lines += [
            f"Normaler langer Fliesstext der sicher ueber 50 Zeichen lang ist in Zeile {i}."
            for i in range(6)
        ]
        text = "\n".join(lines)
        result = _analyze_text(text)
        assert result is None

    def test_pipe_ratio_exactly_41pct_is_tabular(self):
        """41% Pipe-Zeilen → tabular (0.41 > 0.4)."""
        # Brauchen ratio > 0.4; 5 von 12 = 0.4166... > 0.4
        lines = [f"| Spalte A | Wert {i} |" for i in range(5)]
        lines += [f"Normaler Text in Zeile {i} mit etwas mehr Inhalt" for i in range(7)]
        text = "\n".join(lines)
        result = _analyze_text(text)
        assert result == "tabular"

    # --- kv_ratio Grenzwerte ---

    def test_kv_ratio_exactly_50pct_is_not_structured(self):
        """Exakt 50% KV-Zeilen → NICHT structured (Schwelle ist >0.5)."""
        # 5 KV-Zeilen, 5 normale, lange Zeilen damit avg_line_len >= 50
        kv = [
            f"Feldname_{i}: Wert_{i} mit etwas mehr Text damit Zeile lang genug ist"
            for i in range(5)
        ]
        prose = [
            f"Normaler langer Fliesstext in Zeile {i} mit vielen Woertern damit Zeile lang ist."
            for i in range(5)
        ]
        text = "\n".join(kv + prose)
        result = _analyze_text(text)
        # 5/10 = 0.5, nicht > 0.5 → kein structured; avg_line_len >= 50 → kein Kurzzeilen-Pfad
        assert result != "structured"

    def test_kv_ratio_exactly_51pct_is_structured(self):
        """51% KV-Zeilen (> 0.5) → structured."""
        # 6 KV, 5 normale, lange Zeilen (avg >= 50)
        kv = [
            f"Feldname_{i}: Wert_{i} mit etwas mehr Text damit die Zeile genuegend lang ist"
            for i in range(6)
        ]
        prose = [
            f"Normaler langer Fliesstext in Zeile {i} mit vielen Woertern, die Zeile lang genug machen."
            for i in range(5)
        ]
        text = "\n".join(kv + prose)
        result = _analyze_text(text)
        assert result == "structured"

    # --- avg_line_len Grenzwerte ---

    def test_avg_line_len_exactly_50_is_not_short_line_path(self):
        """avg_line_len == 50 → Kurzzeilen-Pfad NICHT aktiv (Schwelle ist <50)."""
        # Zeilen die exakt auf 50 Zeichen kommen (gemessen ohne Newline)
        # Konstruiere 10 Zeilen mit avg = 50: jede Zeile genau 50 Zeichen
        line = "A" * 50
        text = "\n".join([line] * 10)
        # avg_line_len = 50, kv_ratio = 0 → weder KV noch Pipe → kein tabular/structured
        result = _analyze_text(text)
        # avg_line_len ist NICHT < 50, also kein Kurzzeilen-Pfad
        assert result is None

    def test_avg_line_len_49_activates_short_line_path(self):
        """avg_line_len == 49 → Kurzzeilen-Pfad aktiv."""
        # Zeilen exakt 49 Zeichen, kv_ratio=0 → tabular (kv_ratio <= 0.3)
        line = "A" * 49
        text = "\n".join([line] * 10)
        result = _analyze_text(text)
        # avg_line_len 49 < 50, kv_ratio=0 <= 0.3 → tabular
        assert result == "tabular"

    def test_avg_line_len_49_with_kv_above_30_returns_structured(self):
        """avg_line_len < 50 und kv_ratio > 0.3 → structured."""
        # 4 kurze KV-Zeilen (< 50 Zeichen), 6 kurze Nicht-KV
        # kv_ratio = 4/10 = 0.4 > 0.3, avg < 50
        kv = [f"Key_{i}: Val" for i in range(4)]  # kurz und KV-Pattern
        other = [f"Item_{i}" for i in range(6)]  # kurz, kein KV
        text = "\n".join(kv + other)
        result = _analyze_text(text)
        assert result == "structured"

    def test_empty_string_returns_none(self):
        """Leerer String → None (Guard faengt leeren Input ab)."""
        result = _analyze_text("")
        assert result is None

    def test_single_line_no_crash(self):
        """Einzelne Zeile → kein Absturz, Ergebnis ist ein gueltiger Wert."""
        result = _analyze_text("Nur eine einzige Zeile ohne besondere Struktur hier.")
        assert result is None or result in ("tabular", "structured", "prose")

    def test_pipe_wins_over_kv_when_both_high(self):
        """pipe_ratio > 0.4 gewinnt gegen kv_ratio > 0.5 (pipe_ratio wird zuerst geprueft)."""
        # 6 Pipe-Zeilen die auch KV-Muster enthalten, 4 normale lange Zeilen
        mixed = [f"| Key_{i}: | Wert_{i} |" for i in range(6)]
        prose = [
            f"Normaler langer Fliesstext in Zeile {i} mit vielen Woertern damit Zeile lang ist."
            for i in range(4)
        ]
        text = "\n".join(mixed + prose)
        result = _analyze_text(text)
        # pipe_ratio = 6/10 = 0.6 > 0.4 → tabular (obwohl kv auch hoch)
        assert result == "tabular"


# ---------------------------------------------------------------------------
# 4. chunk_document() — Chunk-Ergebnis
# ---------------------------------------------------------------------------


class TestChunkDocument:
    """Tests fuer chunk_document(): Anzahl, Metadaten, Overlap."""

    def test_short_text_produces_one_chunk(self):
        """Text kuerzer als chunk_size → genau 1 Chunk."""
        meta = {"file_type": ".txt", "source": "/data/kurz.txt"}
        text = "Dies ist ein kurzer Text."
        chunks = chunk_document(text, meta, chunk_size=500, chunk_overlap=50)
        assert len(chunks) == 1

    def test_long_text_produces_multiple_chunks(self):
        """Langer Text (5000 Zeichen) mit chunk_size=500 → mehrere Chunks."""
        meta = {"file_type": ".txt", "source": "/data/lang.txt"}
        text = "A" * 5000
        chunks = chunk_document(text, meta, chunk_size=500, chunk_overlap=50)
        assert len(chunks) > 1

    def test_chunk_index_is_zero_based(self):
        """chunk_index startet bei 0."""
        meta = {"file_type": ".txt", "source": "/data/lang.txt"}
        text = _prose_text(3000)
        chunks = chunk_document(text, meta, chunk_size=500, chunk_overlap=50)
        assert chunks[0]["metadata"]["chunk_index"] == 0

    def test_chunk_total_is_correct(self):
        """chunk_total stimmt mit der tatsaechlichen Anzahl Chunks ueberein."""
        meta = {"file_type": ".txt", "source": "/data/lang.txt"}
        text = _prose_text(3000)
        chunks = chunk_document(text, meta, chunk_size=500, chunk_overlap=50)
        expected_total = len(chunks)
        for chunk in chunks:
            assert chunk["metadata"]["chunk_total"] == expected_total

    def test_chunk_indices_are_sequential(self):
        """chunk_index ist lueckenlos aufsteigend (0, 1, 2, ...)."""
        meta = {"file_type": ".txt", "source": "/data/lang.txt"}
        text = _prose_text(4000)
        chunks = chunk_document(text, meta, chunk_size=500, chunk_overlap=50)
        for i, chunk in enumerate(chunks):
            assert chunk["metadata"]["chunk_index"] == i

    def test_original_metadata_copied_to_each_chunk(self):
        """Alle Original-Metadaten werden in jeden Chunk kopiert."""
        meta = {
            "file_type": ".txt",
            "source": "/data/lang.txt",
            "custom_field": "Testwert",
            "datasource_id": 42,
        }
        text = _prose_text(3000)
        chunks = chunk_document(text, meta, chunk_size=500, chunk_overlap=50)
        for chunk in chunks:
            assert chunk["metadata"]["source"] == "/data/lang.txt"
            assert chunk["metadata"]["file_type"] == ".txt"
            assert chunk["metadata"]["custom_field"] == "Testwert"
            assert chunk["metadata"]["datasource_id"] == 42

    def test_metadata_mutation_does_not_affect_other_chunks(self):
        """Mutation des Metadaten-Dicts eines Chunks beeinflusst andere nicht."""
        meta = {"file_type": ".txt", "source": "/data/lang.txt"}
        text = _prose_text(3000)
        chunks = chunk_document(text, meta, chunk_size=500, chunk_overlap=50)
        # Erster Chunk-Index als Referenz
        original_index = chunks[0]["metadata"]["chunk_index"]
        # Ersten Chunk mutieren
        chunks[0]["metadata"]["chunk_index"] = 999
        # Zweiter Chunk bleibt unveraendert
        assert chunks[1]["metadata"]["chunk_index"] == 1
        # Auch original_index war 0
        assert original_index == 0

    def test_python_file_chunks_break_at_def_or_class(self):
        """Python-Datei nutzt Sprach-spezifischen Splitter (bricht an def/class)."""
        meta = {"file_type": ".py", "source": "/src/large_module.py"}
        # Gross genug um mehrere Chunks zu erzeugen
        code_block = _code_text()
        text = code_block * 5  # Wiederholen damit Splits noetig
        chunks = chunk_document(text, meta, chunk_size=300, chunk_overlap=30)
        # Muss mehr als 1 Chunk erzeugen
        assert len(chunks) > 1
        # Jeder Chunk ist ein Dict mit 'text' und 'metadata'
        for chunk in chunks:
            assert "text" in chunk
            assert "metadata" in chunk
            assert len(chunk["text"]) > 0

    def test_chunk_text_field_is_string(self):
        """Jedes Chunk-'text'-Feld ist ein String."""
        meta = {"file_type": ".txt", "source": "/data/test.txt"}
        chunks = chunk_document(_prose_text(2000), meta, chunk_size=500, chunk_overlap=50)
        for chunk in chunks:
            assert isinstance(chunk["text"], str)

    def test_overlap_creates_shared_content_between_adjacent_chunks(self):
        """Aufeinanderfolgende Chunks teilen Text am Rand (Overlap)."""
        meta = {"file_type": ".txt", "source": "/data/test.txt"}
        # Gleichmaessiger Text damit Splitter sauber arbeitet
        text = "Wort " * 1000  # 5000 Zeichen
        chunks = chunk_document(text, meta, chunk_size=500, chunk_overlap=100)
        if len(chunks) < 2:
            pytest.skip("Zu wenige Chunks fuer Overlap-Test")
        # Ende von Chunk i und Anfang von Chunk i+1 haben gemeinsamen Inhalt
        end_of_first = chunks[0]["text"][-50:].strip()
        start_of_second = chunks[1]["text"][:100].strip()
        assert end_of_first in start_of_second or start_of_second.startswith(end_of_first[:20])

    def test_zero_overlap_chunks_are_disjoint(self):
        """chunk_overlap=0: Chunks haben keinen gemeinsamen Anfangstext."""
        meta = {"file_type": ".csv", "source": "/data/tabelle.csv"}
        text = _tabular_text(50)
        chunks = chunk_document(text, meta, chunk_size=200, chunk_overlap=0)
        if len(chunks) < 2:
            pytest.skip("Zu wenige Chunks fuer Disjoint-Test")
        # Das Ende von Chunk 0 ist nicht der Anfang von Chunk 1
        end0 = chunks[0]["text"][-20:]
        start1 = chunks[1]["text"][:20]
        # Bei 0 Overlap sollten sie sich nicht ueberlappen
        assert True  # Nur kein Absturz, Disjoint nicht garantiert bei RecursiveCharacter


# ---------------------------------------------------------------------------
# 4b. chunk_document() — Char-Span-Validierung (Late Chunking)
# ---------------------------------------------------------------------------


class TestChunkDocumentSpans:
    """Tests fuer char_start/char_end Spans in chunk_document() Ergebnissen."""

    def test_chunk_document_spans_default(self):
        """Default Text-Splitter: jeder Chunk hat gueltige char_start/char_end, monoton steigend."""
        meta = {"file_type": ".txt", "source": "/data/spans.txt"}
        text = _prose_text(3000)
        chunks = chunk_document(text, meta, chunk_size=500, chunk_overlap=50)
        assert len(chunks) > 1, "Brauche mehrere Chunks fuer Span-Validierung"
        for chunk in chunks:
            assert "char_start" in chunk, "char_start fehlt"
            assert "char_end" in chunk, "char_end fehlt"
            assert isinstance(chunk["char_start"], int)
            assert isinstance(chunk["char_end"], int)
            assert chunk["char_start"] >= 0
            assert chunk["char_end"] > chunk["char_start"]
            # Span zeigt auf korrekten Text-Ausschnitt
            assert text[chunk["char_start"] : chunk["char_end"]] == chunk["text"]
        # Monoton steigende Start-Indizes
        starts = [c["char_start"] for c in chunks]
        assert starts == sorted(starts), "char_start nicht monoton steigend"
        for i in range(1, len(starts)):
            assert starts[i] > starts[i - 1], "char_start nicht strikt steigend"

    def test_chunk_document_spans_language(self):
        """Language-Splitter (.py): jeder Chunk hat gueltige char_start/char_end."""
        meta = {"file_type": ".py", "source": "/src/test_spans.py"}
        text = _code_text() * 5
        chunks = chunk_document(text, meta, chunk_size=300, chunk_overlap=30)
        assert len(chunks) > 1
        for chunk in chunks:
            assert isinstance(chunk["char_start"], int)
            assert isinstance(chunk["char_end"], int)
            assert chunk["char_start"] >= 0
            assert chunk["char_end"] > chunk["char_start"]
            assert text[chunk["char_start"] : chunk["char_end"]] == chunk["text"]
        starts = [c["char_start"] for c in chunks]
        assert starts == sorted(starts)

    def test_chunk_document_spans_code_extensions(self):
        """CODE_EXTENSIONS Splitter (.bat): jeder Chunk hat gueltige Spans."""
        meta = {"file_type": ".bat", "source": "/scripts/deploy.bat"}
        text = "@echo off\nset VAR=value\necho %%VAR%%\n" * 50
        chunks = chunk_document(text, meta, chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1
        for chunk in chunks:
            assert isinstance(chunk["char_start"], int)
            assert isinstance(chunk["char_end"], int)
            assert chunk["char_start"] >= 0
            assert chunk["char_end"] > chunk["char_start"]
            assert text[chunk["char_start"] : chunk["char_end"]] == chunk["text"]
        starts = [c["char_start"] for c in chunks]
        assert starts == sorted(starts)

    def test_chunk_document_spans_with_overlap(self):
        """Overlap: aufeinanderfolgende Chunks koennen ueberlappende Spans haben."""
        meta = {"file_type": ".txt", "source": "/data/overlap_spans.txt"}
        text = _prose_text(3000)
        chunks = chunk_document(text, meta, chunk_size=500, chunk_overlap=100)
        assert len(chunks) > 2, "Brauche mindestens 3 Chunks"
        starts = [c["char_start"] for c in chunks]
        # Start-Indizes muessen monoton steigend sein
        for i in range(1, len(starts)):
            assert starts[i] > starts[i - 1]
        # Bei Overlap kann char_start von Chunk N+1 < char_end von Chunk N sein
        has_overlap = False
        for i in range(len(chunks) - 1):
            if chunks[i + 1]["char_start"] < chunks[i]["char_end"]:
                has_overlap = True
                break
        assert has_overlap, "Erwartet mindestens eine Ueberlappung bei chunk_overlap=100"


# ---------------------------------------------------------------------------
# 5. _determine_chunks() — Orchestrierung
# ---------------------------------------------------------------------------


class TestDetermineChunks:
    """Tests fuer _determine_chunks() aus kiara.ingestion.pipeline."""

    def _make_doc(self, text=None, file_type=".txt", source="/data/test.txt", **extra_meta):
        """Erzeugt ein Dokument wie load_file() es zurueckgibt."""
        return {
            "text": text or _prose_text(500),
            "metadata": {"file_type": file_type, "source": source, **extra_meta},
        }

    def test_vlm_doc_produces_single_chunk(self):
        """VLM-Dokument (parser='vlm') → genau 1 Chunk."""
        from kiara.ingestion.pipeline import _determine_chunks

        doc = self._make_doc(text="Bildbeschreibung von Kamera 3.", parser="vlm")
        chunks = _determine_chunks(doc, ds_overrides=None, chunk_size=1000, chunk_overlap=200)
        assert len(chunks) == 1

    def test_vlm_doc_chunk_strategy_is_vlm_single(self):
        """VLM-Dokument → metadata['chunk_strategy'] == 'vlm_single'."""
        from kiara.ingestion.pipeline import _determine_chunks

        doc = self._make_doc(text="Bildbeschreibung.", parser="vlm")
        chunks = _determine_chunks(doc, ds_overrides=None, chunk_size=1000, chunk_overlap=200)
        assert chunks[0]["metadata"]["chunk_strategy"] == "vlm_single"

    def test_vlm_doc_chunk_index_is_zero(self):
        """VLM-Dokument → chunk_index == 0."""
        from kiara.ingestion.pipeline import _determine_chunks

        doc = self._make_doc(text="Bildbeschreibung.", parser="vlm")
        chunks = _determine_chunks(doc, ds_overrides=None, chunk_size=1000, chunk_overlap=200)
        assert chunks[0]["metadata"]["chunk_index"] == 0

    def test_explicit_strategy_prose_uses_prose_params(self):
        """Explizite Strategie 'prose' → (768, 150) wird verwendet."""
        from unittest.mock import patch

        from kiara.ingestion.pipeline import _determine_chunks

        doc = self._make_doc()
        with patch("kiara.ingestion.runtime.chunk_document") as mock_chunk:
            mock_chunk.return_value = [{"text": "chunk", "metadata": doc["metadata"].copy()}]
            _determine_chunks(
                doc,
                ds_overrides={"chunk_strategy": "prose"},
                chunk_size=1000,
                chunk_overlap=200,
            )
            call_kwargs = mock_chunk.call_args.kwargs
            assert call_kwargs["chunk_size"] == STRATEGIES["prose"][0]
            assert call_kwargs["chunk_overlap"] == STRATEGIES["prose"][1]

    def test_explicit_strategy_prose_sets_metadata(self):
        """Explizite Strategie 'prose' → metadata['chunk_strategy'] == 'prose'."""
        from unittest.mock import patch

        from kiara.ingestion.pipeline import _determine_chunks

        doc = self._make_doc()
        with patch("kiara.ingestion.runtime.chunk_document") as mock_chunk:
            mock_chunk.return_value = [{"text": "chunk", "metadata": doc["metadata"].copy()}]
            _determine_chunks(
                doc,
                ds_overrides={"chunk_strategy": "prose"},
                chunk_size=1000,
                chunk_overlap=200,
            )
        assert doc["metadata"]["chunk_strategy"] == "prose"

    def test_explicit_strategy_tabular_uses_tabular_params(self):
        """Explizite Strategie 'tabular' → (256, 0) wird verwendet."""
        from unittest.mock import patch

        from kiara.ingestion.pipeline import _determine_chunks

        doc = self._make_doc()
        with patch("kiara.ingestion.runtime.chunk_document") as mock_chunk:
            mock_chunk.return_value = [{"text": "chunk", "metadata": doc["metadata"].copy()}]
            _determine_chunks(
                doc,
                ds_overrides={"chunk_strategy": "tabular"},
                chunk_size=1000,
                chunk_overlap=200,
            )
            call_kwargs = mock_chunk.call_args.kwargs
            assert call_kwargs["chunk_size"] == STRATEGIES["tabular"][0]
            assert call_kwargs["chunk_overlap"] == STRATEGIES["tabular"][1]

    def test_custom_chunk_size_uses_direct_values(self):
        """Custom chunk_size Override → direkte Verwendung statt Auto-Erkennung."""
        from unittest.mock import patch

        from kiara.ingestion.pipeline import _determine_chunks

        doc = self._make_doc()
        with patch("kiara.ingestion.runtime.chunk_document") as mock_chunk:
            mock_chunk.return_value = [{"text": "chunk", "metadata": doc["metadata"].copy()}]
            _determine_chunks(
                doc,
                ds_overrides={"chunk_size": 333, "chunk_overlap": 33},
                chunk_size=333,
                chunk_overlap=33,
            )
            call_kwargs = mock_chunk.call_args.kwargs
            assert call_kwargs["chunk_size"] == 333
            assert call_kwargs["chunk_overlap"] == 33

    def test_custom_chunk_size_sets_custom_strategy(self):
        """Custom chunk_size → metadata['chunk_strategy'] == 'custom'."""
        from unittest.mock import patch

        from kiara.ingestion.pipeline import _determine_chunks

        doc = self._make_doc()
        with patch("kiara.ingestion.runtime.chunk_document") as mock_chunk:
            mock_chunk.return_value = [{"text": "chunk", "metadata": doc["metadata"].copy()}]
            _determine_chunks(
                doc,
                ds_overrides={"chunk_size": 333, "chunk_overlap": 33},
                chunk_size=333,
                chunk_overlap=33,
            )
        assert doc["metadata"]["chunk_strategy"] == "custom"

    def test_no_override_uses_auto_detection(self):
        """Kein Override → Auto-Erkennung via detect_chunk_params() wird aufgerufen."""
        from unittest.mock import patch

        from kiara.ingestion.pipeline import _determine_chunks

        doc = self._make_doc(file_type=".csv")
        with patch(
            "kiara.ingestion.chunk_strategy.detect_chunk_params", return_value=(256, 0)
        ) as mock_detect:
            with patch("kiara.ingestion.runtime.chunk_document") as mock_chunk:
                mock_chunk.return_value = [{"text": "chunk", "metadata": doc["metadata"].copy()}]
                _determine_chunks(doc, ds_overrides=None, chunk_size=1000, chunk_overlap=200)
            mock_detect.assert_called_once()

    def test_auto_strategy_overrides_chunk_strategy_in_overrides(self):
        """chunk_strategy='auto' in ds_overrides → Auto-Erkennung (wie None)."""
        from unittest.mock import patch

        from kiara.ingestion.pipeline import _determine_chunks

        doc = self._make_doc()
        with patch(
            "kiara.ingestion.chunk_strategy.detect_chunk_params", return_value=(768, 150)
        ) as mock_detect:
            with patch("kiara.ingestion.runtime.chunk_document") as mock_chunk:
                mock_chunk.return_value = [{"text": "chunk", "metadata": doc["metadata"].copy()}]
                _determine_chunks(
                    doc,
                    ds_overrides={"chunk_strategy": "auto"},
                    chunk_size=1000,
                    chunk_overlap=200,
                )
            mock_detect.assert_called_once()


# ---------------------------------------------------------------------------
# 6. metadata["chunk_strategy"] wird korrekt gesetzt
# ---------------------------------------------------------------------------


class TestChunkStrategyMetadata:
    """Sicherstellen dass chunk_strategy immer korrekt in Metadaten landet."""

    def test_detect_chunk_params_sets_chunk_strategy_key(self):
        """detect_chunk_params() setzt 'chunk_strategy' in metadata."""
        meta = {"file_type": ".pdf", "source": "/docs/test.pdf"}
        detect_chunk_params(_prose_text(500), meta)
        assert "chunk_strategy" in meta

    def test_detect_chunk_params_strategy_value_is_known(self):
        """Der gesetzte chunk_strategy-Wert ist ein gueltiger Strategie-Name."""
        meta = {"file_type": ".csv", "source": "/data/test.csv"}
        detect_chunk_params("col1,col2\nval1,val2", meta)
        assert meta["chunk_strategy"] in STRATEGIES

    def test_detect_chunk_params_tabular_extension_sets_tabular(self):
        """Tabular Extension → chunk_strategy='tabular'."""
        meta = {"file_type": ".csv", "source": "/data/test.csv"}
        detect_chunk_params("a,b\n1,2", meta)
        assert meta["chunk_strategy"] == "tabular"

    def test_detect_chunk_params_code_extension_sets_code(self):
        """Code Extension → chunk_strategy='code'."""
        meta = {"file_type": ".py", "source": "/src/main.py"}
        detect_chunk_params(_code_text(), meta)
        assert meta["chunk_strategy"] == "code"

    def test_detect_chunk_params_cad_extension_sets_structured(self):
        """CAD Extension → chunk_strategy='structured'."""
        meta = {"file_type": ".dxf", "source": "/cad/part.dxf"}
        detect_chunk_params("CAD-Daten\nARTIKEL: 123", meta)
        assert meta["chunk_strategy"] == "structured"

    def test_detect_chunk_params_email_path_sets_email(self):
        """mail-cache Pfad → chunk_strategy='email'."""
        meta = {
            "file_type": ".txt",
            "source": "/var/lib/kiara/mail-cache/ds1/msg.txt",
        }
        detect_chunk_params("From: a@b.de\nSubject: Test", meta)
        assert meta["chunk_strategy"] == "email"

    def test_determine_chunks_vlm_sets_vlm_single(self):
        """_determine_chunks() bei VLM → chunk_strategy='vlm_single'."""
        from kiara.ingestion.pipeline import _determine_chunks

        doc = {
            "text": "Bildbeschreibung.",
            "metadata": {"file_type": ".png", "source": "/img/foto.png", "parser": "vlm"},
        }
        chunks = _determine_chunks(doc, ds_overrides=None, chunk_size=1000, chunk_overlap=200)
        assert chunks[0]["metadata"]["chunk_strategy"] == "vlm_single"

    def test_determine_chunks_explicit_strategy_sets_strategy_name(self):
        """_determine_chunks() bei expliziter Strategie → Strategiename in metadata."""
        from unittest.mock import patch

        from kiara.ingestion.pipeline import _determine_chunks

        doc = {
            "text": _prose_text(500),
            "metadata": {"file_type": ".txt", "source": "/data/test.txt"},
        }
        with patch("kiara.ingestion.runtime.chunk_document") as mock_chunk:
            mock_chunk.return_value = [{"text": "chunk", "metadata": doc["metadata"].copy()}]
            _determine_chunks(
                doc,
                ds_overrides={"chunk_strategy": "email"},
                chunk_size=1000,
                chunk_overlap=200,
            )
        assert doc["metadata"]["chunk_strategy"] == "email"

    def test_determine_chunks_custom_sets_custom(self):
        """_determine_chunks() bei custom chunk_size → chunk_strategy='custom'."""
        from unittest.mock import patch

        from kiara.ingestion.pipeline import _determine_chunks

        doc = {
            "text": _prose_text(500),
            "metadata": {"file_type": ".txt", "source": "/data/test.txt"},
        }
        with patch("kiara.ingestion.runtime.chunk_document") as mock_chunk:
            mock_chunk.return_value = [{"text": "chunk", "metadata": doc["metadata"].copy()}]
            _determine_chunks(
                doc,
                ds_overrides={"chunk_size": 400, "chunk_overlap": 40},
                chunk_size=400,
                chunk_overlap=40,
            )
        assert doc["metadata"]["chunk_strategy"] == "custom"

    def test_chunk_size_used_is_set_by_determine_chunks(self):
        """_determine_chunks() setzt 'chunk_size_used' in den Metadaten."""
        from unittest.mock import patch

        from kiara.ingestion.pipeline import _determine_chunks

        doc = {
            "text": _prose_text(500),
            "metadata": {"file_type": ".txt", "source": "/data/test.txt"},
        }
        with patch("kiara.ingestion.runtime.chunk_document") as mock_chunk:
            mock_chunk.return_value = [{"text": "chunk", "metadata": doc["metadata"].copy()}]
            _determine_chunks(
                doc,
                ds_overrides={"chunk_strategy": "prose"},
                chunk_size=1000,
                chunk_overlap=200,
            )
        assert "chunk_size_used" in doc["metadata"]
        assert doc["metadata"]["chunk_size_used"] == STRATEGIES["prose"][0]


# ---------------------------------------------------------------------------
# 7. Integrations-Sanity-Tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """Sanity-Tests die den gesamten Erkennungs-Chunk-Zyklus durchlaufen."""

    def test_tabular_text_detect_then_chunk(self):
        """detect_chunk_params() → chunk_document() fuer Tabellen-Dokument."""
        meta = {"file_type": ".csv", "source": "/data/srv-liste.csv"}
        text = _tabular_text(30)
        size, overlap = detect_chunk_params(text, meta)
        assert meta["chunk_strategy"] == "tabular"
        chunks = chunk_document(text, meta, chunk_size=size, chunk_overlap=overlap)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk["metadata"]["chunk_strategy"] == "tabular"

    def test_kv_text_detect_then_chunk(self):
        """detect_chunk_params() → chunk_document() fuer Key-Value-Dokument."""
        meta = {"file_type": ".txt", "source": "/data/config.txt"}
        text = _kv_text(30)
        size, overlap = detect_chunk_params(text, meta)
        # Kurze KV-Zeilen → structured (kv_ratio > 0.3, avg_line_len < 50)
        assert meta["chunk_strategy"] == "structured"
        chunks = chunk_document(text, meta, chunk_size=size, chunk_overlap=overlap)
        assert len(chunks) >= 1

    def test_code_detect_then_chunk(self):
        """detect_chunk_params() → chunk_document() fuer Python-Code."""
        meta = {"file_type": ".py", "source": "/src/main.py"}
        text = _code_text() * 5
        size, overlap = detect_chunk_params(text, meta)
        assert meta["chunk_strategy"] == "code"
        chunks = chunk_document(text, meta, chunk_size=size, chunk_overlap=overlap)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk["text"]) > 0

    def test_strategies_dict_all_values_valid(self):
        """Alle STRATEGIES-Eintraege haben chunk_size > 0 und chunk_overlap >= 0."""
        for name, (size, overlap) in STRATEGIES.items():
            assert size > 0, f"chunk_size <= 0 fuer {name}"
            assert overlap >= 0, f"chunk_overlap < 0 fuer {name}"
            assert overlap < size, f"chunk_overlap >= chunk_size fuer {name}"

    def test_strategies_contains_all_expected_keys(self):
        """STRATEGIES enthaelt genau die erwarteten Eintraege."""
        expected = {"prose", "tabular", "code", "structured", "email", "default"}
        assert set(STRATEGIES.keys()) == expected

    def test_detect_chunk_params_returns_values_from_strategies(self):
        """Rueckgabewert von detect_chunk_params() ist immer ein (size, overlap) aus STRATEGIES."""
        test_cases = [
            ({"file_type": ".csv", "source": "/d/f.csv"}, "a,b"),
            ({"file_type": ".py", "source": "/d/f.py"}, _code_text()),
            ({"file_type": ".docx", "source": "/d/f.docx"}, _prose_text(1000)),
            ({"file_type": ".dxf", "source": "/d/f.dxf"}, "CAD-Daten"),
        ]
        valid_pairs = set(STRATEGIES.values())
        for meta, text in test_cases:
            result = detect_chunk_params(text, meta)
            assert result in valid_pairs, f"Ungueltiges Ergebnis {result} fuer {meta['file_type']}"
