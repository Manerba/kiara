"""Leser für die Objektströme in Creo-Elements/Direct-.pkg-Dateien (Einträge 0, 1, 2 des ZIP).

Formatbeschreibung: FORMAT.md. Alle Zahlen Big-Endian, Zeichenketten UTF-16BE.
Der Stand ist v0: Kennungen und Nutzlasten sind belegt, ihre Bedeutung teils Hypothese.
"""
import struct
from dataclasses import dataclass, field

# Nutzlastlänge der Sonderkennungen 0x80 xx
# 0x31, 0x32, 0x28: Marken ohne Nutzlast im Segmentkopf von Normteilen aus PARTsolutions (31 vor dem Namen, 32 vor
# einer Kennung „00000002000003e8“, 28 vor D k), 31 auch im part_struct (Praxistest 30.10.2026)
# 0x41: Marke ohne Nutzlast im selben Normteilkopf vor 0x31, gefolgt von einem c-Text; in den Ingestproben vom
# 5.11.2026 stets ein Bibliotheksdateipfad (…/*.sdp). Rolle darüber hinaus nicht belegt.
# 0x12: Kurz-ID wie 0x11, nach einem zweiten 8002 (Verweis auf Inhalt aus einer anderen Ladung, Praxistest)
SUB80 = {0x02: 24, 0x11: 5, 0x12: 5, 0x14: 9, 0x21: 0, 0x22: 0, 0x25: 0, 0x26: 0, 0x28: 0, 0x31: 0, 0x32: 0,
         0x41: 0, 0x29: 4}
# 80 29: bisher ausschließlich vier Nullbytes, danach c-Text im Deklarationskopf.
# Native Original-/Neuspeicherbelege vom 12.11.2026; Rolle der Nullform offen.
# Nichtnullwerte bleiben unbekannt: aus der Nullform folgt keine allgemeine Nutzlastlänge.
# n enthält eine Elementanzahl, keine allgemeine Byteanzahl: Typ 3 = Byte, Typ 5 = 4-Byte-Element.
# Der Typ-5-Block des F2_G_TRI_FAN bleibt roh, ohne eine Geometrieinterpretation zu behaupten.
ROHBLOCK_BREITE = {3: 1, 5: 4}
# Kurz-ID in 80 11/80 12/80 14: erstes Nutzlastbyte = Marke, dann ID-hi, ID-lo und der Deklarationsindex k
# (80 11/80 12; bei 80 14 folgt danach noch ein u32). Die Marke beschreibt drei Längen in je zwei Bits,
# Code 1 → 1 Byte, 2 → 2 Byte, 3 → 4 Byte: Bits 0–1 = hi, Bits 2–3 = lo, Bits 4–5 = k.
# Bis 4.11.2026 belegt: 0x19 (1, 2, 1), 0x1a (2, 2, 1), 0x1d (1, 4, 1), 0x1e (2, 4, 1) – 0x1e in einem Paket aus
# einer langen Sitzung (hi = Ladezähler 0x348, Praxistest 30.10.2026).
# Ingestproben 5.11.2026 (Kiara Job 33): 0x15/0x16 (lo 1 Byte) in 80 14 und 0x29/0x2a/0x2d/0x2e (k 2 Byte) in
# 80 11, jeweils direkt nach der Deklaration `80 21 D 256 … 80 22`; die beiden Indexbytes sind genau dieses k = 256.
# Kundenmodell 6.11.2026: k = 256…509, jede 2-Byte-Kurz-ID direkt hinter ihrer Deklaration D k mit gleicher ID, Felder
# stets minimal lang; dasselbe Segment in zwei Paketen mit Ladezähler hi 0x99 bzw. 0x153 trägt 0x2d bzw. 0x2e,
# sonst bytegleiche Nutzlast (Hauptbaugruppe eines Kundenmodells).
# Zugelassen sind nur die beobachteten Feldlängen (hi 1/2, lo 1/2/4, k 1/2); 0x25/0x26 folgen aus derselben Regel.
KURZ_ID_CODE = {1: 1, 2: 2, 3: 4}


def _kurz_id_form(marke):
    hi, lo, k = (KURZ_ID_CODE.get((marke >> s) & 3) for s in (0, 2, 4))
    if marke >> 6 or hi not in (1, 2) or lo is None or k not in (1, 2):
        return None
    return hi, lo, k


# Marke → (hi, lo, k) in Byte; die bisherigen vier Marken zuerst (Schreiber wählen daraus, siehe kurz_id_kodieren)
KURZ_ID_FORMEN = {m: _kurz_id_form(m) for m in (0x19, 0x1A, 0x1D, 0x1E)}
KURZ_ID_FORMEN.update({m: f for m in range(256) if m not in KURZ_ID_FORMEN and (f := _kurz_id_form(m))})
KURZ_ID_TEILE = {m: (hi, lo) for m, (hi, lo, _) in KURZ_ID_FORMEN.items()}
KURZ_ID_LO = {m: hi + lo + k - 2 for m, (hi, lo, k) in KURZ_ID_FORMEN.items()}   # Mehrlänge gegenüber 0x19, +2


def kurz_id_index(art, nutzlast):
    """Deklarationsindex k einer Kurz-ID (Nutzlast als Hex): bei 80 11/80 12 das Indexfeld hinter ID-lo (Länge aus
    der Marke), bei 80 14 das letzte u32 (FORMAT.md, „Zähler und Deklarationen“)."""
    b = bytes.fromhex(nutzlast)
    if art == "8014":
        return int.from_bytes(b[-4:], "big")
    hi, lo, k = KURZ_ID_FORMEN[b[0]]
    return int.from_bytes(b[1 + hi + lo:1 + hi + lo + k], "big")


def ist_acht_bit(b):
    """Strom mit 8-Bit-Kopf `Rz` + u32 + `1` + u8-Länge + Versionstext (SolidDesigner 11.65/14.00, Ingest 5.11.2026).
    Texte stehen dort als a (u8-Länge) bzw. b (u16-Länge) in Roman-8; tokens() liefert sie als c, damit Namen und
    Attribute wie in 20.3-Strömen erscheinen, und kodieren() schreibt c in einem Strom mit 1-Kopf wieder als a/b."""
    return b[:2] == b"Rz" and b[6:7] == b"1"


class Unbekannt(Exception):
    """Unbekannte Kennung; offset zeigt auf das Kennungsbyte."""

    def __init__(self, offset, kennung):
        super().__init__(f"unbekannte Kennung {kennung:#04x} bei {offset:#x}")
        self.offset = offset
        self.kennung = kennung


@dataclass(slots=True)        # slots: ein Drittel des Speichers je Token (Pakete mit über 10 Mio. Tokens)
class Token:
    offset: int
    art: str
    wert: object = None

    def __iter__(self):
        # erlaubt "for o, k, w in tokens(b)"
        return iter((self.offset, self.art, self.wert))


def tokens(b, start=None):
    """Zerlegt einen Objektstrom in Token. Symbole werden je Segment aufgelöst.

    start: Offset eines Segmentanfangs ('q'), um nach einem Fehler wieder aufzusetzen;
    ohne start beginnt die Zerlegung beim Kopf.
    """
    i = start or 0
    syms = {}
    acht_bit = ist_acht_bit(b)
    if start is None and b[0:2] == b"Rz":
        (n,) = struct.unpack_from(">I", b, 2)
        yield Token(0, "R", n)
        i = 6
    while i < len(b):
        o, k = i, b[i]
        i += 1
        c = chr(k)
        # Native PLAIN_TEXT can retain isolated UTF-16 code units. Preserve
        # those units symmetrically; do not normalize embedded NUL.
        if c in "dc":
            (n,) = struct.unpack_from(">I", b, i)
            i += 4
            yield Token(o, c, b[i:i + 2 * n].decode("utf-16-be", errors="surrogatepass"))
            i += 2 * n
        elif c == "S":
            x, n = struct.unpack_from(">II", b, i)
            i += 8
            yield Token(o, c, (x, b[i:i + 2 * n].decode("utf-16-be", errors="surrogatepass")))
            i += 2 * n
        elif c in "qruIo":                 # o: Klammer ohne Nutzlast um eine Werteliste (Probe 30, A3D_TAG)
            if c == "q":
                syms = {}
            yield Token(o, c)
        elif c in "a1":                    # kurze Zeichenkette: u8-Länge + Bytes im Zeichensatz der Sitzung (Roman-8),
            n = b[i]                       # wie im alten Format; z. B. Katalogtexte „Fertigung“ (Praxistest 30.10.2026).
            i += 1                         # 1: Versionstext im Kopf von SolidDesigner-11/14-Strömen statt d (Ingest 5.11.2026)
            # 8-Bit-Strom: a ist dort die gewöhnliche Textform (SD 14.00: 2084 a, kein c) und erscheint als c
            yield Token(o, "c" if c == "a" and acht_bit else c, b[i:i + n].decode("hp_roman8", "surrogateescape"))
            i += n
        elif c == "b" and acht_bit:        # lange Zeichenkette (u16-Länge) wie im SD-11-Altformat; in SD-14-ZIP-Proben
            (n,) = struct.unpack_from(">H", b, i)       # noch nicht beobachtet, nur in diesem 8-Bit-Modus zugelassen
            i += 2
            yield Token(o, "c", b[i:i + n].decode("hp_roman8", "surrogateescape"))
            i += n
        elif c == "n":                     # u8 Elementtyp + u32 Elementanzahl + dicht gepackte Rohdaten
            art, n = struct.unpack_from(">BI", b, i)
            i += 5
            if art not in ROHBLOCK_BREITE:
                raise Unbekannt(o, k)
            n *= ROHBLOCK_BREITE[art]
            if i + n > len(b):
                raise Unbekannt(o, k)
            yield Token(o, c, (art, bytes(b[i:i + n])))
            i += n
        elif c == "U":                     # u16 zwischen Objektende u und nächstem t (Feature-/Vererbungsobjekte,
            (n,) = struct.unpack_from(">H", b, i)    # Praxistest 30.10.2026); Bedeutung offen
            i += 2
            yield Token(o, c, n)
        elif c == "s":
            idx, n = struct.unpack_from(">HB", b, i)
            i += 3
            syms[idx] = b[i:i + n].decode("ascii")
            i += n
            yield Token(o, c, (idx, syms[idx]))
        elif c == "z":
            (idx,) = struct.unpack_from(">H", b, i)
            i += 2
            yield Token(o, c, syms.get(idx, f"?{idx}"))
        elif c == "t":
            obj, sym = struct.unpack_from(">IH", b, i)
            i += 6
            yield Token(o, c, (obj & 0x7FFFFFFF, syms.get(sym, f"?{sym}")))
        elif c == "v":
            (obj,) = struct.unpack_from(">I", b, i)
            i += 4
            yield Token(o, c, obj & 0x7FFFFFFF if obj & 0x80000000 else ("roh", obj))
        elif c in "BD":
            yield Token(o, c, struct.unpack_from(">i", b, i)[0])
            i += 4
        elif c == "O":
            yield Token(o, c, b[i])
            i += 1
        elif c == "F":
            yield Token(o, c, struct.unpack_from(">d", b, i)[0])
            i += 8
        elif c == "K":
            yield Token(o, c, struct.unpack_from(">f", b, i)[0])
            i += 4
        elif k == 0x7F:
            yield Token(o, "NIL", b[i])
            i += 1
        elif k == 0x81:
            yield Token(o, "ID", "%x:%x" % struct.unpack_from(">II", b, i))
            i += 8
        elif k == 0x80 and i < len(b) and b[i] in SUB80:
            sub = b[i]
            n = SUB80[sub]
            i += 1
            if sub == 0x29 and b[i:i + 4] != b'\0\0\0\0':
                raise Unbekannt(o, k)
            if sub in (0x11, 0x12, 0x14):
                # Kurz-ID: Marke beschreibt die Längen von ID-hi, ID-lo und Deklarationsindex (KURZ_ID_FORMEN)
                if i >= len(b) or b[i] not in KURZ_ID_LO:
                    raise Unbekannt(i, b[i] if i < len(b) else -1)
                n += KURZ_ID_LO[b[i]] - 2
            yield Token(o, f"80{sub:02x}", b[i:i + n].hex() if n else None)
            i += n
        else:
            raise Unbekannt(o, k)


@dataclass
class Segment:
    start: int
    ende: int = -1
    wurzel: str = ""
    id: str = ""
    klassen: dict = field(default_factory=dict)


@dataclass
class Strom:
    version: str = ""
    zeichensatz: str = ""
    segmente: list = field(default_factory=list)
    fehler: Unbekannt = None


def gliedern(b):
    """Kopf und Segmente eines Objektstroms; bricht bei unbekannter Kennung mit fehler ab."""
    s = Strom()
    seg = None
    tiefe = 0
    try:
        for t in tokens(b):
            if t.art == "d" and seg is None:
                s.version = t.wert
            elif t.art == "S":
                s.zeichensatz = t.wert[1]
            elif t.art == "q":
                seg = Segment(t.offset)
                s.segmente.append(seg)
            elif t.art == "r":
                seg.ende = t.offset
            elif t.art == "z" and tiefe == 0 and not seg.wurzel:
                seg.wurzel = t.wert
            elif t.art == "ID" and tiefe == 0 and not seg.id:
                seg.id = t.wert
            elif t.art == "t":
                tiefe += 1
                seg.klassen[t.wert[1]] = seg.klassen.get(t.wert[1], 0) + 1
            elif t.art == "u":
                tiefe -= 1
    except Unbekannt as e:
        s.fehler = e
    return s


def kodieren(toks):
    """Umkehrung von tokens(): Tokenfolge → Bytes des Objektstroms. Symbolindizes je Segment aus den s-Token."""
    out = bytearray()
    syms = {}
    kopf_ab = None
    acht_bit = False
    for t in toks:
        k, w = t.art, t.wert
        if k == "1" and kopf_ab is not None:       # Versionstext 1 im Kopf: 8-Bit-Strom (ist_acht_bit)
            acht_bit = True
        if k == "R":
            out += b"Rz" + b"\0\0\0\0"          # Länge wird am Ende des Kopfes eingetragen
            kopf_ab = len(out)
        elif k == "c" and acht_bit:          # 8-Bit-Strom (Kopf mit 1): Texte als a bzw. ab 256 Byte als b
            roh = w.encode("hp_roman8", "surrogateescape")
            out += (b"a" + bytes([len(roh)]) if len(roh) <= 0xFF else b"b" + struct.pack(">H", len(roh))) + roh
        elif k in ("d", "c"):
            roh = w.encode("utf-16-be", errors="surrogatepass")
            out += k.encode() + struct.pack(">I", len(roh) // 2) + roh
        elif k == "S":
            roh = w[1].encode("utf-16-be", errors="surrogatepass")
            out += b"S" + struct.pack(">II", w[0], len(roh) // 2) + roh
        elif k == "q":
            if kopf_ab is not None:
                struct.pack_into(">I", out, kopf_ab - 4, len(out) - kopf_ab + 5)   # ab "z" einschließlich Längenfeld
                kopf_ab = None
            syms = {}
            out += b"q"
        elif k in ("r", "u", "I", "o"):
            out += k.encode()
        elif k == "U":
            out += b"U" + struct.pack(">H", w)
        elif k in ("a", "1"):
            roh = w.encode("hp_roman8", "surrogateescape")
            out += k.encode() + bytes([len(roh)]) + roh
        elif k == "n":
            width = ROHBLOCK_BREITE.get(w[0])
            if width is None or len(w[1]) % width:
                raise ValueError("Nicht belegter Rohblocktyp oder unvollstaendiges Element")
            out += b"n" + struct.pack(">BI", w[0], len(w[1]) // width) + w[1]
        elif k == "s":
            syms[w[1]] = w[0]
            out += b"s" + struct.pack(">HB", w[0], len(w[1])) + w[1].encode("ascii")
        elif k == "z":
            out += b"z" + struct.pack(">H", syms[w])
        elif k == "t":                       # Zeigerform-t (ohne Bit 31) wird hier mit Bit 31 geschrieben: offen
            out += b"t" + struct.pack(">IH", w[0] | 0x80000000, syms[w[1]])
        elif k == "v":                       # ('roh', x): Originalwert ohne Bit 31 (bis 4.11.2026 mit Bit 31)
            out += b"v" + struct.pack(">I", w[1] if isinstance(w, tuple) else w | 0x80000000)
        elif k in ("B", "D"):
            out += k.encode() + struct.pack(">i", w)
        elif k == "O":
            out += b"O" + bytes([w])
        elif k == "F":
            out += b"F" + struct.pack(">d", w)
        elif k == "K":
            out += b"K" + struct.pack(">f", w)
        elif k == "NIL":
            out += b"\x7f" + bytes([w])
        elif k == "ID":
            hi, lo = w.split(":")
            out += b"\x81" + struct.pack(">II", int(hi, 16), int(lo, 16))
        elif k.startswith("80"):
            if k == "8029" and w != "00000000":
                raise ValueError("Nur die belegte Nullform von 80 29 ist kodierbar")
            out += b"\x80" + bytes([int(k[2:], 16)]) + (bytes.fromhex(w) if w else b"")
        else:
            raise ValueError(f"Tokenart {k} nicht kodierbar")
    return bytes(out)
