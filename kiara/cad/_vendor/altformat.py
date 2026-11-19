"""Altes Paketformat (SolidDesigner 11 / CoCreate OneSpace) für die Analyse lesen.

Das alte Paket ist kein ZIP, sondern ein Unix-compress-Container (LZW, Magie 1f 9d) um einen einzelnen
Objektstrom: 28 Byte äußerer Kopf, `Rp1` + u8-Länge + ASCII-Version (`SD-VERSION SolidDesigner 11.65`),
dann Segmente wie im 20.3-Format (`q` … `r`), am Ende ein 185-Byte-Nachspann (`c7 00 01 00 02 81 b6` …).
Unterschiede im Strom: Zeichenketten `a` + u8-Länge + HP-Roman-8 statt `c`/`d` + u32 + UTF-16BE; `t`/`v`
tragen zeigerartige Objektnummern ohne Bit 31 (hier je Segment fortlaufend nummeriert); Kennung `o` ohne
Nutzlast (wie `I`); kein `S`-Zeichensatz-Token. Die Bedeutung der Felder ist sonst gleich; Abweichungen
der Geometrie (Körperlage, subshell, Torusbogen, Phantomknoten …) stehen in FORMAT.md „Altes Format“.

oeffnen(pfad) liefert die Einträge (Name, Bytes) eines Pakets beider Formate; alte Ströme werden in die
20.3-Tokenkodierung umgesetzt (objstrom.kodieren), so dass pkgleser, geometrie und pkg2step sie direkt lesen.
Diese Analysenormalform ist keine vollständige Konvertierung in ein natives 20.3-Paket: die alten Objektfelder
und Referenzformen bleiben bestehen. In der nativen Gegenprobe zu #377 lädt Creo das SD11-Original, weist
das normalisierte ZIP jedoch mit „Keine Daten gefunden“ ab. Ein nativer Original→20.3-Resave erfolgt in Creo.

Aufruf: altformat.py ALT.pkg ANALYSE.pkg – schreibt die Analysenormalform als ZIP (Eintrag `2`) für pkgdump & Co.
"""
import os
import struct
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from objstrom import KURZ_ID_LO, SUB80, Token, kodieren  # noqa: E402

COMPRESS_MAGIE = b"\x1f\x9d"
NACHSPANN = b"q\xc7\x00\x01\x00\x02\x81\xb6"                    # Anfang des 185-Byte-Nachspanns (und des Außenkopfs)
ROMAN8 = {0xC0: "â", 0xC1: "ê", 0xC2: "ô", 0xC3: "û", 0xC4: "á", 0xC5: "é", 0xC6: "ó", 0xC7: "ú", 0xC8: "à", 0xC9: "è",
          0xCA: "ò", 0xCB: "ù", 0xCC: "ä", 0xCD: "ë", 0xCE: "ö", 0xCF: "ü", 0xD0: "Å", 0xD1: "î", 0xD2: "Ø", 0xD3: "Æ",
          0xD4: "å", 0xD5: "í", 0xD6: "ø", 0xD7: "æ", 0xD8: "Ä", 0xD9: "ì", 0xDA: "Ö", 0xDB: "Ü", 0xDC: "É", 0xDD: "ï",
          0xDE: "ß", 0xDF: "Ô"}


def roman8(raw):
    """HP Roman-8 (Zeichensatz von SolidDesigner) nach Unicode, soweit für Namen gebraucht."""
    return "".join(ROMAN8.get(x, chr(x)) for x in raw)


def lzw_entpacken(b):
    """Unix compress (ncompress, .Z): LZW mit 9–16 Bit, Codes LSB-first, Blockmodus mit Löschcode 256.
    Der Packer schreibt Codes in Blöcken zu 8 Codes (n_bits Byte); beim Breitenwechsel und beim Löschcode
    bleibt der Rest des laufenden Blocks ungenutzt. Die Blöcke zählen ab dem letzten Breitenwechsel bzw.
    Löschcode (ncompress setzt dort seinen Puffer neu auf) – so lesen es ncompress und gzip."""
    if b[:2] != COMPRESS_MAGIE:
        raise ValueError("keine compress-Magie 1f 9d")
    maxbits, block = b[2] & 0x1F, bool(b[2] & 0x80)
    if not 9 <= maxbits <= 16:
        raise ValueError(f"compress: {maxbits} Bit nicht unterstützt")
    daten = b[3:]
    gesamt = len(daten) * 8
    tabelle = [bytes((i,)) for i in range(256)] + ([b""] if block else [])   # 256 = Löschcode (Platzhalter)
    n_bits, pos, marke, vorher = 9, 0, 0, None
    out = bytearray()

    def blockende(pos):
        g = n_bits * 8
        return marke + -(-(pos - marke) // g) * g

    while True:
        if len(tabelle) > (1 << n_bits) - 1 and n_bits < maxbits:              # Tabelle voll: breiter lesen
            pos = marke = blockende(pos)
            n_bits += 1
        if pos + n_bits > gesamt:
            break
        byte = pos >> 3
        code = (int.from_bytes(daten[byte:byte + 3], "little") >> (pos & 7)) & ((1 << n_bits) - 1)
        pos += n_bits
        if block and code == 256:
            pos = marke = blockende(pos)
            n_bits = 9
            del tabelle[257:]
            vorher = None
            continue
        if code < len(tabelle):
            eintrag = tabelle[code]
        elif code == len(tabelle) and vorher is not None:                        # KwKwK
            eintrag = vorher + vorher[:1]
        else:
            raise ValueError(f"compress: ungültiger Code {code} bei Bit {pos}")
        out += eintrag
        if vorher is not None and len(tabelle) < (1 << maxbits):
            tabelle.append(vorher + eintrag[:1])
        vorher = eintrag
    return bytes(out)


def kopf_finden(b):
    """Offset des Versionstexts `1` + u8-Länge + ASCII im Kopf eines alten Stroms, sonst -1.
    Zwei Kopfformen: `Rp1 …` (Fahrwerk DX95) und `Rz` + u32 + `1 …` (Ingestproben 5.11.2026, ebenfalls
    SolidDesigner 11.65 nach demselben 28-Byte-Außenkopf; das u32 ist dort 6, Bedeutung offen)."""
    j = b.find(b"Rp1", 0, 64)
    if j >= 0:
        return j + 2
    j = b.find(b"Rz", 0, 64)
    while 0 <= j <= 64:
        if b[j + 6:j + 7] == b"1" and b[j + 8:j + 18] == b"SD-VERSION":
            return j + 6
        j = b.find(b"Rz", j + 1, 64)
    return -1


def tokens_alt(b):
    """Tokenfolge eines alten (entpackten) Objektstroms in der Darstellung von objstrom.tokens; der Nachspann
    nach dem letzten Segment wird ignoriert."""
    j = kopf_finden(b)
    if j < 0:
        raise ValueError("kein 'Rp1'-Kopf eines SolidDesigner-Stroms")
    i = j + 1
    n = b[i]
    i += 1
    version = b[i:i + n].decode("latin-1")
    i += n
    yield Token(0, "R", 209)
    yield Token(0, "d", version)
    yield Token(0, "S", (8, "ROMAN8"))
    syms, nummern = {}, {}
    tiefe = 0                                                    # offene Segmente (q … r)
    offen = None                                                 # zurückgehaltenes q (der Nachspann beginnt mit q)

    def nr(ptr):
        if ptr not in nummern:
            nummern[ptr] = len(nummern)
        return nummern[ptr]

    while i < len(b):
        o, k = i, b[i]
        i += 1
        c = chr(k)
        if tiefe == 0 and c != "q" and b.startswith(NACHSPANN, i):
            return                                               # ein Einzelbyte vor dem Nachspann (Ingest 5.11.2026: 7a)
        if c == "a":
            n = b[i]
            i += 1
            t = Token(o, "c", roman8(b[i:i + n]))
            i += n
        elif c == "b":                                           # lange Zeichenkette: u16-Länge + Roman-8 (Ingest 5.11.2026,
            (n,) = struct.unpack_from(">H", b, i)                # Gewinde-/Normteiltexte über 255 Byte)
            i += 2
            t = Token(o, "c", roman8(b[i:i + n]))
            i += n
        elif c in "dc":
            (n,) = struct.unpack_from(">I", b, i)
            i += 4
            t = Token(o, c, b[i:i + 2 * n].decode("utf-16-be"))
            i += 2 * n
        elif c == "o":                                           # Kennung ohne Nutzlast
            t = Token(o, "I")
        elif c == "q":
            syms, nummern = {}, {}
            tiefe += 1
            offen = Token(o, c)
            continue
        elif c in "ruI":
            if c == "r":
                tiefe -= 1
            t = Token(o, c)
        elif c == "s":
            idx, n = struct.unpack_from(">HB", b, i)
            i += 3
            syms[idx] = b[i:i + n].decode("ascii")
            i += n
            t = Token(o, c, (idx, syms[idx]))
        elif c == "z":
            (idx,) = struct.unpack_from(">H", b, i)
            i += 2
            t = Token(o, c, syms.get(idx, f"?{idx}"))
        elif c == "t":
            obj, sym = struct.unpack_from(">IH", b, i)
            i += 6
            t = Token(o, c, (nr(obj), syms.get(sym, f"?{sym}")))
        elif c == "v":
            (obj,) = struct.unpack_from(">I", b, i)
            i += 4
            t = Token(o, c, nr(obj))
        elif c in "BD":
            t = Token(o, c, struct.unpack_from(">i", b, i)[0])
            i += 4
        elif c == "O":
            t = Token(o, c, b[i])
            i += 1
        elif c == "F":
            t = Token(o, c, struct.unpack_from(">d", b, i)[0])
            i += 8
        elif c == "K":
            t = Token(o, c, struct.unpack_from(">f", b, i)[0])
            i += 4
        elif k == 0x7F:
            t = Token(o, "NIL", b[i])
            i += 1
        elif k == 0x81:
            t = Token(o, "ID", "%x:%x" % struct.unpack_from(">II", b, i))
            i += 8
        elif k == 0x80 and i < len(b) and b[i] in SUB80:
            sub = b[i]
            n = SUB80[sub]
            i += 1
            if sub in (0x11, 0x12, 0x14):
                if i >= len(b) or b[i] not in KURZ_ID_LO:
                    raise ValueError(f"Kurz-ID-Form {b[i] if i < len(b) else -1:#x} bei {i:#x}")
                n += KURZ_ID_LO[b[i]] - 2
            t = Token(o, f"80{sub:02x}", b[i:i + n].hex() if n else None)
            i += n
        elif tiefe == 0 or (offen is not None and tiefe == 1):   # Nachspann: `r q c7 00 01 00 02 81 b6 …`
            return
        else:
            raise ValueError(f"unbekannte Kennung {k:#04x} bei {o:#x}: {b[o:o + 16].hex()}")
        if offen is not None:
            yield offen
            offen = None
        yield t


def ist_alt(b):
    """Anfang einer Datei: compress-Container oder bereits entpackter SolidDesigner-Strom."""
    return b[:2] == COMPRESS_MAGIE or kopf_finden(b) >= 0


def umsetzen(b):
    """Altes Paket → Analysestrom in 20.3-Tokenkodierung; kein natives 20.3-Paket."""
    if b[:2] == COMPRESS_MAGIE:
        b = lzw_entpacken(b)
    return kodieren(list(tokens_alt(b)))


def oeffnen(pfad):
    """Einträge (Name, Bytes) eines Pakets: ZIP (20.3) unverändert, altes Paket als ein Eintrag '2'."""
    with open(pfad, "rb") as f:
        anfang = f.read(64)
    if anfang[:2] == b"PK":
        with zipfile.ZipFile(pfad) as z:
            return [(info.filename, z.read(info)) for info in z.infolist()]
    if ist_alt(anfang):
        with open(pfad, "rb") as f:
            return [("2", umsetzen(f.read()))]
    raise ValueError(f"{pfad}: weder ZIP-Paket noch SolidDesigner-Paket (compress)")


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    with open(sys.argv[1], "rb") as f:
        daten = umsetzen(f.read())
    with zipfile.ZipFile(sys.argv[2], "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("2", daten)
    print(f"{sys.argv[2]}: Eintrag 2 mit {len(daten)} Byte geschrieben")
    return 0


if __name__ == "__main__":
    sys.exit(main())
