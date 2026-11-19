"""Konservative Diagnose der numerischen ZIP-PKG-Eintragsnamen.

Fehlender Startstrom, Luecken und doppelte Namen sind Containerbefunde, keine
vollstaendige Pruefung nativer Ladbarkeit. Im neutralen 0/2-Gegenversuch
lud Creo den Koerper in Strom 2 nicht; eine allgemeine Regel fuer beliebige
Luecken ist daraus nicht bewiesen. Ein befundfreier Container
garantiert weder gueltige Objektstroeme noch ein nativ gueltiges Modell.

Die Namen und Nutzdaten werden nicht umgeschrieben. SD-alt liefert ueber
altformat absichtlich einen analytischen Eintrag '2'; diese interne Normalform
unterliegt nicht den Regeln fuer einen tatsaechlichen ZIP-Container.
"""
from collections import Counter
import re
import struct
import zipfile

import altformat


def _nachfolger(name):
    """Dezimale Nachbarpruefung ohne int-Limit oder Aufzaehlen grosser Luecken."""
    ziffern = list(name)
    i = len(ziffern) - 1
    while i >= 0 and ziffern[i] == '9':
        ziffern[i] = '0'
        i -= 1
    if i < 0:
        return '1' + ''.join(ziffern)
    ziffern[i] = chr(ord(ziffern[i]) + 1)
    return ''.join(ziffern)


def zip_diagnose(namen):
    """Befunde fuer echte ZIP-Namen; keine Alias-/Pfad-/Unicode-Normalisierung.

    Nur kanonische ASCII-Dezimalnamen 0, 1, ... bilden die gepruefte Folge.
    ZIP-Reihenfolge und sonstige Eintragsnamen bleiben ohne Rollenannahme.
    Die Folge wird als Liste uebergeben, damit Duplikate sichtbar bleiben.
    """
    zahlen = Counter(n for n in namen if re.fullmatch(r'0|[1-9][0-9]*', n))
    folge = sorted(zahlen, key=lambda n: (len(n), n))
    befunde = []
    if '0' not in zahlen:
        befunde.append("ZIP-Container: numerischer Startstrom '0' fehlt.")
    for name in folge:
        if zahlen[name] > 1:
            befunde.append(f"ZIP-Container: doppelter numerischer Eintragsname {name!r} "
                           f"({zahlen[name]} Vorkommen); Zuordnung mehrdeutig.")
    for vorher, name in zip(folge, folge[1:]):
        if _nachfolger(vorher) != name:
            befunde.append(f"ZIP-Container: Luecke zwischen numerischen Eintragsnamen "
                           f"{vorher!r} und {name!r}; native Vollstaendigkeit nicht gesichert.")
    return befunde


def lokale_koepfe_diagnose(pfad):
    """Text zu einer ZIP-Datei ohne lesbares zentrales Verzeichnis: folgt den lokalen Koepfen ab Dateianfang
    und meldet, ob die angegebenen Datenlaengen ueber das Dateiende hinausreichen (abgeschnittene Datei).
    Liest nur; numerische Eintragsnamen werden genannt, andere nicht."""
    with open(pfad, 'rb') as f:
        daten = f.read()
    o, n = 0, 0
    while daten[o:o + 4] == b'PK\x03\x04' and o + 30 <= len(daten):
        flags, _, _, _, _, csize, _, nlen, elen = struct.unpack_from('<HHHHIIIHH', daten, o + 6)
        name = daten[o + 30:o + 30 + nlen].decode('cp437', 'replace')
        anzeige = repr(name) if re.fullmatch(r'0|[1-9][0-9]*', name) else 'ohne Zahlnamen'
        ende = o + 30 + nlen + elen + csize
        n += 1
        if flags & 8:
            return f"lokaler Eintrag {n} ({anzeige}) mit Datendeskriptor, Laenge im Kopf nicht bindend"
        if ende > len(daten):
            return (f"Datei abgeschnitten: lokaler Eintrag {n} ({anzeige}) verlangt {csize} komprimierte Byte "
                    f"bis Offset {ende}, die Datei endet nach {len(daten)} Byte; kein zentrales Verzeichnis")
        o = ende
    zentral = daten.find(b'PK\x01\x02', o)
    return (f"{n} lokale Eintraege vollstaendig, danach Offset {o}; "
            + ("zentrales Verzeichnis vorhanden, aber nicht lesbar" if zentral >= 0 else "kein zentrales Verzeichnis"))


def oeffnen_mit_diagnose(pfad):
    """(Eintragsliste, Befunde); SD-alt bleibt unveraendert analytisch lesbar.

    ZIP-Nutzdaten werden ueber ZipInfo statt ueber Namen gelesen. So bleiben
    auch gleichnamige Eintraege mit verschiedenen Bytes einzeln erhalten.
    Format-, CRC- und Lesefehler werden nicht als leere Diagnose verschluckt.
    """
    with open(pfad, 'rb') as datei:
        if datei.read(2) == b'PK':
            try:
                archiv = zipfile.ZipFile(datei)
            except zipfile.BadZipFile as e:
                raise zipfile.BadZipFile(f"{e}; {lokale_koepfe_diagnose(pfad)}") from e
            with archiv:
                infos = archiv.infolist()
                befunde = zip_diagnose(i.filename for i in infos)
                eintraege = [(i.filename, archiv.read(i)) for i in infos]
            return eintraege, befunde
    return altformat.oeffnen(pfad), []
