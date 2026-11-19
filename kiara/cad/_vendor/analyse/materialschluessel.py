"""Read the documented Sheet Metal table key as data, never as Lisp code.

Manufacturer anchors: reference/sheetmetal_inq.html and the supplied
SheetAdvisor/sha_demoshop.lsp table definition (:MATERIAL, :THICK, :KEY).
The material description is a catalog label, not a normalized material grade.
Additional catalog properties and units require the matching shop table.
"""
import math
import re

_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eEdD][+-]?[0-9]+)?\Z")
_KEYWORD = re.compile(r":[A-Za-z][A-Za-z0-9_-]*\Z")


class _Keyword(str):
    pass


class _Unbekannt(ValueError):
    pass


def _read(text):
    """Bounded subset: lists, keywords, quoted strings and finite numbers."""
    if len(text) > 16384:
        raise _Unbekannt("Schluessel_zu_lang")
    pos = 0

    def value(depth=0):
        nonlocal pos
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos == len(text) or depth > 3:
            raise _Unbekannt("Syntax_nicht_unterstuetzt")
        char = text[pos]
        if char == '(':
            pos += 1
            items = []
            while True:
                while pos < len(text) and text[pos].isspace():
                    pos += 1
                if pos == len(text):
                    raise _Unbekannt("Syntax_nicht_unterstuetzt")
                if text[pos] == ')':
                    pos += 1
                    return items
                items.append(value(depth + 1))
                if len(items) > 32:
                    raise _Unbekannt("Syntax_nicht_unterstuetzt")
        if char == '"':
            pos += 1
            chars = []
            while pos < len(text):
                char = text[pos]
                pos += 1
                if char == '"':
                    return ''.join(chars)
                if char == '\\':
                    if pos == len(text):
                        break
                    char = text[pos]
                    pos += 1
                chars.append(char)
            raise _Unbekannt("Syntax_nicht_unterstuetzt")
        start = pos
        while pos < len(text) and not text[pos].isspace() and text[pos] not in '()"':
            pos += 1
        word = text[start:pos]
        if _KEYWORD.fullmatch(word):
            return _Keyword(word.upper())
        if word.upper() == "NIL":                   # leerer Wert, z. B. :ROWKEY NIL (Ingestprobe 5.11.2026)
            return None
        if _NUMBER.fullmatch(word):
            number = float(word.replace('D', 'e').replace('d', 'e'))
            if math.isfinite(number):
                return number
        raise _Unbekannt("Syntax_nicht_unterstuetzt")

    parsed = value()
    if text[pos:].strip():
        raise _Unbekannt("Syntax_nicht_unterstuetzt")
    return parsed


def _plist(value, expected):
    if type(value) is not list or len(value) != 2 * len(expected):
        raise _Unbekannt("Schluesselfelder_unbekannt")
    result = {}
    for key, val in zip(value[::2], value[1::2]):
        if type(key) is not _Keyword or key not in expected or key in result:
            raise _Unbekannt("Schluesselfelder_unbekannt")
        result[key] = val
    if set(result) != set(expected):
        raise _Unbekannt("Schluesselfelder_unbekannt")
    return result


def lesen(text):
    """Typed catalog lookup key, or an explicit opaque/unsupported result.

    THICK belongs to the catalog row key. Its unit is declared by that
    external table, and it must not replace mbu_material's separate mm value.
    A standalone MATERIAL/THICK row key has no known shop/table context.
    Legacy tool ids are deliberately not relabelled as a material grade.
    """
    if type(text) is not str:
        return {"schema_erkannt": False, "status": "ungueltiger_typ"}
    if not text.strip():
        return {"schema_erkannt": False, "status": "leer"}
    if not text.lstrip().startswith('('):
        return {"schema_erkannt": False, "status": "legacy_oder_opak"}
    try:
        parsed = _read(text)
        standalone = type(parsed) is list and len(parsed) == 4
        if standalone:
            # Native material inquiry, original and private resave (12.11.2026):
            # this exact row-key form is itself the stored material tool key.
            # It does not identify an external shop/table or a thickness unit.
            root = {':SHOPNAME': None, ':TABNAME': None}
            row = _plist(parsed, {':MATERIAL', ':THICK'})
        else:
            root = _plist(parsed, {':SHOPNAME', ':TABNAME', ':ROWKEY'})
            if any(type(root[k]) is not str or not root[k] for k in (':SHOPNAME', ':TABNAME')):
                raise _Unbekannt("Schluesselwerte_unbekannt")
        if not standalone and root[':ROWKEY'] is None:
            # Tabellenschluessel mit ausdruecklich leerem Zeilenschluessel: keine Katalogzeile gespeichert.
            return {
                "schema_erkannt": True, "status": "tabellenschluessel_ohne_zeile",
                "shopname": root[':SHOPNAME'], "tabellenname": root[':TABNAME'],
                "materialbezeichnung": None, "dickenwert_im_schluessel": None,
                "dickeneinheit_im_schluessel": None, "normierte_werkstoffguete": None,
                "katalogabgleich": "nicht_durchgefuehrt",
            }
        if not standalone:
            row = _plist(root[':ROWKEY'], {':MATERIAL', ':THICK'})
        if (type(row[':MATERIAL']) is not str or not row[':MATERIAL']
                or type(row[':THICK']) is not float):
            raise _Unbekannt("Schluesselwerte_unbekannt")
    except _Unbekannt as exc:
        return {"schema_erkannt": False, "status": "unbekannte_form", "grund": str(exc)}
    return {
        "schema_erkannt": True,
        "status": "zeilenschluessel_ohne_tabelle" if standalone else "tabellenschluessel",
        "shopname": root[':SHOPNAME'], "tabellenname": root[':TABNAME'],
        "materialbezeichnung": row[':MATERIAL'],
        "dickenwert_im_schluessel": row[':THICK'],
        "dickeneinheit_im_schluessel": None,
        "normierte_werkstoffguete": None,
        "katalogabgleich": "nicht_durchgefuehrt",
    }
