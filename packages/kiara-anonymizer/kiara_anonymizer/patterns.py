"""Konsolidiertes Regex-Pattern-Set fuer strukturierte PII-Erkennung.

Erkennt IBAN, Steuernummern, Telefonnummern, E-Mail-Adressen, IPv4-Adressen
und benutzerdefinierte Muster. Zweite Stufe nach dem Aho-Corasick-Scan.

entity_type-Werte sind UPPERCASE (IBAN, TAX_ID, PHONE, EMAIL, IP_ADDRESS, CUSTOM)
— konsistent mit der Prefix-Tabelle und Typ-Normalisierung in automaton.py.
"""

import itertools
import re

# Vorkompilierte Regex-Muster — entity_type in UPPERCASE
_BUILTIN_PATTERNS = [
    {
        "name": "iban",
        "entity_type": "IBAN",
        "pattern": re.compile(
            r"\b[A-Z]{2}\d{2}"
            r"(?:[A-Z0-9]{11,30}|(?:\s?[A-Z0-9]{4}){2,7}(?:\s?[A-Z0-9]{1,3})?)\b",
            re.IGNORECASE,
        ),
    },
    {
        "name": "tax_id_slash",
        "entity_type": "TAX_ID",
        "pattern": re.compile(r"\b\d{2,3}/\d{3}/\d{4,5}\b"),
    },
    {
        "name": "tax_id_numeric",
        "entity_type": "TAX_ID",
        "pattern": re.compile(
            r"(?i)(?:steuer[\-\s]?(?:nummer|ident(?:ifikationsnummer)?|id|nr)|"
            r"tin|idnr|st[\.\-]?nr)[\s.:]*"
            r"(?<![/\d])\d{10,11}\b(?![/\d])"
        ),
    },
    {
        "name": "ust_id",
        "entity_type": "TAX_ID",
        "pattern": re.compile(r"\bDE\s?\d{9}\b", re.IGNORECASE),
    },
    {
        "name": "phone_intl",
        "entity_type": "PHONE",
        "pattern": re.compile(
            r"(?<![\d+])(?:\+49|0049)"
            r"(?:"
            r"[\s/\-]?\(0\)\s?\d{2,5}(?:[\s/\-]?\d{2,5}){1,4}"
            r"|[\s/\-]?\(?0?\d{2,5}\)?(?:[\s/\-]?\d{2,5}){1,4}"
            r"|\d{8,12}"
            r")"
            r"(?!\d)"
        ),
    },
    {
        "name": "phone_de",
        "entity_type": "PHONE",
        "pattern": re.compile(r"(?<!\d)\(?0\d{2,5}\)?(?:[\s/\-]?\d{2,5}){1,4}(?!\d)"),
    },
    {
        "name": "email",
        "entity_type": "EMAIL",
        "pattern": re.compile(
            r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b",
            re.UNICODE,
        ),
    },
    {
        "name": "ipv4_private",
        "entity_type": "IP_ADDRESS",
        "pattern": re.compile(
            r"\b(?:"
            r"10\.(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\.(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\.(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)"
            r"|172\.(?:1[6-9]|2\d|3[01])\.(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\.(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)"
            r"|192\.168\.(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\.(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)"
            r")\b"
        ),
    },
]

# ReDoS-Schutz fuer benutzerdefinierte Patterns
_NESTED_QUANTIFIER_RE = re.compile(r"\([^()]*[+*{][^()]*\)[+*{]")
_QUANTIFIED_ALTERNATION_RE = re.compile(r"\((?:\?:)?[^()]*\|[^()]*\)[+*{]")
_QUANTIFIED_WILDCARD_RE = re.compile(r"\([^()]*\.[*+][^()]*\)[+*{]")
_BACKREFERENCE_RE = re.compile(r"\\[1-9]")
_URL_RE = re.compile(r'(?i)\b[a-z][a-z0-9+.-]*://[^\s<>"\'\])}]+')
_MAX_CUSTOM_PATTERN_LEN = 500
_MAX_MATCHES_PER_PATTERN = 500
_IBAN_LENGTHS = {
    "AD": 24,
    "AE": 23,
    "AL": 28,
    "AT": 20,
    "AZ": 28,
    "BA": 20,
    "BE": 16,
    "BG": 22,
    "BH": 22,
    "BI": 27,
    "BR": 29,
    "BY": 28,
    "CH": 21,
    "CR": 22,
    "CY": 28,
    "CZ": 24,
    "DE": 22,
    "DK": 18,
    "DO": 28,
    "EE": 20,
    "EG": 29,
    "ES": 24,
    "FI": 18,
    "FO": 18,
    "FR": 27,
    "GB": 22,
    "GE": 22,
    "GI": 23,
    "GL": 18,
    "GR": 27,
    "GT": 28,
    "HR": 21,
    "HU": 28,
    "IE": 22,
    "IL": 23,
    "IQ": 23,
    "IS": 26,
    "IT": 27,
    "JO": 30,
    "KZ": 20,
    "KW": 30,
    "LB": 28,
    "LC": 32,
    "LI": 21,
    "LT": 20,
    "LU": 20,
    "LV": 21,
    "LY": 25,
    "MC": 27,
    "MD": 24,
    "ME": 22,
    "MK": 19,
    "MR": 27,
    "MT": 31,
    "MU": 30,
    "NL": 18,
    "NO": 15,
    "PK": 24,
    "PL": 28,
    "PS": 29,
    "PT": 25,
    "QA": 29,
    "RO": 24,
    "RS": 22,
    "SA": 24,
    "SC": 31,
    "SD": 18,
    "SE": 24,
    "SI": 19,
    "SK": 24,
    "SM": 27,
    "SO": 23,
    "ST": 25,
    "SV": 28,
    "TL": 23,
    "TN": 24,
    "TR": 26,
    "UA": 29,
    "VA": 22,
    "VG": 24,
    "XK": 20,
}


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merges sorted or unsorted half-open intervals."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if start >= end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _overlaps_any(start: int, end: int, intervals: list[tuple[int, int]]) -> bool:
    return any(iv_start < end and start < iv_end for iv_start, iv_end in intervals)


def _inside_any(start: int, end: int, intervals: list[tuple[int, int]]) -> bool:
    return any(iv_start <= start and end <= iv_end for iv_start, iv_end in intervals)


def _is_safe_custom_pattern(raw: str) -> bool:
    """Rejects custom regexes with obvious catastrophic-backtracking shapes."""
    if len(raw) > _MAX_CUSTOM_PATTERN_LEN:
        return False
    if _BACKREFERENCE_RE.search(raw):
        return False
    if _NESTED_QUANTIFIER_RE.search(raw):
        return False
    if _QUANTIFIED_ALTERNATION_RE.search(raw):
        return False
    return not _QUANTIFIED_WILDCARD_RE.search(raw)


def _valid_candidate(pattern_name: str, value: str) -> bool:
    """Pattern-specific post validation for broad regexes."""
    if pattern_name == "iban":
        return _is_valid_iban(value)
    if pattern_name.startswith("phone_"):
        digits = re.sub(r"\D", "", value)
        if value.lstrip().startswith("+49") or digits.startswith("0049"):
            return 10 <= len(digits) <= 15
        return 8 <= len(digits) <= 13
    return True


def _is_valid_iban(value: str) -> bool:
    """Validates IBAN country length and ISO 7064 Mod-97 checksum."""
    iban = re.sub(r"\s+", "", value).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]+", iban):
        return False

    expected_length = _IBAN_LENGTHS.get(iban[:2])
    if expected_length is None or len(iban) != expected_length:
        return False

    rearranged = iban[4:] + iban[:4]
    remainder = 0
    for char in rearranged:
        digits = char if char.isdigit() else str(ord(char) - ord("A") + 10)
        for digit in digits:
            remainder = (remainder * 10 + int(digit)) % 97
    return remainder == 1


def _resolve_pattern_overlaps(matches: list[dict]) -> list[dict]:
    """Keeps the longest regex match; pattern order only breaks ties."""
    matches.sort(key=lambda m: (-(m["end"] - m["start"]), m["start"], m["_order"]))
    resolved: list[dict] = []
    resolved_intervals: list[tuple[int, int]] = []

    for match in matches:
        start, end = match["start"], match["end"]
        if _overlaps_any(start, end, resolved_intervals):
            continue
        resolved.append(match)
        resolved_intervals.append((start, end))

    resolved.sort(key=lambda x: x["start"])
    for match in resolved:
        match.pop("_order", None)
    return resolved


class RegexPatternMatcher:
    """Erkennt strukturierte PII-Muster: IBAN, Steuernummern, Telefon, E-Mail, IPv4.

    Wird als zweite Stufe nach dem Aho-Corasick-Scan eingesetzt.
    Bereits durch den Automaten abgedeckte Bereiche werden uebersprungen.
    """

    def __init__(self, custom_patterns: list[dict] | None = None):
        """Initialisiert den Matcher mit optionalen benutzerdefinierten Mustern.

        Args:
            custom_patterns: [{"name": "...", "pattern": "regex-string",
                              "category": "...", "type": "..."}]
                type/category werden als entity_type interpretiert (UPPERCASE).
        """
        self._patterns = list(_BUILTIN_PATTERNS)
        if custom_patterns:
            for cp in custom_patterns:
                try:
                    raw = cp["pattern"]
                except KeyError:
                    continue
                if not _is_safe_custom_pattern(raw):
                    continue
                try:
                    compiled = re.compile(raw)
                except re.error:
                    continue
                # Akzeptiere sowohl "type" als auch "category" als Key.
                # str() haertet gegen Nicht-String-Werte aus YAML (Integer/Bool/None),
                # die sonst .upper() mit AttributeError sprengen wuerden.
                entity_type = str(cp.get("type", cp.get("category", "CUSTOM"))).upper()
                self._patterns.append(
                    {
                        "name": cp.get("name", "custom"),
                        "entity_type": entity_type,
                        "pattern": compiled,
                    }
                )

    def find_patterns(
        self,
        text: str,
        already_covered: list[tuple[int, int]] | set[tuple[int, int]] | None = None,
    ) -> list[dict]:
        """Findet strukturierte PII-Muster im Text.

        Args:
            text: Der zu durchsuchende Text.
            already_covered: Bereiche (start, end), die bereits durch den
                            Aho-Corasick-Automaten abgedeckt sind.

        Returns:
            [{"start": ..., "end": ..., "value": ...,
              "entity_type": "IBAN"|"TAX_ID"|"PHONE"|"EMAIL"|"IP_ADDRESS"|...,
              "pattern_name": "iban"|"phone_de"|...}, ...]
        """
        covered = []
        if already_covered:
            covered = _merge_intervals(list(already_covered))

        url_spans = [(m.start(), m.end()) for m in _URL_RE.finditer(text)]

        matches = []
        seen_candidates = set()

        for order, pat in enumerate(self._patterns):
            for m in itertools.islice(pat["pattern"].finditer(text), _MAX_MATCHES_PER_PATTERN):
                ms, me = m.start(), m.end()
                if ms == me:
                    continue

                # Duplikate vermeiden
                identity = (ms, me, pat["entity_type"], pat["name"])
                if identity in seen_candidates:
                    continue

                # Ueberlappung mit bereits abgedeckten Bereichen pruefen
                if _overlaps_any(ms, me, covered):
                    continue

                if pat["entity_type"] == "EMAIL" and _inside_any(ms, me, url_spans):
                    continue

                value = m.group()
                if not _valid_candidate(pat["name"], value):
                    continue

                seen_candidates.add(identity)
                matches.append(
                    {
                        "start": ms,
                        "end": me,
                        "value": value,
                        "entity_type": pat["entity_type"],
                        "pattern_name": pat["name"],
                        "_order": order,
                    }
                )

        return _resolve_pattern_overlaps(matches)
