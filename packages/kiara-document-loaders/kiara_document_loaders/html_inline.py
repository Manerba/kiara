"""Keep lexical tokens intact when emphasis covers only a part of a token."""

import re

from bs4 import Comment, NavigableString

_EMPHASIS = {"strong", "b", "em", "i"}
_INLINE = _EMPHASIS | {"span", "a", "abbr", "small", "code", "sub", "sup", "time"}
_TOKEN_FRAGMENT = re.compile(r"[\w./-]+\Z")
_TOKEN_END = re.compile(r"\w[-./_]?\Z")
_TOKEN_START = re.compile(r"[-./_]?\w")


def _edge_text(node, backwards):
    if isinstance(node, Comment):
        return None
    if isinstance(node, NavigableString):
        return str(node) or None
    if node.name not in _INLINE:
        return ""  # A block or line break ends the adjacent text flow.
    children = reversed(node.contents) if backwards else node.contents
    for child in children:
        text = _edge_text(child, backwards)
        if text is not None:
            return text
    return None


def _adjacent_text(element, backwards):
    attribute = "previous_sibling" if backwards else "next_sibling"
    current = element
    while current is not None:
        sibling = getattr(current, attribute)
        while sibling is not None:
            text = _edge_text(sibling, backwards)
            if text is not None:
                return text
            sibling = getattr(sibling, attribute)
        current = current.parent
        if current is None or current.name not in _INLINE:
            return ""
    return ""


def normalize_inline_emphasis(soup) -> tuple[str, ...]:
    """Neutralize emphasis only when its boundary interrupts a contiguous token.

    Element attributes and links remain intact. Whole emphasized words, spaces,
    block boundaries and line breaks keep their existing Markdown treatment.
    """
    changed = False
    for element in soup.find_all(_EMPHASIS):
        text = element.get_text()
        if not _TOKEN_FRAGMENT.fullmatch(text):
            continue
        if any(child.name not in _INLINE for child in element.find_all(True)):
            continue
        before = _adjacent_text(element, True)
        after = _adjacent_text(element, False)
        if _TOKEN_END.search(before) or _TOKEN_START.match(after):
            # Keep id/class and any nested link; only the Markdown emphasis
            # marker would interrupt the token represented by these text nodes.
            element.name = "span"
            changed = True
    return ("inline:partial_emphasis_normalized",) if changed else ()
