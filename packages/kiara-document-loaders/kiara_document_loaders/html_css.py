"""Conservative display cascade for HTML ingestion, without a browser viewport.

Only compound type/class/id selectors and display declarations decide removal.
Unresolved rules keep potentially affected content. Inline visibility styles and
hiding attributes are neutralized so the native converter cannot make a second,
less informed visibility decision. Other layout styles remain available to its
mail-table detection. No viewport or responsive-variant selection is performed.
"""

from __future__ import annotations

from dataclasses import dataclass

import tinycss2

_DISPLAY_VALUES = {
    "none",
    "block",
    "inline",
    "inline-block",
    "flow-root",
    "flex",
    "inline-flex",
    "grid",
    "inline-grid",
    "table",
    "inline-table",
    "table-row",
    "table-cell",
    "table-row-group",
    "table-header-group",
    "table-footer-group",
    "table-caption",
    "table-column",
    "table-column-group",
    "list-item",
    "contents",
}
_VISIBILITY_PROPERTIES = {
    "all",
    "visibility",
    "opacity",
    "content-visibility",
    "clip",
    "clip-path",
    "mso-hide",
    "animation",
    "animation-name",
    "transition",
    "transition-property",
}
_DISPLAY_RESET_PROPERTIES = {
    "all",
    "animation",
    "animation-name",
    "transition",
    "transition-property",
}


@dataclass(frozen=True)
class _Selector:
    tag: str | None
    ids: tuple[str, ...]
    classes: tuple[str, ...]

    @property
    def specificity(self):
        return (0, len(self.ids), len(self.classes), int(self.tag is not None))

    def matches(self, element):
        return (
            (self.tag is None or element.name.lower() == self.tag)
            and all(element.get("id") == name for name in self.ids)
            and all(name in element.get("class", ()) for name in self.classes)
        )


def _selector(tokens):
    tokens = [token for token in tokens if token.type != "comment"]
    while tokens and tokens[0].type == "whitespace":
        tokens = tokens[1:]
    while tokens and tokens[-1].type == "whitespace":
        tokens = tokens[:-1]
    tag, ids, classes = None, [], []
    index = 0
    if tokens and tokens[0].type == "ident":
        tag = tokens[0].value.lower()
        index += 1
    while index < len(tokens):
        token = tokens[index]
        if token.type == "hash" and token.is_identifier:
            ids.append(token.value)
        elif (
            token.type == "literal"
            and token.value == "."
            and index + 1 < len(tokens)
            and tokens[index + 1].type == "ident"
        ):
            index += 1
            classes.append(tokens[index].value)
        else:
            return None
        index += 1
    return _Selector(tag, tuple(ids), tuple(classes)) if tag or ids or classes else None


def _selectors(tokens):
    groups = [[]]
    for token in tokens:
        if token.type == "literal" and token.value == ",":
            groups.append([])
        else:
            groups[-1].append(token)
    return [_selector(group) for group in groups]


def _declarations(content, diagnostics):
    displays = []
    uncertain = False
    for node in tinycss2.parse_declaration_list(content, skip_comments=True, skip_whitespace=True):
        if node.type != "declaration":
            diagnostics.add("css:unparsed_declaration")
            uncertain = True
            continue
        if node.lower_name == "display":
            values = [token for token in node.value if token.type not in {"whitespace", "comment"}]
            value = (
                values[0].value.lower() if len(values) == 1 and values[0].type == "ident" else None
            )
            if value not in _DISPLAY_VALUES:
                diagnostics.add("css:unsupported_display_value")
                uncertain = True
            else:
                displays.append((value, node.important))
        elif node.lower_name in _VISIBILITY_PROPERTIES:
            diagnostics.add(f"css:unsupported_property:{node.lower_name}")
            uncertain |= node.lower_name in _DISPLAY_RESET_PROPERTIES
    return displays, uncertain


def _safe_style(style, display):
    # The native library also uses layout styles to recognize mail tables.
    # Keep those cues while neutralizing every unsupported visibility hint.
    declarations = [
        node
        for node in tinycss2.parse_declaration_list(style, skip_comments=True, skip_whitespace=True)
        if node.type == "declaration"
        and node.lower_name != "display"
        and node.lower_name not in _VISIBILITY_PROPERTIES
    ]
    sanitized = tinycss2.serialize(declarations)
    if display and display != "none":
        sanitized += f"display:{display};"
    return sanitized


def _collect_rules(
    rules, diagnostics, declarations, uncertain_selectors, *, conditional=False, depth=0
):
    if depth > 32:
        diagnostics.add("css:nesting_limit_preserved")
        uncertain_selectors.append(None)
        return
    for rule in rules:
        if rule.type == "qualified-rule":
            selectors = _selectors(rule.prelude)
            displays, uncertain = _declarations(rule.content, diagnostics)
            if None in selectors:
                diagnostics.add("css:unsupported_selector")
            if displays or uncertain:
                if conditional or uncertain or None in selectors:
                    uncertain_selectors.extend(selectors)
                else:
                    for value, important in displays:
                        for selector in selectors:
                            declarations.append((selector, value, important))
        elif rule.type == "at-rule":
            if rule.lower_at_keyword == "charset":
                continue
            if rule.lower_at_keyword == "import":
                diagnostics.add("css:external_stylesheet_preserved")
                uncertain_selectors.append(None)
            elif rule.content is not None:
                diagnostics.add(f"css:conditional_or_unsupported_rule:{rule.lower_at_keyword}")
                if rule.lower_at_keyword in {"font-face", "property", "page"}:
                    displays, uncertain = _declarations(rule.content, diagnostics)
                    if displays or uncertain:
                        uncertain_selectors.append(None)
                else:
                    _collect_rules(
                        tinycss2.parse_rule_list(
                            rule.content, skip_comments=True, skip_whitespace=True
                        ),
                        diagnostics,
                        declarations,
                        uncertain_selectors,
                        conditional=True,
                        depth=depth + 1,
                    )
            else:
                diagnostics.add(f"css:unsupported_rule:{rule.lower_at_keyword}")
        else:
            diagnostics.add("css:unparsed_stylesheet")
            uncertain_selectors.append(None)


def apply_visibility(soup) -> tuple[str, ...]:
    """Apply the supported visibility subset to a BeautifulSoup DOM in place."""
    diagnostics: set[str] = set()
    declarations = []
    uncertain_selectors = []
    for element in soup.find_all("link"):
        if "stylesheet" in [str(value).lower() for value in element.get("rel", ())]:
            diagnostics.add("css:external_stylesheet_preserved")
            uncertain_selectors.append(None)
            element.decompose()
    for style in list(soup.find_all("style")):
        media = str(style.get("media", "")).strip().lower()
        conditional = bool(media and media != "all")
        if conditional:
            diagnostics.add("css:conditional_style_element")
        if str(style.get("type", "text/css")).lower() != "text/css":
            diagnostics.add("css:unsupported_style_type")
            uncertain_selectors.append(None)
        else:
            _collect_rules(
                tinycss2.parse_stylesheet(
                    style.get_text(), skip_comments=True, skip_whitespace=True
                ),
                diagnostics,
                declarations,
                uncertain_selectors,
                conditional=conditional,
            )
        style.decompose()

    # Decide before changing the DOM. A hidden ancestor removes its full subtree.
    removals = []
    for element in list(soup.find_all(True)):
        uncertain = any(
            selector is None or selector.matches(element) for selector in uncertain_selectors
        )
        candidates = []
        if element.has_attr("hidden"):
            if str(element.get("hidden", "")).lower() == "until-found":
                diagnostics.add("css:hidden_until_found_preserved")
                uncertain = True
            candidates.append(((False, (-1, 0, 0, 0), -1), "none"))
        for order, (selector, value, important) in enumerate(declarations):
            if selector.matches(element):
                candidates.append(((important, selector.specificity, order), value))
        if element.has_attr("style"):
            inline, inline_uncertain = _declarations(str(element["style"]), diagnostics)
            uncertain |= inline_uncertain
            for order, (value, important) in enumerate(inline):
                candidates.append(((important, (1, 0, 0, 0), order), value))
        if element.has_attr("aria-hidden"):
            diagnostics.add("css:aria_hidden_preserved")
        display = max(candidates)[1] if candidates else None
        if not uncertain and display == "none":
            removals.append(element)
        if element.has_attr("style"):
            style = _safe_style(str(element["style"]), display)
            if style:
                element["style"] = style
            else:
                element.attrs.pop("style", None)
        for name in ("hidden", "aria-hidden", "inert"):
            element.attrs.pop(name, None)
    for element in removals:
        if element.parent is not None:
            element.decompose()
    return tuple(sorted(diagnostics))
