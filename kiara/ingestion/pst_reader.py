"""PST-Leser: Ordner, Mails, Anhaenge und Termine aus Outlook-Datendateien.

Duenne Schicht ueber libpff (``pypff``). Die Python-Bindung liefert Ordner,
Nachrichten, Anhaenge und rohe MAPI-Properties, loest aber keine benannten
Properties (0x8000+) auf. Die Name-to-ID-Map wird deshalb hier selbst gelesen
(MS-PST 2.4.7), damit Terminfelder (Beginn, Ende, Ort, Serie) verfuegbar sind.
Bodies, die nur als RTF vorliegen (z.B. Besprechungsanfragen), werden nach
MS-OXRTFEX entkapselt (HTML bzw. Text).
"""

import re
import struct
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone

# --- MAPI-Property-Tags (Property-ID ohne Typ) ---------------------------------

PR_MESSAGE_CLASS = 0x001A
PR_SUBJECT = 0x0037
PR_CLIENT_SUBMIT_TIME = 0x0039
PR_SENT_REPRESENTING_NAME = 0x0042
PR_SENT_REPRESENTING_SMTP = 0x5D02
PR_SENDER_NAME = 0x0C1A
PR_SENDER_EMAIL_ADDRESS = 0x0C1F
PR_SENDER_SMTP_ADDRESS = 0x5D01
PR_RECIPIENT_TYPE = 0x0C15
PR_MESSAGE_DELIVERY_TIME = 0x0E06
PR_DISPLAY_NAME = 0x3001
PR_EMAIL_ADDRESS = 0x3003
PR_SMTP_ADDRESS = 0x39FE
PR_CREATION_TIME = 0x3007
PR_INTERNET_CPID = 0x3FDE
PR_IPM_SUBTREE_ENTRYID = 0x35E0
PR_ATTACH_FILENAME = 0x3704
PR_ATTACH_METHOD = 0x3705
PR_ATTACH_FLAGS = 0x3714
PR_ATTACHMENT_HIDDEN = 0x7FFE

ATTACH_BY_VALUE = 1
ATTACH_EMBEDDED_MSG = 5
ATTACH_OLE = 6
ATT_MHTML_REF = 0x4

RECIPIENT_TO = 1
RECIPIENT_CC = 2

PT_LONG = 0x0003
PT_BOOLEAN = 0x000B
PT_STRING8 = 0x001E
PT_UNICODE = 0x001F
PT_SYSTIME = 0x0040

# Benannte Properties: (Property-Set, LID)
PSETID_APPOINTMENT = uuid.UUID("00062002-0000-0000-c000-000000000046")
PS_MAPI = uuid.UUID("00020328-0000-0000-c000-000000000046")
PS_PUBLIC_STRINGS = uuid.UUID("00020329-0000-0000-c000-000000000046")

LID_BUSY_STATUS = 0x8205
LID_LOCATION = 0x8208
LID_APPOINTMENT_START = 0x820D
LID_APPOINTMENT_END = 0x820E
LID_APPOINTMENT_ALL_DAY = 0x8215
LID_APPOINTMENT_STATE_FLAGS = 0x8217
LID_RECURRING = 0x8223
LID_RECURRENCE_PATTERN = 0x8232
LID_ALL_ATTENDEES = 0x8238
LID_TO_ATTENDEES = 0x823B
LID_CC_ATTENDEES = 0x823C

ASF_CANCELED = 0x4

# Internet-Codepages (PR_INTERNET_CPID) ohne direkten "cpNNNN"-Codec in Python.
_CODEPAGES = {
    20127: "ascii",
    20866: "koi8_r",
    21866: "koi8_u",
    28591: "iso8859_1",
    28592: "iso8859_2",
    28593: "iso8859_3",
    28594: "iso8859_4",
    28595: "iso8859_5",
    28596: "iso8859_6",
    28597: "iso8859_7",
    28598: "iso8859_8",
    28599: "iso8859_9",
    28603: "iso8859_13",
    28605: "iso8859_15",
    50220: "iso2022_jp",
    51932: "euc_jp",
    54936: "gb18030",
    65001: "utf-8",
}

MAIL_CLASSES = ("IPM.NOTE", "IPM.SCHEDULE.MEETING", "IPM.POST", "REPORT.")
APPOINTMENT_CLASSES = ("IPM.APPOINTMENT",)


def codepage_to_charset(codepage: int | None) -> str:
    """Windows-/Internet-Codepage auf einen Python-Codec abbilden (Fallback utf-8)."""
    import codecs

    if not codepage:
        return "utf-8"
    name = _CODEPAGES.get(codepage, f"cp{codepage}")
    try:
        return codecs.lookup(name).name
    except LookupError:
        return "utf-8"


def item_kind(message_class: str) -> str | None:
    """Ordnet eine Message-Class ein: "mail", "appointment" oder None (uebergehen)."""
    upper = (message_class or "").upper()
    if upper.startswith(APPOINTMENT_CLASSES):
        return "appointment"
    if upper.startswith(MAIL_CLASSES):
        return "mail"
    return None


# --- Properties ----------------------------------------------------------------


class Props:
    """Properties des ersten Record-Sets eines pypff-Items, nach Property-ID."""

    def __init__(self, item, named: dict[tuple[uuid.UUID, int], int] | None = None):
        self._entries: dict[int, object] = {}
        self._named = named or {}
        try:
            if item.number_of_record_sets < 1:
                return
            record_set = item.get_record_set(0)
            for index in range(record_set.number_of_entries):
                entry = record_set.get_entry(index)
                self._entries.setdefault(entry.entry_type, entry)
        except OSError:
            # Defekte Property-Tabelle: Item ohne Properties behandeln.
            self._entries = {}

    def _entry(self, key):
        if isinstance(key, tuple):
            key = self._named.get(key)
            if key is None:
                return None
        return self._entries.get(key)

    def string(self, key) -> str:
        entry = self._entry(key)
        if entry is None or entry.value_type not in (PT_STRING8, PT_UNICODE):
            return ""
        try:
            return (entry.data_as_string or "").strip("\x00").strip()
        except (OSError, UnicodeError):
            return ""

    def integer(self, key) -> int | None:
        entry = self._entry(key)
        if entry is None or entry.value_type != PT_LONG:
            return None
        try:
            return entry.data_as_integer
        except OSError:
            return None

    def boolean(self, key) -> bool | None:
        entry = self._entry(key)
        if entry is None or entry.value_type != PT_BOOLEAN:
            return None
        try:
            return entry.data_as_boolean
        except OSError:
            return None

    def time(self, key) -> datetime | None:
        """PT_SYSTIME als aware UTC-Zeitpunkt (None bei fehlendem/leerem Wert)."""
        entry = self._entry(key)
        if entry is None or entry.value_type != PT_SYSTIME:
            return None
        try:
            raw = entry.data
            if not raw or raw == b"\x00" * 8:
                return None
            value = entry.data_as_datetime
        except (OSError, ValueError, OverflowError):
            return None
        return value.replace(tzinfo=timezone.utc) if value else None

    def binary(self, key) -> bytes:
        entry = self._entry(key)
        if entry is None:
            return b""
        try:
            return entry.data or b""
        except OSError:
            return b""


def _read_named_map(pst) -> dict[tuple[uuid.UUID, int], int]:
    """Liest die Name-to-ID-Map (nur numerische Namen) -> Property-ID 0x8000+n."""
    try:
        node = pst.get_name_to_id_map()
    except OSError:
        return {}
    if node is None:
        return {}
    props = Props(node)
    guids = props.binary(0x0002)
    entries = props.binary(0x0003)
    result: dict[tuple[uuid.UUID, int], int] = {}
    for offset in range(0, len(entries) - 7, 8):
        name_id, guid_and_kind, prop_index = struct.unpack_from("<IHH", entries, offset)
        if guid_and_kind & 0x1:
            continue  # String-Name, hier nicht benoetigt
        guid_index = guid_and_kind >> 1
        if guid_index == 1:
            guid = PS_MAPI
        elif guid_index == 2:
            guid = PS_PUBLIC_STRINGS
        else:
            start = (guid_index - 3) * 16
            if guid_index < 3 or start + 16 > len(guids):
                continue
            guid = uuid.UUID(bytes_le=guids[start : start + 16])
        result[(guid, name_id)] = 0x8000 + prop_index
    return result


# --- RTF-Entkapselung (MS-OXRTFEX) -----------------------------------------------

_RTF_TOKEN = re.compile(
    rb"\\([a-zA-Z]+)(-?\d+)? ?|\\'([0-9a-fA-F]{2})|\\(.)|([{}])|([\r\n]+)|([^\\{}\r\n]+)",
    re.DOTALL,
)
_RTF_SKIP_DESTINATIONS = {
    b"fonttbl",
    b"colortbl",
    b"stylesheet",
    b"info",
    b"pict",
    b"object",
    b"header",
    b"footer",
    b"listtable",
    b"listoverridetable",
    b"rsidtbl",
    b"xmlnstbl",
    b"themedata",
    b"datastore",
    b"latentstyles",
}
_RTF_SYMBOLS = {
    b"par": "\n",
    b"line": "\n",
    b"tab": "\t",
    b"lquote": "\u2018",
    b"rquote": "\u2019",
    b"ldblquote": "\u201c",
    b"rdblquote": "\u201d",
    b"bullet": "\u2022",
    b"endash": "\u2013",
    b"emdash": "\u2014",
    b"emspace": " ",
    b"enspace": " ",
}


@dataclass
class _RtfGroup:
    skip: bool = False
    htmltag: bool = False
    htmlrtf: bool = False
    uc: int = 1


def decapsulate_rtf(rtf: bytes) -> tuple[str, str]:
    """RTF-Body in ("text/html", html) oder ("text/plain", text) umwandeln.

    Bei ``\\fromhtml`` wird das eingebettete HTML rekonstruiert, sonst der
    sichtbare Text extrahiert (auch fuer RTF ohne Kapselung, best effort).
    """
    is_html = b"\\fromhtml" in rtf[:1024]
    match = re.search(rb"\\ansicpg(\d+)", rtf[:1024])
    charset = codepage_to_charset(int(match.group(1)) if match else 1252)

    out: list[str] = []
    pending = bytearray()
    stack: list[_RtfGroup] = []
    group = _RtfGroup()
    skip_chars = 0
    expect_destination = False
    position = 0
    length = len(rtf)

    def flush() -> None:
        if pending:
            out.append(pending.decode(charset, errors="replace"))
            pending.clear()

    def visible() -> bool:
        return not group.skip and (group.htmltag or not group.htmlrtf)

    while position < length:
        match = _RTF_TOKEN.match(rtf, position)
        if match is None:
            position += 1
            continue
        position = match.end()
        word, param, hex_byte, symbol, brace, newline, text = match.groups()

        if brace == b"{":
            stack.append(group)
            group = _RtfGroup(group.skip, group.htmltag, group.htmlrtf, group.uc)
            expect_destination = True
            continue
        expect_now, expect_destination = expect_destination, False
        if brace == b"}":
            group = stack.pop() if stack else _RtfGroup()
            continue
        if newline is not None:
            continue
        if word is not None:
            if word == b"bin" and param:
                position += max(int(param), 0)
                continue
            if skip_chars:
                skip_chars -= 1
                continue
            if expect_now and word in _RTF_SKIP_DESTINATIONS:
                group.skip = True
            elif word == b"htmlrtf":
                group.htmlrtf = param != b"0"
            elif word == b"uc" and param is not None:
                group.uc = max(int(param), 0)
            elif word == b"u" and param is not None:
                if visible():
                    flush()
                    code = int(param)
                    out.append(chr(code + 65536 if code < 0 else code))
                skip_chars = group.uc
            elif word in _RTF_SYMBOLS and visible():
                pending.extend(_RTF_SYMBOLS[word].encode(charset, errors="replace"))
            continue
        if symbol is not None:
            if symbol == b"*":
                # Unbekannte \*-Ziele ignorieren, \*\htmltag dagegen ausgeben.
                tag = re.match(rb"\\(htmltag|mhtmltag)(-?\d+)? ?", rtf[position:])
                if tag and tag.group(1) == b"htmltag":
                    position += tag.end()
                    group.htmltag = True
                else:
                    group.skip = True
                continue
            if skip_chars:
                skip_chars -= 1
                continue
            if visible():
                if symbol in (b"\\", b"{", b"}"):
                    pending.extend(symbol)
                elif symbol == b"~":
                    pending.extend("\xa0".encode(charset, errors="replace"))
                elif symbol == b"_":
                    pending.extend(b"-")
                elif symbol in (b"\n", b"\r"):
                    pending.extend(b"\n")
            continue
        if hex_byte is not None:
            if skip_chars:
                skip_chars -= 1
                continue
            if visible():
                pending.append(int(hex_byte, 16))
            continue
        if text is not None:
            if skip_chars:
                consumed = min(skip_chars, len(text))
                skip_chars -= consumed
                text = text[consumed:]
            if text and visible():
                pending.extend(text)
    flush()
    result = "".join(out)
    # \u-Surrogatpaare zusammensetzen
    result = result.encode("utf-16", "surrogatepass").decode("utf-16", errors="replace")
    if is_html:
        return "text/html", result
    return "text/plain", re.sub(r"\n{3,}", "\n\n", result).strip()


# --- Datenmodell -----------------------------------------------------------------


@dataclass
class PstAttachment:
    filename: str
    data: bytes


@dataclass
class PstItem:
    """Eine Nachricht oder ein Termin, unabhaengig von libpff."""

    kind: str  # "mail" | "appointment"
    identifier: int
    message_class: str
    subject: str
    sender: str
    to: str
    cc: str
    date: datetime | None
    body: bytes
    body_mime: str
    body_charset: str
    start: datetime | None = None
    end: datetime | None = None
    all_day: bool = False
    location: str = ""
    attendees: str = ""
    recurrence: str = ""
    canceled: bool = False
    attachments: list[PstAttachment] = field(default_factory=list)
    skipped_attachments: int = 0


class PstFile:
    """Geoeffnete PST-Datei (Kontextmanager)."""

    def __init__(self, path: str):
        import pypff

        self.path = path
        self._file = pypff.file()
        self._file.open(path)
        self._named: dict[tuple[uuid.UUID, int], int] | None = None

    def close(self) -> None:
        try:
            self._file.close()
        except OSError:
            pass

    def __enter__(self) -> "PstFile":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    @property
    def named(self) -> dict[tuple[uuid.UUID, int], int]:
        if self._named is None:
            self._named = _read_named_map(self._file)
        return self._named

    def display_name(self) -> str:
        try:
            return Props(self._file.get_message_store()).string(PR_DISPLAY_NAME)
        except OSError:
            return ""

    # --- Ordner ---

    def _ipm_root(self):
        """Oberster Ordner der persoenlichen Ordner (ohne Such-/Systemordner)."""
        root = self._file.get_root_folder()
        try:
            entry_id = Props(self._file.get_message_store()).binary(PR_IPM_SUBTREE_ENTRYID)
        except OSError:
            entry_id = b""
        if len(entry_id) >= 24:
            (nid,) = struct.unpack_from("<I", entry_id, 20)
            for index in range(root.number_of_sub_folders):
                folder = root.get_sub_folder(index)
                if folder.identifier == nid:
                    return folder
        return root

    def iter_folders(self) -> Iterator[tuple[str, object]]:
        """Alle Ordner unterhalb der persoenlichen Ordner als ("A/B", folder)."""

        def walk(folder, prefix: str):
            for index in range(folder.number_of_sub_folders):
                try:
                    child = folder.get_sub_folder(index)
                except OSError:
                    continue
                name = (child.name or f"Ordner {child.identifier}").replace("/", "\u2215")
                path = f"{prefix}/{name}" if prefix else name
                yield path, child
                yield from walk(child, path)

        yield from walk(self._ipm_root(), "")

    def folder_tree(self) -> list[dict]:
        """Hierarchische Ordnerliste im Format des Folder-Pickers."""
        nodes: dict[str, dict] = {}
        roots: list[dict] = []
        for path, folder in self.iter_folders():
            name = path.rsplit("/", 1)[-1]
            count = folder.number_of_sub_messages
            node = {
                "name": f"{name} ({count})" if count else name,
                "path": path,
                "raw_path": path,
                "selectable": True,
                "children": [],
            }
            nodes[path] = node
            parent = nodes.get(path.rsplit("/", 1)[0]) if "/" in path else None
            (parent["children"] if parent else roots).append(node)
        return roots

    # --- Nachrichten ---

    def read_item(
        self,
        message,
        *,
        with_attachments: bool,
        max_attachment_bytes: int,
    ) -> PstItem | None:
        """Liest eine Nachricht; None fuer Klassen ohne Indexwert (Kontakte, Aufgaben ...)."""
        props = Props(message, self.named)
        message_class = props.string(PR_MESSAGE_CLASS)
        kind = item_kind(message_class)
        if kind is None:
            return None
        return self._build_item(
            message, props, message_class, kind, with_attachments, max_attachment_bytes, depth=0
        )

    def _build_item(
        self, message, props, message_class, kind, with_attachments, max_bytes, depth
    ) -> PstItem:
        to, cc = self._recipients(message)
        body, mime, charset = self._body(message, props)
        item = PstItem(
            kind=kind,
            identifier=message.identifier,
            message_class=message_class,
            subject=props.string(PR_SUBJECT),
            sender=self._sender(props),
            to=to,
            cc=cc,
            date=(
                props.time(PR_CLIENT_SUBMIT_TIME)
                or props.time(PR_MESSAGE_DELIVERY_TIME)
                or props.time(PR_CREATION_TIME)
            ),
            body=body,
            body_mime=mime,
            body_charset=charset,
        )
        start = props.time((PSETID_APPOINTMENT, LID_APPOINTMENT_START))
        if start is not None:
            item.start = start
            item.end = props.time((PSETID_APPOINTMENT, LID_APPOINTMENT_END))
            item.all_day = bool(props.boolean((PSETID_APPOINTMENT, LID_APPOINTMENT_ALL_DAY)))
            item.location = props.string((PSETID_APPOINTMENT, LID_LOCATION))
            item.attendees = props.string((PSETID_APPOINTMENT, LID_ALL_ATTENDEES)) or ", ".join(
                part
                for part in (
                    props.string((PSETID_APPOINTMENT, LID_TO_ATTENDEES)),
                    props.string((PSETID_APPOINTMENT, LID_CC_ATTENDEES)),
                )
                if part
            )
            if props.boolean((PSETID_APPOINTMENT, LID_RECURRING)):
                item.recurrence = props.string((PSETID_APPOINTMENT, LID_RECURRENCE_PATTERN)) or "ja"
            flags = props.integer((PSETID_APPOINTMENT, LID_APPOINTMENT_STATE_FLAGS)) or 0
            item.canceled = bool(flags & ASF_CANCELED)
        if with_attachments:
            self._attachments(message, item, max_bytes, depth)
        return item

    @staticmethod
    def _address(name: str, smtp: str, fallback: str = "") -> str:
        address = smtp or (fallback if "@" in fallback else "")
        if name and address and name != address:
            return f"{name} <{address}>"
        return name or address

    def _sender(self, props: Props) -> str:
        sender = self._address(
            props.string(PR_SENDER_NAME),
            props.string(PR_SENDER_SMTP_ADDRESS),
            props.string(PR_SENDER_EMAIL_ADDRESS),
        )
        return sender or self._address(
            props.string(PR_SENT_REPRESENTING_NAME), props.string(PR_SENT_REPRESENTING_SMTP)
        )

    def _recipients(self, message) -> tuple[str, str]:
        to: list[str] = []
        cc: list[str] = []
        try:
            recipients = message.recipients
            count = recipients.number_of_record_sets if recipients is not None else 0
        except OSError:
            return "", ""
        for index in range(count):
            try:
                record_set = recipients.get_record_set(index)
            except OSError:
                continue
            values: dict[int, object] = {}
            for entry_index in range(record_set.number_of_entries):
                entry = record_set.get_entry(entry_index)
                try:
                    if entry.value_type in (PT_STRING8, PT_UNICODE):
                        values[entry.entry_type] = (entry.data_as_string or "").strip("\x00")
                    elif entry.value_type == PT_LONG:
                        values[entry.entry_type] = entry.data_as_integer
                except (OSError, UnicodeError):
                    continue
            address = self._address(
                str(values.get(PR_DISPLAY_NAME, "")),
                str(values.get(PR_SMTP_ADDRESS, "")),
                str(values.get(PR_EMAIL_ADDRESS, "")),
            )
            if not address:
                continue
            (cc if values.get(PR_RECIPIENT_TYPE) == RECIPIENT_CC else to).append(address)
        return ", ".join(to), ", ".join(cc)

    @staticmethod
    def _body(message, props: Props) -> tuple[bytes, str, str]:
        """Hauptbody wie beim IMAP-Scanner: Klartext mit Inhalt vor HTML, RTF zuletzt."""
        try:
            plain = message.plain_text_body or b""
        except OSError:
            plain = b""
        if plain.strip(b"\x00 \r\n\t"):
            return plain.rstrip(b"\x00"), "text/plain", "utf-8"
        try:
            html = message.html_body or b""
        except OSError:
            html = b""
        if html.strip(b"\x00 \r\n\t"):
            return (
                html.rstrip(b"\x00"),
                "text/html",
                codepage_to_charset(props.integer(PR_INTERNET_CPID)),
            )
        try:
            rtf = message.rtf_body or b""
        except OSError:
            rtf = b""
        if rtf:
            mime, text = decapsulate_rtf(rtf)
            return text.encode("utf-8"), mime, "utf-8"
        return b"", "text/plain", "utf-8"

    def _attachments(self, message, item: PstItem, max_bytes: int, depth: int) -> None:
        try:
            count = message.number_of_attachments
        except OSError:
            return
        for index in range(count):
            try:
                attachment = message.get_attachment(index)
                props = Props(attachment)
                if props.boolean(PR_ATTACHMENT_HIDDEN) or (
                    (props.integer(PR_ATTACH_FLAGS) or 0) & ATT_MHTML_REF
                ):
                    continue  # Inline-Bilder (Signaturen, eingebettete Grafiken)
                method = props.integer(PR_ATTACH_METHOD)
                if method == ATTACH_EMBEDDED_MSG:
                    self._embedded(attachment, item, max_bytes, depth)
                    continue
                if method not in (ATTACH_BY_VALUE, ATTACH_OLE, None):
                    continue  # Verweise auf externe Dateien
                size = attachment.size
                if not size or size > max_bytes:
                    item.skipped_attachments += 1
                    continue
                filename = (
                    _safe_attr(attachment, "long_filename")
                    or props.string(PR_ATTACH_FILENAME)
                    or props.string(PR_DISPLAY_NAME)
                    or f"anhang_{index + 1}"
                )
                attachment.seek_offset(0)
                item.attachments.append(PstAttachment(filename, attachment.read_buffer(size)))
            except OSError:
                item.skipped_attachments += 1

    def _embedded(self, attachment, item: PstItem, max_bytes: int, depth: int) -> None:
        """Angehaengte Mail als Textanhang ablegen, ihre Anhaenge mit uebernehmen."""
        if depth >= 3 or attachment.number_of_sub_items < 1:
            item.skipped_attachments += 1
            return
        inner = attachment.get_sub_item(0)
        props = Props(inner, self.named)
        message_class = props.string(PR_MESSAGE_CLASS) or "IPM.Note"
        embedded = self._build_item(inner, props, message_class, "mail", True, max_bytes, depth + 1)
        item.attachments.append(
            PstAttachment(f"{embedded.subject or 'Mail'}.txt", render_embedded(embedded))
        )
        item.attachments.extend(embedded.attachments)
        item.skipped_attachments += embedded.skipped_attachments

    def count_messages(self, folders: list[tuple[str, object]]) -> int:
        return sum(folder.number_of_sub_messages for _path, folder in folders)


def _safe_attr(obj, name: str) -> str:
    try:
        return (getattr(obj, name) or "").strip()
    except (OSError, UnicodeError):
        return ""


def format_local(value: datetime | None, *, date_only: bool = False) -> str:
    if value is None:
        return ""
    local = value.astimezone()
    return local.strftime("%Y-%m-%d" if date_only else "%Y-%m-%d %H:%M")


def render_embedded(item: PstItem) -> bytes:
    """Angehaengte Mail als einfacher Text (Header + Body)."""
    from kiara.ingestion.imap_scanner import _MailBody

    text, _metadata = _MailBody(item.body, item.body_mime, item.body_charset).convert()
    header = (
        f"Betreff: {item.subject}\n"
        f"Von: {item.sender}\n"
        f"An: {item.to}\n"
        f"Datum: {format_local(item.date)}\n"
        f"---\n"
    )
    return (header + text).encode("utf-8")
