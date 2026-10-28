#!/usr/bin/env python3
"""Projekt 64: begrenzte Vorschau speichern und je DS den naechsten Modelljob einreihen.

Beispiele:
  python scripts/migrate_html_p64.py preview --ds-id 25 --path /bestand/a.html --manifest plan.json
  python scripts/migrate_html_p64.py enqueue-next --manifest plan.json

enqueue-next nach Abschluss erneut explizit aufrufen. Es gibt keinen neuen Scheduler.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preview", "enqueue-next"), nargs="?", default="preview")
    parser.add_argument("--ds-id", type=int, action="append", default=[])
    parser.add_argument("--path", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None, help="Maximale Pfade je DS/Modell")
    parser.add_argument("--manifest", type=Path, help="Vorschaudatei schreiben bzw. einlesen")
    args = parser.parse_args()

    from kiara.config import load_config
    from kiara.ingestion.html_migration import enqueue_next, preview_migration, validate_selection
    from kiara.shared.db import init_db

    config = load_config()
    # Die Vorschau darf keine Schema-/Datenmigration beim DB-Setup ausloesen.
    init_db(config["database"]["postgresql"]["dsn"], force=True)
    if args.action == "preview":
        if not args.ds_id:
            parser.error("preview erfordert mindestens eine explizite --ds-id")
        result = preview_migration(config, args.ds_id, paths=args.path, limit=args.limit)
        rendered = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2)
        if args.manifest:
            with args.manifest.open("x", encoding="utf-8") as stream:
                stream.write(rendered + "\n")
    else:
        if not args.manifest or args.ds_id or args.path or args.limit is not None:
            parser.error("enqueue-next erfordert ausschliesslich das gespeicherte --manifest")
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        if manifest.get("unresolved"):
            parser.error("Vorschau enthaelt unaufgeloeste Bestandsquellen")
        selections = manifest.get("selections", [manifest])
        if not selections:
            parser.error("Manifest enthaelt keine betroffene Dateiauswahl")
        # Gesamte Eingabe vor dem ersten Enqueue pruefen.
        selections = [validate_selection(selection) for selection in selections]
        if len({selection["datasource_id"] for selection in selections}) != len(selections):
            parser.error("Manifest enthaelt mehrere Auswahlen derselben Datenquelle")
        result = [enqueue_next(selection) for selection in selections]
        rendered = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2)
    print(rendered)


if __name__ == "__main__":
    main()
