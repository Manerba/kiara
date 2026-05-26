# Mitwirken an Kiara

Danke für dein Interesse an Kiara! Dieses Dokument beschreibt, wie du das Projekt
lokal aufsetzt, welche Konventionen gelten und wie Beiträge eingereicht werden.

> **Hinweis zur Lizenz:** Kiara steht unter der **AGPL-3.0-or-later**. Beiträge
> werden unter derselben Lizenz aufgenommen (*inbound = outbound*). Du behältst
> das Urheberrecht an deinem Beitrag; eine CLA oder Rechteübertragung ist
> **nicht** erforderlich. Details und die DCO-Bestätigung stehen unter
> [Lizenz, Urheberrecht und DCO](#lizenz-urheberrecht-und-dco).

## Entwicklungsumgebung

Kiara benötigt **Python 3.12** und **PostgreSQL**. Optionale Funktionen
(Reranking, CAD, Vision) ziehen weitere Abhängigkeiten nach.

```bash
# Lokale Entwicklungs-venv mit Dev-Tools (ruff, pre-commit, pytest) aufsetzen
make dev

# pre-commit-Hooks installieren (laufen vor jedem Commit)
make pre-commit-install
```

`make dev` erstellt ein `.venv/` im Projektverzeichnis und installiert die
gepinnten Abhängigkeiten aus `requirements-dev.lock` (CPU-only Torch).

## Vor jedem Pull Request

Lass das vollständige Quality-Gate durchlaufen — es ist dieselbe Prüfung, die
auch die CI ausführt:

```bash
make quality-gate
```

Das umfasst: Lockfile-Konsistenz, `pip check`, Compile-Check, **Ruff** Lint +
Format-Check, Pytest-Collection, Smoke-Tests und die pre-commit-Hooks.

Einzelschritte bei Bedarf:

```bash
make lint           # Ruff-Lint
make format         # Ruff-Format anwenden
make format-check   # Format nur prüfen
make smoke-test     # schnelle Smoke-Tests
make test           # vollständige Test-Suite
```

## Code-Konventionen

Diese Regeln sind verbindlich und werden teils von Ruff erzwungen:

- **Kein `__all__`.** Öffentliche vs. private API wird ausschließlich über die
  Underscore-Konvention signalisiert (`_name` = intern).
- **Keine Wildcard-Importe.** `from modul import *` wird nicht verwendet; alle
  Importe sind explizit.
- **Konfiguration nie hardcoden.** Alle Pfade, URLs, Modellnamen und Limits
  kommen aus der Konfiguration (`config/kiara.yaml` als Vorlage, zur Laufzeit
  `/etc/kiara/kiara.yaml`).
- **Lazy Imports** für optionale Abhängigkeiten (z. B. `pyodbc`, `psycopg2`,
  `qdrant-client`), damit eine Minimalinstallation lauffähig bleibt.
- **Deutschsprachige, nutzersichtbare Texte** (Prompts, UI, Meldungen) immer mit
  korrekten Umlauten (`ä ö ü ß`), nicht mit ASCII-Ersatz. In Code-Bezeichnern,
  Logs und Dateinamen ist ASCII erlaubt.
- **Code-Stil:** Ruff mit Zeilenlänge 100; die aktiven Regeln stehen in
  `pyproject.toml`.

## Commits und Versionierung

- Schreibe aussagekräftige Commit-Nachrichten, idealerweise mit einer Kategorie
  im Präfix, z. B. `[Bugfix]`, `[Feature]`, `[Refactoring]`, `[Docs]`,
  `[Packaging]`, `[Tooling]`.
- Die Version folgt dem Schema `0.0.1+N` (Basisversion fest, Build-Nummer zählt
  hoch). Du musst die Version in einem PR nicht selbst erhöhen.
- **Signiere deine Commits** mit `git commit -s` (DCO-Bestätigung, siehe
  [Lizenz, Urheberrecht und DCO](#lizenz-urheberrecht-und-dco)).

## Pull-Request-Prozess

1. Branch von `master` abzweigen.
2. Änderung umsetzen, `make quality-gate` grün bekommen.
3. PR mit klarer Beschreibung öffnen (Was? Warum? Wie getestet?).
4. Verlinke ggf. ein Issue.

## Lizenz, Urheberrecht und DCO

Kiara ist **AGPL-3.0-or-later**-lizenziert. Es gilt **inbound = outbound**: Dein
Beitrag wird unter genau dieser Lizenz Teil des Projekts. Es gibt **keine
Contributor License Agreement (CLA)** und keine Rechteübertragung — du behältst
das Urheberrecht an deinem Code, Kiara erhält lediglich das AGPL-Nutzungsrecht.

Damit die Herkunft jedes Beitrags nachvollziehbar bleibt, nutzen wir die
**Developer Certificate of Origin (DCO)** — eine schlanke Herkunfts-Bestätigung,
keine Rechteabtretung. Du bestätigst sie, indem du deine Commits signierst:

```bash
git commit -s -m "[Bugfix] Kurzbeschreibung"
```

Das hängt eine Zeile `Signed-off-by: Dein Name <mail@example.org>` an den Commit
und bezeugt, dass du den Beitrag unter der AGPL beisteuern darfst (Wortlaut unter
<https://developercertificate.org>). Achte darauf, dass `git config user.name`
und `git config user.email` deinen echten Namen und eine erreichbare Adresse
enthalten.

Die Urheberrechtshinweise und die Lizenzen der mitgelieferten Drittkomponenten
sind in der Datei [`NOTICE`](NOTICE) zusammengefasst.

## Sicherheitslücken

Melde **keine** Sicherheitslücken über öffentliche Issues. Der korrekte Weg ist
in [SECURITY.md](SECURITY.md) beschrieben.
