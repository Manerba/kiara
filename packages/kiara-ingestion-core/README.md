# kiara-ingestion-core

Monorepo-internes Package fuer das Hauptprojekt `kiara`.

## Contract

- kein stabilisiertes Standalone-SDK
- darf `kiara.shared.*`, `kiara.repositories.*` und `kiara-db` direkt voraussetzen
- dient nur dazu, die im Monorepo extrahierten Ingestion-Bausteine klarer zu schneiden

## Verifikation

Die package-lokale Smoke-Suite laeuft vom Monorepo-Root aus ueber:

```bash
pytest -q packages/kiara-ingestion-core/tests
```
