# Seitenweise PDF-OCR für docling-serve

Dieses Image erweitert docling-serve 1.12.0 um den Ergebnisstatus fehlgeschlagener Aufträge und eine OCR für einzelne PDF-Seiten. KIara nutzt diese Schnittstellen über das konfigurierte DDU-Backend.

Bei einem bekannten Dokumentfehler erhalten Clients die Task-ID und die Parserursache in einer Antwort mit `status=failure`. Der Ergebnisabruf liefert für laufende Aufträge HTTP 202, für unbekannte Task-IDs HTTP 404 und für interne Workerfehler HTTP 503. Bei einem verwertbaren PDF-Ergebnis enthält das Dokument `source_page_count` und `missing_text_pages`.

KIara übernimmt vorhandenes Markdown, liest fehlenden Text zuerst mit pypdf und fordert bei aktivierter OCR nur die verbleibenden Seiten über `POST /v1/convert/pdf-pages/ocr` an. Der Endpunkt verwendet dieselbe Authentifizierung wie die Konvertierung. PDFium rendert die Seiten mit bis zu 150 dpi und 18 Megapixeln. Tesseract sucht auch verstreute Beschriftungen. Pro Anfrage gelten höchstens 128 Seiten und 128 MiB, pro Tesseract-Aufruf 45 Sekunden.

## Image bauen und prüfen

Der Build benötigt Docker, `apt-get`, `dpkg-deb` und Zugriff auf das Ubuntu-Paket `tesseract-ocr-deu=1:4.1.0-2build1`. Die Basis ist per Image-Digest festgelegt. Das Buildskript prüft die SHA-256-Summe der deutschen Sprachdaten. Der lokale Image-Name lässt sich über `DOCLING_IMAGE_TAG` setzen.

```sh
cd deploy/docling-page-ocr
./build.sh
docker run --rm -i --network none \
  -e DOCLING_DEVICE=cpu -e DOCLING_SERVE_LOAD_MODELS_AT_BOOT=false \
  --entrypoint /opt/app-root/bin/python \
  kiara/docling-serve-cu126:1.12.0-page-ocr-2 - < check_responses.py
```

Für den Betrieb das gebaute Image im bestehenden docling-serve-Dienst verwenden. Die HTTP-Tests prüfen die Task-Zustände und senden eine im Test erzeugte zweiseitige PDF an den OCR-Endpunkt. Für jede einzeln angeforderte Seite prüfen sie die Rückgabe der Beschriftung oder die Kennzeichnung der leeren Seite.
