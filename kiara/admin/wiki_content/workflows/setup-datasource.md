# Workflow: Datenquelle einrichten

## Ueberblick

Dieser Workflow fuehrt durch die Einrichtung einer neuen Datenquelle. Er umfasst die Wahl des Konnektor-Typs, die Konfiguration der Verbindungsparameter, den Verbindungstest und die Anlage der Datenquelle. Am Ende ist die Datenquelle bereit fuer die Indexierung.

## Voraussetzungen

- Mindestens ein Embedding-Backend muss konfiguriert und aktiv sein (siehe Workflow "KI-Backend einrichten")
- Zugangsdaten fuer die Datenquelle (Pfad, Server, Benutzername, Passwort je nach Typ)
- Admin-Zugang zur KIara-Admin-UI oder zum Admin-Chat

## Unterstuetzte Konnektoren

KIara unterstuetzt diese Konnektor-Typen:

| Typ | Beschreibung | Typischer Einsatz |
|-----|-------------|------------------|
| **local** | Lokales Dateisystem | Dokumente auf dem KIara-Server |
| **smb** | Windows/Samba-Netzwerkfreigaben | Abteilungs-Laufwerke, Fileserver |
| **webdav** | WebDAV (z.B. Nextcloud) | Cloud-Speicher, Kollaborationsplattformen |
| **imap** | E-Mail via IMAP | E-Mail-Postfaecher, Archiv-Mailboxen |
| **exchange_online** | Exchange Online (OAuth2) | Microsoft 365, Exchange Online |
| **sql** | SQL-Datenbank | ERP-Daten, Ticket-Systeme, Wissensdatenbanken |

Beim Typ `sql` wird die konkrete Datenbank ueber `config.driver`
gewaehlt. Gueltige Driver-Optionen sind `mssql`, `postgresql` und `mysql`.

## Schritt 1: Konnektor-Typen anzeigen

Der Agent ruft `datasource.connector_types` auf, um die verfuegbaren Konnektoren mit ihren Konfigurationsfeldern anzuzeigen. Das zeigt fuer jeden Typ:

- Name und Beschreibung
- Pflichtfelder und optionale Felder
- Feldtypen (Text, Passwort, Checkbox, Zahl)
- Standard-Werte

## Schritt 2: Datenquelle beschreiben

Sammle die notwendigen Informationen je nach Konnektor-Typ:

### Beispiel: SMB-Freigabe

| Parameter | Beschreibung | Beispielwert |
|-----------|-------------|-------------|
| `name` | Anzeigename | "Technik-Dokumente" |
| `type` | Konnektor-Typ | `smb` |
| `config.server` | Hostname/IP des Fileservers | `fileserver.firma.local` |
| `config.share` | Freigabe-Name | `Technik` |
| `config.username` | Service-Account | `svc-kiara-read` |
| `config.password` | Passwort | (wird verschluesselt gespeichert) |
| `config.domain` | Windows-Domaene (optional) | `FIRMA` |
| `config.sub_path` | Unterverzeichnis (optional) | `Projekte/2025` |

### Beispiel: Lokales Verzeichnis

| Parameter | Beschreibung | Beispielwert |
|-----------|-------------|-------------|
| `name` | Anzeigename | "Technische Dokumentation" |
| `type` | Konnektor-Typ | `local` |
| `config.path` | Absoluter Pfad | `/data/dokumente/technik` |

### Optionale Ingestion-Parameter

| Parameter | Standard | Beschreibung |
|-----------|----------|-------------|
| `ds_chunk_strategy` | `auto` | Chunk-Strategie (auto, prose, code, email, custom) |
| `ds_chunk_size` | (auto) | Chunk-Groesse bei custom-Strategie |
| `ds_chunk_overlap` | (auto) | Ueberlappung bei custom-Strategie |
| `ds_file_types` | (alle) | Dateifilter (z.B. `.pdf,.docx,.xlsx`) |
| `ds_exclude_patterns` | (leer) | Ausschlussmuster (Glob-Syntax) |
| `ds_ddu_enabled` | Aus | Document Understanding aktivieren |
| `ds_vlm_enabled` | Aus | Vision Language Model aktivieren |
| `sync_interval` | `global` | Sync-Intervall (manual, daily, weekly, monthly) |

## Schritt 3: Verbindung testen

`datasource.test_connection` prueft, ob die Konfiguration gueltig ist und eine Verbindung hergestellt werden kann:

- **Lokal:** Verzeichnis existiert und ist lesbar
- **SMB:** Server erreichbar, Authentifizierung erfolgreich, Freigabe zugreifbar
- **WebDAV:** HTTP-Verbindung erfolgreich, Basis-Pfad existiert
- **IMAP/Exchange:** Server erreichbar, Login erfolgreich, Ordner vorhanden
- **SQL:** Datenbankverbindung erfolgreich, Query ausfuehrbar

Bei Fehlern gibt der Agent spezifische Hinweise:

| Fehler | Ursache | Loesung |
|--------|---------|---------|
| "Connection refused" | Server nicht erreichbar | Netzwerk/Firewall pruefen |
| "Authentication failed" | Falsche Credentials | Benutzername/Passwort pruefen |
| "Share not found" | Freigabe existiert nicht | Freigabe-Namen pruefen (`smbclient -L //server`) |
| "Permission denied" | Keine Leserechte | Service-Account-Berechtigungen pruefen |

## Schritt 4: Datenquelle anlegen

`datasource.create` legt die Datenquelle an. Das Feld `config` wird auf `connector_config` gemappt. Die Datenquelle erhaelt eine ID (`ds_id`), die fuer die Indexierung benoetigt wird.

Beispiel im Admin-Chat:

```
Admin: Erstelle eine SMB-Datenquelle "Technik-Dokumente" mit Server
       fileserver.firma.local und Freigabe "Technik".

KIara: [Tool: datasource.create]
       Datenquelle "Technik-Dokumente" erfolgreich angelegt (ID: 12).
       Typ: SMB-Freigabe, Quelle: fileserver.firma.local:/Technik
       Chunk-Strategie: auto, Sync: global

       Soll ich die Verbindung testen?
```

## Schritt 5: Zugriffsrechte konfigurieren (optional)

Falls nur bestimmte Benutzergruppen Zugriff auf die Datenquelle haben sollen:

- `datasource.access.update` -- Gruppen und/oder Benutzer zuweisen
- Ohne Zuweisung haben alle Benutzer Zugriff (Standard)

## Naechste Schritte

Nach der Anlage kann die Indexierung gestartet werden:

1. **Indexierung starten:** `jobs.start` -- Inkrementelle Indexierung (empfohlen fuer den ersten Lauf)
2. **Voll-Indexierung:** `jobs.start_full` -- Alle Dateien verarbeiten (nach Config-Aenderungen)
3. **Status pruefen:** `jobs.status` -- Fortschritt und Ergebnis kontrollieren

Siehe Workflow "Indexierung einrichten" fuer Details.

## Verwandte Funktionen

- `datasource.list` -- Alle Datenquellen anzeigen
- `datasource.update` -- Datenquelle bearbeiten
- `datasource.delete` -- Datenquelle loeschen (erfordert Bestaetigung)
- `datasource.connector_types` -- Verfuegbare Konnektor-Typen
- `datasource.test_connection` -- Verbindungstest
