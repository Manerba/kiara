# Prozessgrenzen des DB-Pools

DB-Verbindungen und Leases gehören genau einem Prozess. Neue Worker sollen
mit `spawn` beziehungsweise einem unbelasteten `forkserver` starten und erst
im Worker Verbindungen öffnen. KIaras bestehende Ingestion verwendet keinen
direkten `os.fork()`-Pfad; diese Korrektur führt keinen ein.

Python 3.12 warnt vor `fork()` bei mehreren Threads. Ein Child-Hook macht die
übrigen Python-, TLS-, Allocator- oder Bibliothekszustände nicht forksicher.
Insbesondere ist Fork während beliebiger Verbindungsaufbauten, Resets oder
anderer fremder Threadaktivität kein unterstütztes allgemeines Prozessmodell.
Psycopg verlangt neue Verbindungen nach einem Fork:

- https://docs.python.org/3.12/library/os.html#os.fork
- https://docs.python.org/3.12/library/multiprocessing.html#contexts-and-start-methods
- https://www.psycopg.org/docs/usage.html#thread-and-process-safety

## Begrenzte defensive Behandlung eines geerbten Pools

Der Child-Hook trennt bekannte Idle- und ausgeliehene Verbindungen aus allen
Poolgenerationen ab. Er schließt nur die im Kind duplizierten Dateideskriptoren
mit `os.close`, verwirft die Poolreferenzen und ersetzt geerbte Python-Locks.
Er ruft weder `connection.close()` noch `reset()` auf. Die alte PID bleibt an
geerbten Pools und Wrappern stehen; deren Verwendung wird zurückgewiesen.
Ein neuer Checkout erstellt einen eigenen Pool mit eigener PID und DB-Session.

Der frühere explizite `close()` rief Psycopgs `conn_close()` auf und wartete
auf einen Mutex, dessen Thread im Kind nicht mehr existierte. Der Psycopg-
Destruktor hat dagegen auf dem eingesetzten Linux/GCC-Build eine PID-Prüfung:
`connection_dealloc()` überspringt `conn_close()` in fremden Prozessen.
Deshalb dürfen Python-Referenzen freigegeben werden, ohne diesen blockierenden
Close-Pfad nachträglich in die Garbage Collection zu verschieben. Die
Deskriptoren werden vorher abgetrennt; spätere Freigabe schließt auch keine
inzwischen unter derselben Nummer neu geöffneten Deskriptoren.

Geprüfte Treiberfassung: `psycopg2-binary 2.9.12`, CPython 3.12.3, Linux/GCC.
Quellen zur tatsächlich eingesetzten Fassung:

- [PID-Guard im Destruktor](https://github.com/psycopg/psycopg2/blob/2.9.12/psycopg/connection_type.c#L1285-L1324)
- [CONN_CHECK_PID für GCC](https://github.com/psycopg/psycopg2/blob/2.9.12/psycopg/config.h#L37-L41)
- [Expliziter Close mit Mutex](https://github.com/psycopg/psycopg2/blob/2.9.12/psycopg/connection_int.c)

Das übersprungene `PQfinish` bedeutet, dass geerbter nativer libpq-Speicher
erst mit dem Kindprozess vom Betriebssystem zurückgenommen wird. Es gibt
keine künstlich dauerhaft gehaltenen Python-Referenzen; normale Leases im
eigenen Prozess werden weiterhin vollständig zurückgegeben und bereinigt.
Dies ist ein weiterer Grund für frische Interpreter bei langlebigen Workern.
Geerbte Raw-Cursor, Raw-Verbindungen und vor dem Fork gebundene Methoden dürfen
im Kind nicht verwendet werden. Der öffentliche Wrapper prüft die Prozess-ID.

## Regressionsnachweis

`tests/test_db_pool.py` startet `tests/helpers/db_fork_lifecycle.py` in einem
eigenen Interpreter und einer eigenen Prozessgruppe, ausschließlich gegen
`kiara_test`. Der Test beobachtet `PgSleep` im anderen Thread vor dem Fork.
Er prüft alte und aktuelle Pools, Ablehnung geerbter Wrapper, abgetrennte
Deskriptoren, Weakref-Freigabe von Verbindungen und benannten Cursorn,
Deskriptor-Wiederverwendung, eigene Kind-Sessions, Session-Reset, explizites
Pool-Cleanup und normalen Interpreter-Shutdown einschließlich später
freigegebener geerbter Objekte. Elternabfrage, Elternsession und Elterncursor
bleiben intakt. Kind-Prozessende und verschwundene Kind-DB-Session werden
geprüft; das Kind nutzt kein `os._exit()`.

Die Kindfrist beträgt vier Sekunden, die äußere Probe fünfzehn Sekunden.
Cleanup signalisiert ausschließlich das eigene Kind beziehungsweise die
private Prozessgruppe. Diese Regressionen müssen bei Änderungen am Treiber
oder an der Plattform erneut laufen. Ihr Bestehen ist keine allgemeine
Garantie für multithreaded Fork.
