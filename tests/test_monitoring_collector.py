"""Tests fuer System-Monitor: Collector (Sampling, Enrichment, SSE, Persistenz).

Testet _sample, _enrich, broadcast/subscribe, _persist, get_latest — 14 Unit-Tests.
"""

import asyncio
import time
from unittest.mock import MagicMock, patch

import kiara.shared.db as _db

_db._dsn = "postgresql://kiara:kiara@localhost:5432/kiara_test"


class TestSample:
    @patch("kiara.monitoring.collector.psutil", create=True)
    def test_sample_returns_all_keys(self, mock_psutil):
        """Alle 12 Metrik-Keys sind im Ergebnis."""
        import kiara.monitoring.collector as col

        col._prev_net = None
        col._prev_ts = None

        mock_psutil.cpu_percent.return_value = 42.5
        mock_psutil.virtual_memory.return_value = MagicMock(used=4_000_000_000, total=8_000_000_000)
        mock_psutil.swap_memory.return_value = MagicMock(used=100_000_000, total=2_000_000_000)
        mock_psutil.getloadavg.return_value = (1.5, 1.2, 0.8)
        mock_psutil.disk_usage.return_value = MagicMock(used=50_000_000_000, total=100_000_000_000)
        mock_psutil.net_io_counters.return_value = MagicMock(bytes_sent=1000, bytes_recv=2000)

        result = col._sample()

        expected_keys = {
            "cpu_percent",
            "ram_used_bytes",
            "ram_total_bytes",
            "swap_used_bytes",
            "swap_total_bytes",
            "load_1m",
            "load_5m",
            "load_15m",
            "net_sent_bps",
            "net_recv_bps",
            "disk_used_bytes",
            "disk_total_bytes",
        }
        assert set(result.keys()) == expected_keys
        assert result["cpu_percent"] == 42.5
        assert result["ram_used_bytes"] == 4_000_000_000
        assert result["load_1m"] == 1.5

    @patch("kiara.monitoring.collector.psutil", create=True)
    def test_sample_network_delta(self, mock_psutil):
        """Netzwerk-Rate wird als Delta zwischen zwei Aufrufen berechnet."""
        import kiara.monitoring.collector as col

        col._prev_net = None
        col._prev_ts = None

        mock_psutil.cpu_percent.return_value = 10.0
        mock_psutil.virtual_memory.return_value = MagicMock(used=1, total=2)
        mock_psutil.swap_memory.return_value = MagicMock(used=0, total=0)
        mock_psutil.getloadavg.return_value = (0, 0, 0)
        mock_psutil.disk_usage.return_value = MagicMock(used=0, total=1)

        mock_psutil.net_io_counters.return_value = MagicMock(bytes_sent=1000, bytes_recv=2000)
        r1 = col._sample()
        assert r1["net_sent_bps"] == 0.0

        mock_psutil.net_io_counters.return_value = MagicMock(bytes_sent=2000, bytes_recv=4000)
        r2 = col._sample()
        assert r2["net_sent_bps"] > 0
        assert r2["net_recv_bps"] > 0


class TestEnrich:
    @patch("kiara.monitoring.collector.psutil", create=True)
    def test_enrich_adds_top_processes(self, mock_psutil):
        """Top-5 Prozesse werden angehaengt."""
        mock_procs = []
        for i in range(7):
            p = MagicMock()
            p.info = {
                "pid": 100 + i,
                "name": f"proc{i}",
                "memory_info": MagicMock(rss=(7 - i) * 100_000_000),
            }
            mock_procs.append(p)
        mock_psutil.process_iter.return_value = mock_procs
        mock_psutil.boot_time.return_value = time.time() - 3600

        import kiara.monitoring.collector as col

        result = col._enrich({"cpu_percent": 10.0})

        assert len(result["top_processes"]) == 5
        assert result["top_processes"][0]["name"] == "proc0"
        assert "ts" in result
        assert result["uptime_seconds"] > 0

    @patch("kiara.monitoring.collector.psutil", create=True)
    def test_enrich_defensive_on_exception(self, mock_psutil):
        """Bei psutil-Fehler crasht _enrich nicht."""
        mock_psutil.process_iter.side_effect = RuntimeError("access denied")
        mock_psutil.boot_time.side_effect = RuntimeError("no boot time")

        import kiara.monitoring.collector as col

        result = col._enrich({"cpu_percent": 5.0})

        assert result["top_processes"] == []
        assert result["boot_time"] == 0
        assert "ts" in result


class TestSSEPubSub:
    def test_subscribe_unsubscribe(self):
        """Subscribe gibt Queue, unsubscribe entfernt sie."""
        import kiara.monitoring.collector as col

        loop = asyncio.new_event_loop()

        async def _test():
            q = col.subscribe()
            assert isinstance(q, asyncio.Queue)
            col.unsubscribe(q)
            with col._sub_lock:
                assert not any(qq is q for qq, _ in col._subscribers)

        loop.run_until_complete(_test())
        loop.close()

    def test_broadcast_delivers(self):
        """_broadcast liefert Daten an Subscriber."""
        import kiara.monitoring.collector as col

        loop = asyncio.new_event_loop()

        async def _test():
            q = col.subscribe()
            col._broadcast({"cpu_percent": 99.9})
            await asyncio.sleep(0.05)
            assert not q.empty()
            assert q.get_nowait()["cpu_percent"] == 99.9
            col.unsubscribe(q)

        loop.run_until_complete(_test())
        loop.close()

    def test_broadcast_overflow_drops_oldest(self):
        """Bei voller Queue wird aeltester Eintrag verworfen."""
        import kiara.monitoring.collector as col

        loop = asyncio.new_event_loop()

        async def _test():
            q = col.subscribe()
            for i in range(7):
                col._broadcast({"i": i})
                await asyncio.sleep(0.02)
            count = 0
            while not q.empty():
                q.get_nowait()
                count += 1
            assert count <= 5
            col.unsubscribe(q)

        loop.run_until_complete(_test())
        loop.close()

    def test_broadcast_dead_subscriber_cleanup(self):
        """Geschlossener Event-Loop wird aus _subscribers entfernt."""
        import kiara.monitoring.collector as col

        loop = asyncio.new_event_loop()
        q = asyncio.Queue(maxsize=5)
        with col._sub_lock:
            col._subscribers.add((q, loop))
        loop.close()

        col._broadcast({"test": True})

        with col._sub_lock:
            assert not any(qq is q for qq, _ in col._subscribers)


class TestPersist:
    @patch("kiara.monitoring.collector.db_transaction")
    def test_persist_inserts_metrics(self, mock_db_tx):
        """_persist fuehrt INSERT mit korrekten Parametern aus."""
        mock_conn = MagicMock()
        mock_db_tx.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db_tx.return_value.__exit__ = MagicMock(return_value=False)

        import kiara.monitoring.collector as col

        metrics = {
            "cpu_percent": 25.5,
            "ram_used_bytes": 4e9,
            "ram_total_bytes": 8e9,
            "swap_used_bytes": 0,
            "swap_total_bytes": 2e9,
            "load_1m": 1.0,
            "load_5m": 0.8,
            "load_15m": 0.5,
            "net_sent_bps": 1024.0,
            "net_recv_bps": 2048.0,
            "disk_used_bytes": 50e9,
            "disk_total_bytes": 100e9,
        }
        col._persist(metrics)

        sql = mock_conn.execute.call_args[0][0]
        params = mock_conn.execute.call_args[0][1]
        assert "INSERT INTO system_metrics" in sql
        assert params["cpu_percent"] == 25.5


class TestGetLatest:
    def test_returns_copy(self):
        """get_latest gibt Kopie zurueck, keine Referenz."""
        import kiara.monitoring.collector as col

        with col._lock:
            col._latest = {"cpu_percent": 33.3}
        result = col.get_latest()
        result["cpu_percent"] = 99.9
        with col._lock:
            assert col._latest["cpu_percent"] == 33.3

    def test_empty(self):
        """Leerer Collector gibt leeres Dict."""
        import kiara.monitoring.collector as col

        with col._lock:
            col._latest = {}
        assert col.get_latest() == {}
