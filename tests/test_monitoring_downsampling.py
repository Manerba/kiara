"""Tests fuer System-Monitor: Downsampling (3-Stufen-Komprimierung).

Testet run_downsampling, Aggregation, leere Tabelle — 4 Unit-Tests.
"""

from unittest.mock import MagicMock, patch

import kiara.shared.db as _db

_db._dsn = "postgresql://kiara:kiara@localhost:5432/kiara_test"


class TestDownsampling:
    @patch("kiara.monitoring.downsampling.db_transaction")
    def test_compress_raw_to_5min(self, mock_db_tx):
        """raw→5min INSERT + DELETE in einer Transaktion."""
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.rowcount = 42
        mock_conn.execute.return_value = mock_result
        mock_db_tx.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db_tx.return_value.__exit__ = MagicMock(return_value=False)

        from kiara.monitoring.downsampling import _compress_raw_to_5min

        deleted = _compress_raw_to_5min()

        assert deleted == 42
        # 2 SQL-Statements: INSERT + DELETE
        assert mock_conn.execute.call_count == 2
        sqls = [c[0][0] for c in mock_conn.execute.call_args_list]
        assert "INSERT INTO system_metrics" in sqls[0]
        assert "DELETE FROM system_metrics" in sqls[1]
        assert "'raw'" in sqls[1]

    @patch("kiara.monitoring.downsampling.db_transaction")
    def test_compress_5min_to_1h(self, mock_db_tx):
        """5min→1h Komprimierung."""
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.rowcount = 10
        mock_conn.execute.return_value = mock_result
        mock_db_tx.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db_tx.return_value.__exit__ = MagicMock(return_value=False)

        from kiara.monitoring.downsampling import _compress_5min_to_1h

        deleted = _compress_5min_to_1h()

        assert deleted == 10
        sqls = [c[0][0] for c in mock_conn.execute.call_args_list]
        assert "'5m'" in sqls[0]
        assert "'5m'" in sqls[1]

    @patch("kiara.monitoring.downsampling.db_transaction")
    def test_delete_expired(self, mock_db_tx):
        """Loescht 1h-Daten aelter als Retention."""
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.rowcount = 5
        mock_conn.execute.return_value = mock_result
        mock_db_tx.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db_tx.return_value.__exit__ = MagicMock(return_value=False)

        from kiara.monitoring.downsampling import _delete_expired

        deleted = _delete_expired()

        assert deleted == 5
        sql = mock_conn.execute.call_args[0][0]
        assert "'1h'" in sql
        assert "DELETE" in sql

    @patch("kiara.monitoring.downsampling._delete_expired", return_value=0)
    @patch("kiara.monitoring.downsampling._compress_5min_to_1h", return_value=0)
    @patch("kiara.monitoring.downsampling._compress_raw_to_5min", return_value=0)
    def test_run_downsampling_empty(self, mock_r5, mock_51h, mock_del):
        """run_downsampling bei leerer Tabelle → alle Schritte laufen, 0 geloescht."""
        from kiara.monitoring.downsampling import run_downsampling

        stats = run_downsampling()

        assert stats["raw_to_5m"] == 0
        assert stats["5m_to_1h"] == 0
        assert stats["deleted_old"] == 0
        mock_r5.assert_called_once()
        mock_51h.assert_called_once()
        mock_del.assert_called_once()
