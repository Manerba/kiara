"""Tests fuer die eigenstaendige 250-ms-Kurzzeithistorie."""

import asyncio
import concurrent.futures
import unittest
from collections import namedtuple

import kiara.monitoring.high_res_history as high_res

CpuTimes = namedtuple(
    "CpuTimes",
    "user nice system idle iowait irq softirq steal",
)
VirtualMemory = namedtuple("VirtualMemory", "percent used")
DiskCounters = namedtuple("DiskCounters", "read_bytes write_bytes")
NetCounters = namedtuple("NetCounters", "bytes_sent bytes_recv")


class _SequencePsutil:
    def __init__(self, cpu, memory, disk, net=None):
        self.cpu = iter(cpu)
        self.memory = iter(memory)
        self.disk = iter(disk)
        self.net = iter(net or [NetCounters(0, 0)] * 100)

    def cpu_times(self):
        return next(self.cpu)

    def virtual_memory(self):
        return next(self.memory)

    def disk_io_counters(self):
        return next(self.disk)

    def net_io_counters(self):
        return next(self.net)


class CpuHelpersTests(unittest.TestCase):
    def test_cpu_delta_excludes_idle(self):
        previous = (100.0, 80.0)
        current = (101.0, 80.25)
        self.assertEqual(high_res._cpu_usage_percent(previous, current), 75.0)

    def test_cpu_delta_rejects_counter_reset(self):
        self.assertIsNone(high_res._cpu_usage_percent((100.0, 80.0), (90.0, 70.0)))

    def test_deadline_skips_missed_intervals_without_catchup(self):
        self.assertEqual(high_res._next_deadline(10.0, 10.25, 0.25), (10.25, 0))
        self.assertEqual(high_res._next_deadline(10.0, 10.80, 0.25), (11.0, 3))


class HighResMetricCollectorTests(unittest.TestCase):
    def test_network_delta_preserves_small_rates_and_counter_resets(self):
        psutil = _SequencePsutil(
            cpu=[],
            memory=[],
            disk=[],
            net=[
                NetCounters(1000, 2000),
                NetCounters(1256, 2512),
                NetCounters(0, 0),
                NetCounters(256, 512),
            ],
        )
        collector = high_res.HighResMetricCollector(psutil_module=psutil)
        self.assertEqual(collector._sample_net(1), (None, None, "warming_up"))
        sent, received, state = collector._sample_net(1.25)
        self.assertAlmostEqual(sent, 1024 / 1024**2, places=6)
        self.assertAlmostEqual(received, 2048 / 1024**2, places=6)
        self.assertEqual(state, "ok")
        buffer = high_res.HighResHistoryBuffer()
        public = buffer.append({"timestamp": 1, "values": {"net_sent_mb_s": sent}}, monotonic_now=1)
        self.assertEqual(public["values"]["net_sent_mb_s"], sent)
        self.assertEqual(collector._sample_net(1.5), (None, None, "reset"))
        self.assertEqual(collector._sample_net(1.75)[2], "ok")

    def test_independent_counter_baselines_and_flat_shape(self):
        mib = 1024**2
        psutil = _SequencePsutil(
            cpu=[
                CpuTimes(10, 0, 10, 80, 0, 0, 0, 0),
                CpuTimes(10.15, 0, 10.05, 80.05, 0, 0, 0, 0),
            ],
            memory=[
                VirtualMemory(50.0, 4 * 1024**3),
                VirtualMemory(51.0, 4.25 * 1024**3),
            ],
            disk=[
                DiskCounters(10 * mib, 20 * mib),
                DiskCounters(11 * mib, 20.5 * mib),
            ],
        )
        collector = high_res.HighResMetricCollector(
            psutil_module=psutil,
            loadavg=lambda: (0.75, 0.5, 0.25),
        )

        first = collector.sample(timestamp=100.0, monotonic_now=10.0)
        second = collector.sample(timestamp=100.25, monotonic_now=10.25)

        self.assertEqual(set(second["values"]), set(high_res.METRIC_KEYS))
        self.assertIsNone(first["values"]["cpu_usage"])
        self.assertIsNone(first["values"]["disk_read_mb_s"])
        self.assertEqual(first["states"]["cpu"], "warming_up")
        self.assertEqual(first["states"]["disk"], "warming_up")
        self.assertEqual(second["values"]["cpu_usage"], 80.0)
        self.assertEqual(second["values"]["cpu_load_1m"], 0.75)
        self.assertEqual(second["values"]["memory_usage"], 51.0)
        self.assertEqual(second["values"]["memory_used_gb"], 4.25)
        self.assertEqual(second["values"]["disk_read_mb_s"], 4.0)
        self.assertEqual(second["values"]["disk_write_mb_s"], 2.0)
        self.assertEqual(second["values"]["net_sent_mb_s"], 0.0)
        self.assertEqual(second["values"]["net_recv_mb_s"], 0.0)

    def test_counter_reset_creates_gap_and_rebaselines(self):
        psutil = _SequencePsutil(
            cpu=[
                CpuTimes(10, 0, 10, 80, 0, 0, 0, 0),
                CpuTimes(5, 0, 5, 40, 0, 0, 0, 0),
            ],
            memory=[VirtualMemory(1, 1), VirtualMemory(1, 1)],
            disk=[DiskCounters(100, 100), DiskCounters(50, 50)],
        )
        collector = high_res.HighResMetricCollector(
            psutil_module=psutil,
            loadavg=lambda: (0, 0, 0),
        )
        collector.sample(timestamp=1.0, monotonic_now=1.0)
        sample = collector.sample(timestamp=1.25, monotonic_now=1.25)

        self.assertIsNone(sample["values"]["cpu_usage"])
        self.assertIsNone(sample["values"]["disk_read_mb_s"])
        self.assertEqual(sample["states"]["cpu"], "reset")
        self.assertEqual(sample["states"]["disk"], "reset")


class HighResHistoryBufferTests(unittest.TestCase):
    @staticmethod
    def _sample(timestamp, value):
        return {
            "timestamp": timestamp,
            "values": {"cpu_usage": value},
            "states": {"cpu": "ok"},
        }

    def test_window_pruning_and_defensive_snapshot(self):
        buffer = high_res.HighResHistoryBuffer(window_seconds=2.0, capacity=10)
        buffer.append(self._sample(100.0, 1), monotonic_now=10.0)
        buffer.append(self._sample(101.0, 2), monotonic_now=11.0)
        newest = buffer.append(self._sample(102.1, 3), monotonic_now=12.1)

        snapshot = buffer.snapshot()
        self.assertEqual([row["sequence"] for row in snapshot], [2, 3])
        self.assertEqual(newest["sequence"], 3)
        snapshot[0]["values"]["cpu_usage"] = 999
        self.assertEqual(buffer.snapshot()[0]["values"]["cpu_usage"], 2.0)

    def test_capacity_is_hard_bounded(self):
        buffer = high_res.HighResHistoryBuffer(window_seconds=60.0, capacity=3)
        for index in range(5):
            buffer.append(self._sample(100 + index, index), monotonic_now=float(index))
        self.assertEqual([row["sequence"] for row in buffer.snapshot()], [3, 4, 5])

    def test_default_ring_retains_ten_minutes_and_snapshots_each_live_window(self):
        buffer = high_res.HighResHistoryBuffer()
        for index in range(2401):
            timestamp = index * high_res.SAMPLE_INTERVAL_SECONDS
            buffer.append(
                self._sample(timestamp, index),
                monotonic_now=timestamp,
            )

        full_snapshot = buffer.snapshot()
        one_minute = buffer.snapshot(window_seconds=60)
        ten_minutes = buffer.snapshot(window_seconds=600)

        self.assertEqual(buffer.window_seconds, 600)
        self.assertEqual(buffer.capacity, 2416)
        self.assertEqual(len(full_snapshot), 2401)
        self.assertEqual(len(one_minute), 240)
        self.assertEqual(len(ten_minutes), 2400)
        self.assertGreater(one_minute[0]["timestamp"], 540)
        self.assertGreater(ten_minutes[0]["timestamp"], 0)

    def test_snapshot_rejects_a_window_larger_than_retention(self):
        buffer = high_res.HighResHistoryBuffer(window_seconds=60)
        with self.assertRaises(ValueError):
            buffer.snapshot(window_seconds=600)

    def test_concurrent_appends_keep_unique_ordered_sequences(self):
        buffer = high_res.HighResHistoryBuffer(window_seconds=60.0, capacity=500)

        def append_batch(offset):
            for index in range(100):
                current = offset + index
                buffer.append(
                    self._sample(float(current), current),
                    monotonic_now=float(current % 100),
                )

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(append_batch, (0, 100, 200, 300)))

        sequences = [row["sequence"] for row in buffer.snapshot()]
        self.assertEqual(sequences, sorted(sequences))
        self.assertEqual(len(sequences), len(set(sequences)))


class _ServiceCollector:
    def __init__(self):
        self.started = 0
        self.closed = 0
        self.samples = 0

    def start(self):
        self.started += 1

    def close(self):
        self.closed += 1

    def sample(self, *, timestamp, monotonic_now):
        self.samples += 1
        return {
            "timestamp": timestamp,
            "values": {"cpu_usage": float(self.samples)},
            "states": {"cpu": "ok", "memory": "ok", "disk": "ok", "network": "ok"},
        }


class _CountingSnapshotBuffer(high_res.HighResHistoryBuffer):
    def __init__(self):
        super().__init__()
        self.snapshot_calls = 0

    def snapshot(self, *, window_seconds=None):
        self.snapshot_calls += 1
        return super().snapshot(window_seconds=window_seconds)


class HighResHistoryServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_snapshot_frame_uses_injected_buffer_once(self):
        buffer = _CountingSnapshotBuffer()
        service = high_res.HighResHistoryService(
            collector=_ServiceCollector(),
            buffer=buffer,
        )

        frame = service.snapshot_frame("1m")

        self.assertIs(service.buffer, buffer)
        self.assertEqual(buffer.snapshot_calls, 1)
        self.assertEqual(frame["status"]["buffered_points"], len(frame["samples"]))

    async def test_immediate_stop_closes_before_task_first_tick(self):
        collector = _ServiceCollector()
        service = high_res.HighResHistoryService(collector=collector)

        service.start()
        await service.stop()

        self.assertFalse(service.running)
        self.assertEqual(collector.started, 1)
        self.assertEqual(collector.closed, 1)

    async def test_start_stop_are_idempotent_and_wait_coalesces(self):
        collector = _ServiceCollector()
        service = high_res.HighResHistoryService(
            collector=collector,
            interval_seconds=0.005,
        )

        task = service.start()
        self.assertIs(service.start(), task)
        await asyncio.sleep(0.03)
        snapshot = service.buffer.snapshot()
        self.assertGreaterEqual(len(snapshot), 3)

        latest = await service.wait_for_next(0)
        self.assertEqual(latest["sequence"], snapshot[-1]["sequence"])
        await service.stop()
        await service.stop()

        self.assertFalse(service.running)
        self.assertEqual(collector.started, 1)
        self.assertEqual(collector.closed, 1)
        with self.assertRaises(high_res.HighResHistoryClosed):
            await service.wait_for_next(latest["sequence"])

    async def test_multiple_waiters_receive_the_same_fanout_sample(self):
        service = high_res.HighResHistoryService(
            collector=_ServiceCollector(),
            interval_seconds=0.02,
        )
        service.start()
        first = await service.wait_for_next(0)

        waiter_a = asyncio.create_task(service.wait_for_next(first["sequence"]))
        waiter_b = asyncio.create_task(service.wait_for_next(first["sequence"]))
        sample_a, sample_b = await asyncio.wait_for(
            asyncio.gather(waiter_a, waiter_b),
            timeout=0.2,
        )
        await service.stop()

        self.assertEqual(sample_a["sequence"], sample_b["sequence"])

    async def test_snapshot_and_point_frames_have_versioned_range_contract(self):
        collector = _ServiceCollector()
        service = high_res.HighResHistoryService(
            collector=collector,
            interval_seconds=0.005,
        )
        service.start()
        await asyncio.sleep(0.01)
        one_minute = service.snapshot_frame("1m")
        ten_minutes = service.snapshot_frame("10m")
        point = service.point_frame(ten_minutes["samples"][-1], "10m")
        await service.stop()

        self.assertEqual(one_minute["type"], "history_1m_snapshot")
        self.assertEqual(one_minute["window_seconds"], 60)
        self.assertEqual(ten_minutes["type"], "history_10m_snapshot")
        self.assertEqual(ten_minutes["window_seconds"], 600)
        self.assertEqual(ten_minutes["status"]["retention_seconds"], 600)
        self.assertEqual(ten_minutes["version"], high_res.PROTOCOL_VERSION)
        self.assertEqual(point["type"], "history_10m_point")
        self.assertIn("sample", point)

        with self.assertRaises(ValueError):
            service.snapshot_frame("30m")
        with self.assertRaises(ValueError):
            service.point_frame({}, "30m")


if __name__ == "__main__":
    unittest.main()
