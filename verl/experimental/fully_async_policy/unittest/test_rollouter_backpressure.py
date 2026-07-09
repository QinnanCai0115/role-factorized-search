import unittest

from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter


_FullyAsyncRollouterClass = FullyAsyncRollouter.__ray_metadata__.modified_class


class _FakeMessageQueueClient:
    def __init__(self, queue_size: int):
        self.queue_size = queue_size

    def get_statistics_sync(self):
        return {"queue_size": self.queue_size}


def _make_rollouter(queue_size: int, staleness_samples: int, active_tasks=None):
    rollouter = object.__new__(_FullyAsyncRollouterClass)
    rollouter.message_queue_client = _FakeMessageQueueClient(queue_size)
    rollouter.active_tasks = set(active_tasks or [])
    rollouter.staleness_samples = staleness_samples
    rollouter.max_queue_size = 64
    rollouter.max_required_samples = 64
    rollouter.paused = True
    rollouter._estimated_train_points_per_rollout_sample = lambda: 8
    return rollouter


class RollouterBackpressureTest(unittest.IsolatedAsyncioTestCase):
    async def test_should_pause_refreshes_staleness_after_message_queue_drain(self):
        rollouter = _make_rollouter(queue_size=0, staleness_samples=64)

        should_pause = await rollouter._should_pause_generation()

        self.assertFalse(should_pause)
        self.assertEqual(rollouter.staleness_samples, 0)

    async def test_should_pause_still_honors_full_message_queue(self):
        rollouter = _make_rollouter(queue_size=64, staleness_samples=0)

        should_pause = await rollouter._should_pause_generation()

        self.assertTrue(should_pause)
        self.assertEqual(rollouter.staleness_samples, 64)


if __name__ == "__main__":
    unittest.main()
