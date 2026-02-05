# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
import math
from collections import deque
from typing import Any

import ray
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=2, max_concurrency=20)
class MessageQueue:
    """
    Simplified Ray-based asynchronous message queue for communication between Rollouter and Trainer
    """

    def __init__(self, config: DictConfig, max_queue_size: int = 1000):
        self.config = config
        if max_queue_size is None:
            raise ValueError(f"max_queue_size cannot be None, got: {max_queue_size}")
        self.max_queue_size = int(max_queue_size)
        self.queue = deque(maxlen=self.max_queue_size)
        self.current_param_version = 0

        self.val_queue = deque()

        try:
            if hasattr(config, "async_training") and config.async_training is not None:
                self.staleness_threshold = getattr(config.async_training, "staleness_threshold", 3)
            else:
                self.staleness_threshold = 3
        except (AttributeError, RecursionError):
            self.staleness_threshold = 3

        self.max_stale_samples = self._compute_max_stale_samples()

        # Asyncio for message handling
        self.running = True

        # async safe
        self._lock = asyncio.Lock()
        self._consumer_condition = asyncio.Condition(self._lock)

        # statistic message
        self.total_produced = 0
        self.total_consumed = 0
        self.dropped_samples = 0
        self.dropped_stale_samples = 0
        self.dropped_too_old_samples = 0

        print(
            f"[MessageQueue] initialized with max_queue_size={max_queue_size},"
            f"staleness_threshold={self.staleness_threshold}"
        )

    def _compute_max_stale_samples(self) -> int | None:
        try:
            require_batches = int(self.config.async_training.require_batches)
            ppo_mini_batch_size = int(self.config.actor_rollout_ref.actor.ppo_mini_batch_size)
            trigger_parameter_sync_step = int(self.config.async_training.trigger_parameter_sync_step)
        except (AttributeError, ValueError, TypeError):
            return None
        total_samples_per_sync = require_batches * ppo_mini_batch_size * trigger_parameter_sync_step
        if total_samples_per_sync <= 0:
            return 0
        if self.staleness_threshold <= 0:
            return 0
        return int(math.floor(total_samples_per_sync * self.staleness_threshold))

    async def put_sample(self, sample: Any, param_version: int) -> bool:
        """
        Put a batch sample into the queue

        Args:
            sample: Sample data
            param_version: Parameter version number

        Returns:
            bool: Whether the sample was successfully put into the queue
        """
        async with self._lock:
            # If queue is full, remove the oldest sample (rarely happens)
            is_drop = False
            if len(self.queue) >= self.max_queue_size:
                self.queue.popleft()
                self.dropped_samples += 1
                is_drop = True
                logger.warning("Queue full, dropped sample")
            self.queue.append((param_version, sample))
            self.total_produced += 1

            # Notify waiting consumers
            self._consumer_condition.notify_all()

            if self.total_produced % 100 == 0:
                print(f"MessageQueue stats: produced={self.total_produced}, queue_size={len(self.queue)}")
            if is_drop:
                return False
            return True

    async def get_sample(self) -> Any | None:
        """
        Get a single sample from the queue, wait until one is available

        Returns:
            Any: Single sample data or None if queue is closed
        """
        async with self._lock:
            while len(self.queue) == 0 and self.running:
                await self._consumer_condition.wait()

            # If queue is closed and empty, return None
            if not self.running and len(self.queue) == 0:
                return None

            # Get one sample
            entry = self.queue.popleft()
            if isinstance(entry, tuple) and len(entry) == 2:
                _, data = entry
            else:
                data = entry
            self.total_consumed += 1
            return data, len(self.queue)

    async def update_param_version(self, version: int):
        """Update current parameter version"""
        async with self._lock:
            old_version = self.current_param_version
            self.current_param_version = version
            self.max_stale_samples = self._compute_max_stale_samples()
            dropped_too_old = 0
            dropped_excess_stale = 0
            stale_version = version - 1
            if stale_version >= 0 and self.queue:
                normalized_entries = []
                for entry in self.queue:
                    if isinstance(entry, tuple) and len(entry) == 2:
                        param_version, sample = entry
                    else:
                        param_version, sample = old_version, entry
                    normalized_entries.append((param_version, sample))
                stale_count = 0
                for param_version, sample in normalized_entries:
                    if sample is None:
                        continue
                    if param_version == stale_version:
                        stale_count += 1
                if self.max_stale_samples is None:
                    stale_to_drop = 0
                else:
                    stale_to_drop = max(0, stale_count - self.max_stale_samples)
                new_queue = deque(maxlen=self.max_queue_size)
                for param_version, sample in normalized_entries:
                    if sample is None:
                        new_queue.append((param_version, sample))
                        continue
                    if param_version < stale_version:
                        dropped_too_old += 1
                        continue
                    if param_version == stale_version and stale_to_drop > 0:
                        stale_to_drop -= 1
                        dropped_excess_stale += 1
                        continue
                    new_queue.append((param_version, sample))
                self.queue = new_queue
                self.dropped_too_old_samples += dropped_too_old
                self.dropped_stale_samples += dropped_excess_stale
            print(
                f"Parameter version updated from {old_version} to {version}. "
                f"dropped_too_old={dropped_too_old}, "
                f"dropped_excess_stale={dropped_excess_stale}, "
                f"queue_size={len(self.queue)}"
            )

    async def get_queue_size(self) -> int:
        """Get current queue length"""
        async with self._lock:
            return len(self.queue)

    async def get_statistics(self) -> dict[str, Any]:
        """Get queue statistics"""
        async with self._lock:
            return {
                "queue_size": len(self.queue),
                "total_produced": self.total_produced,
                "total_consumed": self.total_consumed,
                "dropped_samples": self.dropped_samples,
                "dropped_stale_samples": self.dropped_stale_samples,
                "dropped_too_old_samples": self.dropped_too_old_samples,
                "current_param_version": self.current_param_version,
                "staleness_threshold": self.staleness_threshold,
                "max_stale_samples": self.max_stale_samples,
                "max_queue_size": self.max_queue_size,
            }

    async def clear_queue(self):
        """Clear the queue"""
        async with self._lock:
            cleared_count = len(self.queue)
            self.queue.clear()
            logger.info(f"Cleared {cleared_count} samples from queue")

    async def shutdown(self):
        """Shutdown the message queue"""
        async with self._lock:
            self.running = False
            # Notify all waiting coroutines so they can exit
            self._consumer_condition.notify_all()
        logger.info("MessageQueue shutdown")

    async def get_memory_usage(self) -> dict:
        """Get memory usage statistics"""
        async with self._lock:
            # Estimate memory usage of samples in queue
            import sys

            total_size = 0
            sample_count = len(self.queue)

            if sample_count > 0:
                # Estimate size of a single sample (simplified estimation)
                sample_entry = list(self.queue)[0]
                if isinstance(sample_entry, tuple) and len(sample_entry) == 2:
                    sample = sample_entry[1]
                else:
                    sample = sample_entry
                try:
                    if isinstance(sample, (bytes, bytearray)):
                        sample_size = len(sample)
                    else:
                        sample_size = sys.getsizeof(sample)
                    # Since we now store RolloutSample directly, estimate based on its components
                    if hasattr(sample, "original_batch_dict") and sample.original_batch_dict:
                        # Estimate batch data size
                        batch_data = sample.original_batch_dict.get("batch", {})
                        sample_size += len(batch_data) * 1000  # Roughly estimate 1KB per batch entry
                    if hasattr(sample, "agent_loop_output"):
                        # Estimate AgentLoopOutput size
                        sample_size += 5000  # Roughly estimate 5KB for AgentLoopOutput
                    total_size = sample_size * sample_count
                except Exception:
                    total_size = sample_count * 15000  # Roughly estimate 15KB per RolloutSample

            return {
                "queue_samples": sample_count,
                "estimated_memory_bytes": total_size,
                "estimated_memory_mb": total_size / (1024 * 1024),
            }

    async def put_validate(self, data):
        async with self._lock:
            self.val_queue.append(data)

    async def get_validate(self):
        async with self._lock:
            if self.val_queue:
                return self.val_queue.popleft()
            else:
                return None


class MessageQueueClient:
    """Asyncio-compatible MessageQueue client for communicating with MessageQueue Actor"""

    def __init__(self, queue_actor: Any):
        self.queue_actor = queue_actor

    async def put_sample(self, sample: Any, param_version: int) -> bool:
        """Put batch into queue (async)"""
        future = self.queue_actor.put_sample.remote(sample, param_version)
        return await asyncio.wrap_future(future.future())

    async def put_validate(self, data: Any) -> bool:
        future = self.queue_actor.put_validate.remote(data)
        return await asyncio.wrap_future(future.future())

    def get_validate_sync(self) -> Any | None:
        return ray.get(self.queue_actor.get_validate.remote())

    async def get_sample(self) -> Any | None:
        """Get single sample from queue, wait until one is available (async)"""
        future = self.queue_actor.get_sample.remote()
        return await asyncio.wrap_future(future.future())

    async def get_queue_size(self) -> int:
        """Get queue size (async)"""
        future = self.queue_actor.get_queue_size.remote()
        return await asyncio.wrap_future(future.future())

    async def get_statistics(self) -> dict[str, Any]:
        """Get statistics (async)"""
        future = self.queue_actor.get_statistics.remote()
        return await asyncio.wrap_future(future.future())

    async def clear_queue(self):
        """Clear queue (async)"""
        future = self.queue_actor.clear_queue.remote()
        await asyncio.wrap_future(future.future())

    async def shutdown(self):
        """Shutdown queue (async)"""
        future = self.queue_actor.shutdown.remote()
        await asyncio.wrap_future(future.future())

    async def get_memory_usage(self) -> dict:
        """Get memory usage statistics (async)"""
        future = self.queue_actor.get_memory_usage.remote()
        return await asyncio.wrap_future(future.future())

    # Synchronous version of the method (deprecated)
    def put_sample_sync(self, sample: Any, param_version: int) -> bool:
        """Put batch into queue (sync - deprecated, use put_sample instead)"""
        return ray.get(self.queue_actor.put_sample.remote(sample, param_version))

    def get_sample_sync(self) -> Any | None:
        """Get single sample from queue (sync - deprecated, use get_sample instead)"""
        return ray.get(self.queue_actor.get_sample.remote())

    def get_statistics_sync(self) -> dict[str, Any]:
        """Get statistics (sync - deprecated, use get_statistics instead)"""
        return ray.get(self.queue_actor.get_statistics.remote())

    def update_param_version_sync(self, version: int):
        """Update parameter version (async)"""
        return ray.get(self.queue_actor.update_param_version.remote(version))
