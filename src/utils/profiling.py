"""
Performance profiling and optimization utilities for FlyingCarRL.

This module provides tools for profiling GPU usage, training throughput,
and identifying performance bottlenecks.
"""
import time
import functools
import logging
from typing import Callable, Any, Dict
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """
    Profile performance of training and simulation.
    """

    def __init__(self):
        self.metrics = {}
        self.timings = {}

    def time_function(self, func: Callable) -> Callable:
        """
        Decorator to time function execution.

        Args:
            func: Function to time

        Returns:
            Wrapped function with timing
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time

            func_name = func.__name__
            if func_name not in self.timings:
                self.timings[func_name] = []
            self.timings[func_name].append(elapsed)

            logger.debug(f"{func_name} took {elapsed:.4f}s")
            return result

        return wrapper

    @contextmanager
    def profile_block(self, block_name: str):
        """
        Context manager to profile a code block.

        Args:
            block_name: Name of the block being profiled

        Example:
            >>> profiler = PerformanceProfiler()
            >>> with profiler.profile_block("training_step"):
            >>>     model.train_step()
        """
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            if block_name not in self.timings:
                self.timings[block_name] = []
            self.timings[block_name].append(elapsed)
            logger.debug(f"{block_name} took {elapsed:.4f}s")

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of all profiled operations.

        Returns:
            Dictionary with timing statistics
        """
        import numpy as np

        summary = {}
        for name, times in self.timings.items():
            times_array = np.array(times)
            summary[name] = {
                'count': len(times),
                'total': float(np.sum(times_array)),
                'mean': float(np.mean(times_array)),
                'std': float(np.std(times_array)),
                'min': float(np.min(times_array)),
                'max': float(np.max(times_array)),
                'median': float(np.median(times_array))
            }

        return summary

    def print_summary(self):
        """Print a formatted summary of profiling results."""
        summary = self.get_summary()

        print("\n" + "=" * 80)
        print("PERFORMANCE PROFILING SUMMARY")
        print("=" * 80)

        for name, stats in sorted(summary.items(), key=lambda x: x[1]['total'], reverse=True):
            print(f"\n{name}:")
            print(f"  Count: {stats['count']}")
            print(f"  Total: {stats['total']:.4f}s")
            print(f"  Mean:  {stats['mean']:.4f}s ± {stats['std']:.4f}s")
            print(f"  Range: [{stats['min']:.4f}s, {stats['max']:.4f}s]")

        print("=" * 80 + "\n")


class GPUProfiler:
    """
    Profile GPU usage and memory.
    """

    def __init__(self):
        self.gpu_available = False
        self.torch = None

        try:
            import torch
            self.torch = torch
            self.gpu_available = torch.cuda.is_available()
        except ImportError:
            logger.warning("PyTorch not available, GPU profiling disabled")

    def get_gpu_memory_usage(self) -> Dict[str, float]:
        """
        Get current GPU memory usage.

        Returns:
            Dictionary with memory statistics in MB
        """
        if not self.gpu_available:
            return {}

        memory_stats = {}
        for i in range(self.torch.cuda.device_count()):
            allocated = self.torch.cuda.memory_allocated(i) / 1024**2
            reserved = self.torch.cuda.memory_reserved(i) / 1024**2
            max_allocated = self.torch.cuda.max_memory_allocated(i) / 1024**2

            memory_stats[f'gpu_{i}'] = {
                'allocated_mb': allocated,
                'reserved_mb': reserved,
                'max_allocated_mb': max_allocated
            }

        return memory_stats

    def print_gpu_memory(self):
        """Print GPU memory usage."""
        if not self.gpu_available:
            print("GPU not available")
            return

        memory_stats = self.get_gpu_memory_usage()

        print("\n" + "=" * 80)
        print("GPU MEMORY USAGE")
        print("=" * 80)

        for gpu_name, stats in memory_stats.items():
            print(f"\n{gpu_name.upper()}:")
            print(f"  Allocated:     {stats['allocated_mb']:.2f} MB")
            print(f"  Reserved:      {stats['reserved_mb']:.2f} MB")
            print(f"  Max Allocated: {stats['max_allocated_mb']:.2f} MB")

        print("=" * 80 + "\n")

    @contextmanager
    def profile_memory(self, operation_name: str):
        """
        Profile memory usage for an operation.

        Args:
            operation_name: Name of the operation

        Example:
            >>> gpu_profiler = GPUProfiler()
            >>> with gpu_profiler.profile_memory("model_training"):
            >>>     model.train()
        """
        if not self.gpu_available:
            yield
            return

        # Reset peak stats
        self.torch.cuda.reset_peak_memory_stats()

        start_memory = self.torch.cuda.memory_allocated()
        start_time = time.time()

        try:
            yield
        finally:
            end_memory = self.torch.cuda.memory_allocated()
            end_time = time.time()
            peak_memory = self.torch.cuda.max_memory_allocated()

            memory_increase = (end_memory - start_memory) / 1024**2
            peak_increase = (peak_memory - start_memory) / 1024**2
            elapsed = end_time - start_time

            logger.info(
                f"{operation_name}: "
                f"Memory +{memory_increase:.2f}MB "
                f"(peak +{peak_increase:.2f}MB) "
                f"in {elapsed:.2f}s"
            )


class ThroughputMonitor:
    """
    Monitor training and simulation throughput.
    """

    def __init__(self):
        self.start_time = None
        self.total_steps = 0
        self.total_episodes = 0

    def start(self):
        """Start monitoring."""
        self.start_time = time.time()
        self.total_steps = 0
        self.total_episodes = 0

    def update(self, num_steps: int = 0, num_episodes: int = 0):
        """
        Update counters.

        Args:
            num_steps: Number of steps completed
            num_episodes: Number of episodes completed
        """
        self.total_steps += num_steps
        self.total_episodes += num_episodes

    def get_throughput(self) -> Dict[str, float]:
        """
        Get current throughput metrics.

        Returns:
            Dictionary with throughput statistics
        """
        if self.start_time is None:
            return {}

        elapsed = time.time() - self.start_time
        if elapsed == 0:
            return {}

        return {
            'steps_per_second': self.total_steps / elapsed,
            'episodes_per_second': self.total_episodes / elapsed,
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'elapsed_time': elapsed
        }

    def print_throughput(self):
        """Print throughput metrics."""
        metrics = self.get_throughput()

        if not metrics:
            print("Monitoring not started or no data available")
            return

        print("\n" + "=" * 80)
        print("THROUGHPUT METRICS")
        print("=" * 80)
        print(f"Steps per second:    {metrics['steps_per_second']:.2f}")
        print(f"Episodes per second: {metrics['episodes_per_second']:.2f}")
        print(f"Total steps:         {metrics['total_steps']}")
        print(f"Total episodes:      {metrics['total_episodes']}")
        print(f"Elapsed time:        {metrics['elapsed_time']:.2f}s")
        print("=" * 80 + "\n")


def profile_training_run(profiler: PerformanceProfiler, gpu_profiler: GPUProfiler):
    """
    Comprehensive profiling decorator for training runs.

    Args:
        profiler: Performance profiler instance
        gpu_profiler: GPU profiler instance

    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Start profiling
            with gpu_profiler.profile_memory(func.__name__):
                with profiler.profile_block(func.__name__):
                    result = func(*args, **kwargs)

            # Print summaries
            profiler.print_summary()
            gpu_profiler.print_gpu_memory()

            return result

        return wrapper

    return decorator
