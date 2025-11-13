# A collection of utility functions for the robot.


import time
import warnings
from contextlib import contextmanager
from typing import Optional, Iterator, Generator
from collections import deque
import statistics


class RateLoop:
    """
    A rate-controlled loop that attempts to maintain a specific frequency.

    Example usage:
        with RateLoop(frequency=20) as loop:
            for i in loop:
                # Your code here
                print(f"Iteration {i}")
                if i >= 100:  # Stop after 100 iterations
                    break
    """

    def __init__(
        self,
        frequency: float = 1.0,
        max_iterations: Optional[int] = None,
        warn_threshold: float = 0.1,
        warn_window: int = 10,
        verbose: bool = False,
    ):
        """
        Initialize the RateLoop.

        Args:
            frequency: Target loop frequency in Hz
            max_iterations: Maximum number of iterations (None for infinite)
            warn_threshold: Fraction of target frequency to trigger warning (0.1 = 10%)
            warn_window: Number of recent iterations to consider for frequency calculation
            verbose: If True, print timing statistics periodically
        """
        self.frequency = frequency
        self.period = 1.0 / frequency
        self.max_iterations = max_iterations
        self.warn_threshold = warn_threshold
        self.warn_window = warn_window
        self.verbose = verbose

        self.iteration = 0
        self.start_time = None
        self.last_iteration_time = None
        self.iteration_times = deque(maxlen=warn_window)
        self.warned_recently = False

    def __enter__(self):
        """Context manager entry."""
        self.start_time = time.perf_counter()
        self.last_iteration_time = self.start_time
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.verbose and self.iteration > 0:
            total_time = time.perf_counter() - self.start_time
            avg_freq = self.iteration / total_time
            print(f"\nRateLoop Summary:")
            print(f"  Total iterations: {self.iteration}")
            print(f"  Total time: {total_time:.3f}s")
            print(
                f"  Average frequency: {avg_freq:.2f} Hz (target: {self.frequency} Hz)"
            )

    def __iter__(self):
        """Make the context manager iterable."""
        return self

    def __next__(self):
        """Get next iteration, maintaining timing."""
        # Check if we've reached max iterations
        if self.max_iterations is not None and self.iteration >= self.max_iterations:
            raise StopIteration

        # For all iterations after the first, handle timing
        if self.iteration > 0:
            current_time = time.perf_counter()
            elapsed = current_time - self.last_iteration_time

            # Sleep if we're running ahead of schedule
            if elapsed < self.period:
                time.sleep(self.period - elapsed)
                current_time = time.perf_counter()
                elapsed = current_time - self.last_iteration_time

            # Track iteration timing
            self.iteration_times.append(elapsed)

            # Check if we should warn about frequency deviation
            self._check_frequency_deviation()

            self.last_iteration_time = current_time
        else:
            self.last_iteration_time = time.perf_counter()

        self.iteration += 1
        return self.iteration - 1

    def _check_frequency_deviation(self):
        """Check if actual frequency deviates from target and warn if needed."""
        if len(self.iteration_times) >= self.warn_window:
            avg_period = statistics.mean(self.iteration_times)
            actual_freq = 1.0 / avg_period if avg_period > 0 else 0

            # Calculate deviation
            deviation = abs(actual_freq - self.frequency) / self.frequency

            if deviation > self.warn_threshold:
                if not self.warned_recently:
                    warnings.warn(
                        f"RateLoop frequency deviation: "
                        f"Target={self.frequency:.2f}Hz, "
                        f"Actual={actual_freq:.2f}Hz "
                        f"({deviation * 100:.1f}% deviation)",
                        RuntimeWarning,
                    )
                    self.warned_recently = True
            else:
                self.warned_recently = False

            # Verbose output
            if self.verbose and self.iteration % self.warn_window == 0:
                print(
                    f"[Iter {self.iteration}] Freq: {actual_freq:.2f}Hz "
                    f"(target: {self.frequency}Hz)"
                )


if __name__ == "__main__":
    import random

    # Test run the rate-controlled loop at 20Hz
    print("Example 1: Basic rate-controlled loop at 10Hz")
    print("-" * 50)
    with RateLoop(frequency=20, max_iterations=None, verbose=True) as loop:
        for i in loop:
            # Simulate some work with variable duration
            work_time = random.uniform(0.03, 0.06)
            time.sleep(work_time)
            print(f".", end="", flush=True)
    print()
