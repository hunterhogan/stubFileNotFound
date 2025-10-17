from typing import Any
import concurrent.futures

class BaseCuteExecutor(concurrent.futures.Executor):
    """
    An executor with extra functionality for `map` and `filter`.

    This is a subclass of `concurrent.futures.Executor`, which is a manager for
    parallelizing tasks. What this adds over `concurrent.futures.Executor`:

     - A `.filter` method, which operates like the builtin `filter` except it's
       parallelized with the executor.
     - An `as_completed` argument for both `.map` and `.filter`, which makes
       these methods return results according to the order in which they were
       computed, and not the order in which they were submitted.

    """

    def filter(self, filter_function: Any, iterable: Any, timeout: Any=None, as_completed: bool = False) -> Any:
        """
        Get a parallelized version of `filter(filter_function, iterable)`.

        Specify `as_completed=False` to get the results that were calculated
        first to be returned first, instead of using the order of `iterable`.
        """
    def map(self, function: Any, *iterables: Any, timeout: Any=None, as_completed: bool = False) -> Any:
        """
        Get a parallelized version of `map(function, iterable)`.

        Specify `as_completed=False` to get the results that were calculated
        first to be returned first, instead of using the order of `iterable`.
        """

class CuteThreadPoolExecutor(concurrent.futures.ThreadPoolExecutor, BaseCuteExecutor):
    """
    A thread-pool executor with extra functionality for `map` and `filter`.

    This is a subclass of `concurrent.futures.ThreadPoolExecutor`, which is a
    manager for parallelizing tasks to a thread pool. What this adds over
    `concurrent.futures.ThreadPoolExecutor`:

     - A `.filter` method, which operates like the builtin `filter` except it's
       parallelized with the executor.
     - An `as_completed` argument for both `.map` and `.filter`, which makes
       these methods return results according to the order in which they were
       computed, and not the order in which they were submitted.

    """
class CuteProcessPoolExecutor(concurrent.futures.ProcessPoolExecutor, BaseCuteExecutor):
    """
    A process-pool executor with extra functionality for `map` and `filter`.

    This is a subclass of `concurrent.futures.ThreadPoolExecutor`, which is a
    manager for parallelizing tasks to a process pool. What this adds over
    `concurrent.futures.ThreadPoolExecutor`:

     - A `.filter` method, which operates like the builtin `filter` except it's
       parallelized with the executor.
     - An `as_completed` argument for both `.map` and `.filter`, which makes
       these methods return results according to the order in which they were
       computed, and not the order in which they were submitted.

    """



