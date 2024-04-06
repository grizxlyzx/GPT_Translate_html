import random
import asyncio
import threading
import time
from queue import Queue
from typing import Type


class RateController:
    def __init__(self,
                 limit: int,
                 period_sec: float,
                 backoff_max_retry: int = 10,
                 backoff_init_delay: float = 1,
                 backoff_exp_base: float = 2,
                 backoff_on_errors: tuple[Type[Exception], ...] = ()
                 ):
        """
        Wrapper for rate control in a sliding window fashion with asynchronous support.
        This class enables rate limiting by blocking calls that exceed the defined
        limit within a specified period.
        EXAMPLE USAGE:
            rate_control = RateController(limit=10, period_sec=30)  # maximum 10 calls within 30 seconds

            @rate_control.apply(asynchronous=False)
            def foo(n):
                time.sleep(1)
                return n * 10

            @rate_control.apply(asynchronous=True)
            async def bar(n):
                await asyncio.sleep(1)
                return n * 10

        :param limit: The maximum number of calls allowed within the specified period.
        :param period_sec: The length of the period in seconds.
        :param backoff_max_retry: The maximum number of retries before raising an exception.
        :param backoff_init_delay: The initial delay before the first retry, in seconds.
        :param backoff_exp_base: The base of exponential backoff.
        :param backoff_on_errors: A tuple of exceptions upon which retries should be attempted.
        """
        self.limit = limit
        self.period = period_sec
        self._sw = Queue(maxsize=limit)
        self._t_lock = threading.Lock()
        self._a_lock = asyncio.Lock()

        # retry with backoff
        self.backoff_init_delay = backoff_init_delay
        self.backoff_delay = backoff_init_delay
        self.backoff_max_retry = backoff_max_retry
        self.backoff_exp_base = backoff_exp_base
        self.backoff_on_errors = backoff_on_errors

    def _block_by_rate(self):
        now = time.time()
        if self._sw.full():
            last = self._sw.get()
            if (lag := (now - last)) < self.period:
                time.sleep(self.period - lag)
        self._sw.put(time.time())
        return

    async def _ablock_by_rate(self):
        now = time.time()
        if self._sw.full():
            last = self._sw.get()
            if (lag := (now - last)) < self.period:
                await asyncio.sleep(self.period - lag)
        self._sw.put(time.time())
        return

    def _run(self, func, *fargs, **fkwargs):
        num_retry = 0
        backoff_delay = self.backoff_init_delay
        while True:
            try:
                with self._t_lock:
                    self._block_by_rate()
                return func(*fargs, **fkwargs)

            except Exception as e:
                num_retry += 1
                if num_retry > self.backoff_max_retry:
                    raise Exception(f'Maximum retry reached: {e}')
                if e in self.backoff_on_errors:
                    time.sleep(backoff_delay)
                    backoff_delay *= (self.backoff_exp_base + random.uniform(-1, 1))

    async def _arun(self, func, *fargs, **fkwargs):
        num_retry = 0
        while True:
            try:
                async with self._a_lock:
                    await self._ablock_by_rate()
                ret = await func(*fargs, **fkwargs)
                self.backoff_delay = self.backoff_init_delay
                return ret
            except Exception as e:
                num_retry += 1
                if num_retry > self.backoff_max_retry:
                    raise Exception(f'Maximum retry reached: {e}')
                if any(isinstance(e, e_type) for e_type in self.backoff_on_errors):
                    delay = self.backoff_delay
                    self.backoff_delay *= (self.backoff_exp_base + random.uniform(-1, 1))
                    self.backoff_delay = min(self.backoff_delay, 300)
                else:
                    delay = self.backoff_init_delay
                await asyncio.sleep(delay)  # block the event loop or thread
                print(f'RateController: Error: {e} | Retry: {num_retry}/{self.backoff_max_retry} '
                      f'| BackoffDelay: {delay}s    [{time.ctime()}]')

    def apply(self, asynchronous: bool = False):
        """
        The function wrapper to apply the rate limit control.
        :param asynchronous: If the wrapped function is asynchronous or not
        :return: wrapped function
        """

        def inner(func):
            if asynchronous:
                async def wrapper(*args, **kwargs):
                    return await self._arun(func, *args, **kwargs)
            else:
                def wrapper(*args, **kwargs):
                    return self._run(func, *args, **kwargs)
            return wrapper

        return inner
