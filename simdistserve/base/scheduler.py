from queue import Queue
from typing import List, TYPE_CHECKING, Tuple, Union
import numpy as np

if TYPE_CHECKING:
    from simdistserve.base.request import Request
    from simdistserve.base.worker import Worker

class WeightedRR:
    def __init__(self, weights: list[float]):
        self.virtual_steps = np.array([1 / w for w in weights])
        self.virtual_times = np.zeros_like(self.virtual_steps)
    def next(self) -> int:
        min_idx = np.argmin(self.virtual_times)
        self.virtual_times -= self.virtual_times[min_idx]
        self.virtual_times[min_idx] = self.virtual_steps[min_idx]
        return min_idx

class Scheduler:
    def __init__(self, env, prefill_heads, decode_heads, 
                 prefill_weights: None|list[float] = None,
                 decode_weights: None|list[float] = None):
        self.env = env
        self._prefill_heads: 'List[Worker]' = prefill_heads
        self._prefill_queues = [i.prefill_queue for i in self._prefill_heads]
        self._decode_heads: 'List[Worker]' = decode_heads
        self._decode_queues = [i.decode_queue for i in self._decode_heads]
        
        self._prefill_wrr = None
        self._decode_wrr = None
        if prefill_weights is not None:
            assert len(prefill_weights) == len(prefill_heads)
            self._prefill_wrr = WeightedRR(prefill_weights)
        if decode_weights is not None:
            assert len(decode_weights) == len(decode_heads)
            self._decode_wrr = WeightedRR(decode_weights)
        pass

    @staticmethod
    def _find_best_worker_and_queue(workers, queues) -> 'Tuple[Worker, Union[Queue, List]]':
        # Peak the queue to find the least loaded worker.
        # Assume round-robin
        # Add the pending tasks in prefill
        worker, queue = min(zip(workers, queues), key=lambda x: x[0]._prefill_ips + len(x[1]))
        return worker, queue

    @staticmethod
    def _sched_request(req, worker, queue):
        queue.append(req)
        worker.wakeup()
        return

    def schedule_new_req(self, req: 'Request'):
        if req.counter < 0:
            return self.schedule_prefill(req)
        # This is for the 'decode-only' case.
        return self.schedule_decode(req)

    def schedule_prefill(self, req: 'Request'):
        assert req.counter < 0
        if self._prefill_wrr is not None:
            idx = self._prefill_wrr.next()
            worker = self._prefill_heads[idx]
            queue = self._prefill_queues[idx]
        else:
            worker, queue = self._find_best_worker_and_queue(self._prefill_heads, queues=self._prefill_queues)
        self._sched_request(req, worker, queue)
        return

    def schedule_decode(self, req: 'Request'):
        assert req.counter >= 0
        if req.should_finish():
            # Force request to quit.
            req.finish_decode()
            return

        if self._decode_wrr is not None:
            idx = self._decode_wrr.next()
            worker = self._decode_heads[idx]
            queue = self._decode_queues[idx]
        else:
            worker, queue = self._find_best_worker_and_queue(self._decode_heads, queues=self._decode_queues)
        req.wait_decode(worker.wid) # Artifact to prevent request having FTL != 0 when decode only.
        self._sched_request(req, worker, queue)
        return

    pass


def put_request(env, scheduler: 'Scheduler', delays, requests):
    for r, delay in zip(requests, delays):
        r.init()
        scheduler.schedule_new_req(r)
        yield env.timeout(delay)
    return


def put_request_at_time(env, scheduler: 'Scheduler', time, request: 'Request'):
    yield env.timeout(time)
    request.init()
    scheduler.schedule_new_req(request)
    return


def put_requests_with_interarrivals(env, scheduler: 'Scheduler', inter_arrivals, requests):
    """Put requests with the inter-arrivals."""
    assert len(inter_arrivals) == len(requests), (
        f"Number of requests ({len(requests)}) and inter-arrivals ({len(inter_arrivals)}) "
        f"should be the same."
    )
    wake_time = 0
    for r, ts in zip(requests, inter_arrivals):
        if r.env is None:
            r.env = env
        assert r.env == env
        wake_time += ts
        env.process(put_request_at_time(env, scheduler, wake_time, r))
    return
