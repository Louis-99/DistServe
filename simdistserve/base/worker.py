"""
Worker class for simulation. One worker class manages a TP group.
"""
import random
import warnings
from collections import deque
from typing import Optional, List, Iterable, TYPE_CHECKING, Union, TypedDict, Literal
from uuid import UUID

# from simdistserve.estimators.time_estimator import get_prefill_time, get_decode_time
from simdistserve.estimators.time_estimator import get_decode_time_tree, get_prefill_time_tree
from simdistserve.estimators.power_estimator import get_decode_power_tree, get_prefill_power_interp

if TYPE_CHECKING:
    from simdistserve.base.scheduler import Scheduler
    from simdistserve.base.request import Request

from simdistserve.envs import get_skip_prefill, get_skip_decode, get_gpu_freq

# TODO: (Refactor) Make this a configuration.
class WorkerConfig(TypedDict):
    """Behaviors of worker."""
    TP_Prefill: int  # Tensor parallelism for prefill (default = 1)
    TP_Decode: int  # Tensor parallelism for decode (default = 1)
    model_type: str  # Model type for prefill/decode time calculation (default = ModelType.opt_13b)
    prefill_max_batch_size: int  # Maximum number of prefill request in a batch (default = 10**7)
    decode_max_batch_size: int  # Maximum number of decode request in a batch (default = 10**7)
    prefill_max_tokens: int  # Max tokens in prefill iteration (default = 10**7)
    decode_max_tokens: int  # Max tokens in a iteration forward (default = 10**7)
    enable_chunked_prefill: Optional[bool]  # Enable memory pressure simulation (default = False)
    engine_type: Literal["distserve", "vllm"]  # Engine type for prefill/decode time calculation (default = "distserve")

    # TODO: Deprecated
    TP: Optional[int]  # Tensor parallelism (default = 1)
    pass

class Worker:
    def __init__(
        self, env, wid,
        cluster=None,
        is_last_in_pipeline: bool = False,
        pipe_rank: int = None,
        should_request_stay: bool = True,
        prefill_max_batch_size: int = 10 ** 7,
        decode_max_batch_size: int = 10 ** 7,
        global_scheduler: 'Scheduler' = None,
        model_type: str = None,
        TP: int = 1,
        TP_Prefill: int = None,
        TP_Decode: int = None,
        enable_chunked_prefill=False,
        prefill_max_tokens=10 ** 7,
        decode_max_tokens=10 ** 7,
        decode_back_pressure: float = 0.9,
        engine_type: Literal["distserve", "vllm"] = "distserve",
        freq: None|int = None,
    ):
        
        self.env = env
        self.cluster = cluster  # Refer to the cluster of init.
        self.wid = wid
        self.pipe_rank = pipe_rank
        self.is_last_in_pipeline = is_last_in_pipeline
        self.next_worker: 'Optional[Worker]' = None
        self.model_type = model_type

        if freq is None:
            freq = get_gpu_freq()
        self.freq = freq

        # TODO: (Deprecate) TP should be deprecate in favor of TP_prefill and TP_decode.
        self.TP = TP
        self.TP_Prefill = TP_Prefill
        self.TP_Decode = TP_Decode
        if (self.TP_Prefill is None) and (self.TP_Decode is None):
            warnings.warn(f"TP_Prefill and TP_Decode are not set. Default to {TP = } only apply to prefill.")
            self.TP_Prefill = TP
            self.TP_Decode = 1
        elif (self.TP_Prefill is not None) and (self.TP_Decode is not None):
            # Using the new TP_prefill and TP_decode value, instead of TP.
            pass
        elif (self.TP_Prefill is None) or (self.TP_Decode is None):
            warnings.warn(f"{TP = } will be deprecated soon. Use TP_Prefill and TP_Decode.")
            self.TP_Prefill = TP
            self.TP_Decode = 1
            pass

        # Same request should stay in the same worker.
        # If set to false, then it will forward to the global scheduler.
        self.global_scheduler = global_scheduler
        self.should_request_stay: bool = should_request_stay
        # Maximum number requests to fill in prefill batch. (Default 0 => 10 ** 7, big enough number)
        self.prefill_max_batch_size: int = prefill_max_batch_size if prefill_max_batch_size > 0 else 10 ** 7
        self.decode_max_batch_size: int = decode_max_batch_size if decode_max_batch_size > 0 else 10 ** 7
        # Maximum number of tokens for a prefill request to batch.
        self.prefill_max_tokens: int = prefill_max_tokens if prefill_max_tokens > 0 else 10 ** 7
        self.decode_max_tokens: int = decode_max_tokens if decode_max_tokens > 0 else 10 ** 7
        # Enable chunked prefill (if True) or prioritization scheduling (if False)
        self.enable_chunked_prefill: bool = enable_chunked_prefill
        # Decode worker stop accepting incoming request when this is full.
        self.decode_back_pressure = decode_back_pressure

        self.prefill_queue: 'deque[Request]' = deque()
        self.decode_queue: 'deque[Request]' = deque()
        self.transfer_queue: deque[Request] = deque()
        self._prefill_ips: int = 0  # Elements in progress for prefill
        self._decode_ips: int = 0  # Elements in progress for decode
        self._wakeup_event = env.event()
        self.log: list[tuple[float, str, int, int, int, list[int], list[int], float]] = []

        # Simulate scheduler delay in terms of number of decode rounds.
        self._prefill_sched_delay: int = 0
        self.engine_type = engine_type

        self.last_prefill_end_time_env = 0.0

        pass

    def cal_num_block_tokens(self, num_tokens):
        BLOCK_SIZE=64 
        return (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE

    @property
    def is_first_in_pipeline(self):
        return self.pipe_rank == 0

    # @property
    # def has_back_pressure(self) -> bool:
    #     total_cacpacity = self.decode_max_tokens
    #     threshold = total_cacpacity - self.min_free_tokens_for_KV_transfer
    #     current_tokens = self.current_decode_tokens
        
    #     # threshold = int(self.decode_max_batch_size * self.decode_back_pressure)
    #     return sum(r.current_context_len for r in self.decode_queue) > threshold

    def __repr__(self):
        return f"Worker {self.wid}"

    def _log_event(self, event, num_tokens: int = 0, prefill_bs=0, decode_bs=0,
                   prefill_len_list=None, decode_len_list=None, power=0):
        if prefill_len_list is None:
            prefill_len_list = []
        if decode_len_list is None:
            decode_len_list = []
        item = (self.env.now, event, num_tokens, prefill_bs, decode_bs, prefill_len_list, decode_len_list, power)
        self.log.append(item)
        # print(item)
        return

    def run(self):
        while True:
            if not (self.prefill_queue or self.decode_queue):
                yield self._wakeup_event
            
            if self.prefill_queue:
                yield from self.do_prefill()
            else:
                yield from self.do_decode()

            self._log_event("wait")
            pass

        pass

    def add_ray_overhead(self, sum_of_tokens) -> int:
        # base_overhead = 2
        # k = 0.0001
        # delay = base_overhead + sum_of_tokens * k
        # return delay
        return 0

    # run = run_with_schedule_delay

    def wakeup(self):
        self._wakeup_event.succeed()
        self._wakeup_event = self.env.event()
        return

    def forward_prefill(self, items):
        # if items is not iterable, then make it iterable
        if not items:
            return
        if not isinstance(items, Iterable):
            items = [items]

        self.next_worker.prefill_queue.extend(items)
        self.next_worker.wakeup()
        return

    def forward_decode(self, items: Union['Request', Iterable['Request']], to_scheduler: bool = False):
        if not items:
            return
        if not isinstance(items, Iterable):
            items = [items]

        if not to_scheduler:
            self.next_worker.decode_queue.extendleft(reversed(items))
            self.next_worker.wakeup()
            return

        for item in items:
            self.global_scheduler.schedule_decode(item)
        return

    def _enter_decodes(self, remaining_tok_in_batch: int) -> 'tuple[List[Request], int]':
        # decode_max_tokens

        # Acceptable decode requests is capped by the remaining allowed tokens in this batch.
        # TODO: Hack: Must revert this to use the max token given
        # watermark = 0.9
        watermark = 1.0 # fixed by yunzhao
        decode_max_tokens = self.decode_max_tokens * watermark # fixed by yunzhao

        decode_max_batch_size = self.decode_max_batch_size

        # for req in self.transfer_queue.copy():
        #     if not req.check_KV_finished(self.env.now):
        #         decode_max_tokens -= self.cal_num_block_tokens(req.current_context_len + 1)
        #     else:
        #         self.transfer_queue.remove(req)

        # decode_max_tokens = 50000 # fixed by yunzhao
        _decode_len = min(remaining_tok_in_batch, len(self.decode_queue))
        decode_reqs: list[Request] = []
        decode_queue_index = 0
        for i in range(_decode_len):
            if decode_queue_index >= len(self.decode_queue):
                break
            req = self.decode_queue[decode_queue_index]
            # check state of req
            # if state is prefilled, then start KV transfer, change status to inflight, 
            # but dont schedule
            # Yunzhao: Currently the transfer is buggy, skip for now
            # if req.state == 'prefilled':
            #     # if enough space, then start KV transfer
            #     if (req.current_context_len) > decode_max_tokens:
            #         decode_queue_index += 1
            #         continue
            #     else:
            #         req.start_KV_transfer(self.env.now)
            #         # add tokens, but dont schedule
            #         self.transfer_queue.append(req)
            #         decode_max_tokens -= self.cal_num_block_tokens(req.current_context_len + 1)
            #         decode_queue_index += 1
            #         continue
            # if not req.check_KV_finished(self.env.now):
            #     decode_queue_index += 1
            #     continue

            # at this point only normal decode requests are left
            if self.cal_num_block_tokens(req.current_context_len + 1) > decode_max_tokens:
                break
            if decode_max_batch_size <= 0:
                break

            decode_max_tokens -= self.cal_num_block_tokens(req.current_context_len + 1)
            decode_max_batch_size -= 1
            decode_reqs.append(req)
            self.decode_queue.remove(req)
            if decode_queue_index >= len(self.decode_queue):
                break

        num_recompute = 0
        for r in decode_reqs:
            if r.state == 'evicted':
                # print('RECOMPUTE!!!!!!')
                num_recompute += 1
            r.do_decode(wid=self.wid)
            r.state = 'decode'

        # TODO: (Yunzhao) if a preemption happen to a requests, it will affect TPOT of other requests
        # This basic means this configuration with this RPS is not usable. 
        # We should add code to detect this and avoid these RPS
        #  
        # if decode_max_tokens < 0:
        for req in self.decode_queue:
            # if req.state == 'decode':
            req.state = 'evicted'
                # print("PREEMPTED!!!!!!!!!!")
        return decode_reqs, num_recompute

    def _enter_prefill(self) -> 'List[Request]':
        result: 'List[Request]' = []

        
        # Limit the maximum prefill requests to handle.
        max_request_size = min(self.prefill_max_batch_size, len(self.prefill_queue))

        # TODO: (Refactor) This logic becomes spaghetti.
        # If worker is not the first in pipeline, then it will just identify the chunks of prefill.
        if not self.is_first_in_pipeline:
            # Then just fetch all decode with the same chunk-id.
            chunk_id = self.prefill_queue[0].chunk_id
            for i in range(max_request_size):
                candidate: 'Request' = self.prefill_queue[0]
                if candidate.chunk_id != chunk_id:
                    break
                result.append(self.prefill_queue.popleft())
            pass

        else:
            # Worker is the first in pipeline, then it will do chunked prefill.
            chunk_size = 0
            prefill_max_tokens = self.prefill_max_tokens
            # chunk_id assign as uuid
            chunk_id = UUID(int=random.getrandbits(128))
            for _ in range(max_request_size):
                candidate: 'Request' = self.prefill_queue[0]

                if self.enable_chunked_prefill:
                    # The prefill portion that we picked from the candidate.
                    sched_size = min(
                        # The to-schedule size is the minimum of
                        # (1) the remaining prefill size of the candidate, and
                        # (2) the maximum allowed size of a chunked-prefill batch.
                        # This way we greedily cut and schedule the prefill chunk.
                        candidate.remain_prefill_lens,
                        prefill_max_tokens - chunk_size  # max batch size in a chunked-prefill batch - chunk size
                    )
                    if sched_size <= 0:
                        break
                else:
                    # If the whole request can fit into the chunk,
                    # then just schedule the whole request.
                    sched_size = candidate.remain_prefill_lens
                    if sched_size > prefill_max_tokens:
                        break
                    pass

                # Candidate is picked. Now fill in the chunked-prefill information.
                candidate.current_prefill_lens = sched_size
                candidate.remain_prefill_lens -= sched_size
                prefill_max_tokens -= sched_size
                candidate.chunk_id = chunk_id
                chunk_size += sched_size
                assert candidate.remain_prefill_lens >= 0

                result.append(self.prefill_queue.popleft())
                pass
        for i in result:
            i.do_prefill(wid=self.wid)
        return result

    def _exit_prefill(self, prefill_items: List['Request']):
        for item in prefill_items:
            next_wid = self.next_worker.wid if self.next_worker else None
            item.finish_prefill(is_finished_one_round=self.is_last_in_pipeline, wid=self.wid, next_wid=next_wid)
            item.state = "prefilled"  #omar
            if not self.is_last_in_pipeline or (item.remain_prefill_lens > 0):
                # Finish one chunk of prefill. Now forward to the next worker
                # (or head of worker) to do the rest of the parts.
                self.forward_prefill(item)
                continue

            # Arrive at worker who is at the last of pipeline.
            if item.should_finish():
                # ... just a sanity check to avoid any infinite loop.
                continue

            self.last_prefill_end_time_env = self.env.now

            self.forward_decode(item, to_scheduler=(not self.should_request_stay))
        return

    def _exit_decode(self, decode_reqs):
        if not decode_reqs:
            return
        next_wid = self.next_worker.wid if self.next_worker else None
        for r in decode_reqs:
            r.finish_decode(is_finished_one_round=self.is_last_in_pipeline, next_wid=next_wid)
        next_decode_batch = tuple(r for r in decode_reqs if not r.should_finish())
        self.forward_decode(next_decode_batch)
        return

    def do_prefill(self):
        prefill_items: 'List[Request]' = self._enter_prefill()
        if self.enable_chunked_prefill:
            remaining_tok_in_batch = self.prefill_max_tokens - sum(x.current_prefill_lens for x in prefill_items)
            decode_reqs, _ = self._enter_decodes(remaining_tok_in_batch)
        else:
            decode_reqs = []
        # TODO: (Refactor) The `num_tokens` may be used inaccurately in the get prefill time function.
        num_tokens = sum(x.current_prefill_lens for x in prefill_items)
        num_tokens += len(decode_reqs)

        if not get_skip_prefill():
            power = get_prefill_power_interp(
                num_tokens,
                bs=len(prefill_items),
                decode_bs=len(decode_reqs),
                pp=self.cluster.PP_prefill,
                model_type=self.model_type, TP=self.TP_Prefill,
                prefill_len_list=[x.current_prefill_lens for x in prefill_items],
                engine_type=self.engine_type,
                time_since_last_batch=self.env.now - self.last_prefill_end_time_env,
                freq=self.freq,
            )
        else:
            power = 0

        self._log_event(
            "do_prefill",
            num_tokens=num_tokens,
            prefill_bs=len(prefill_items),
            decode_bs=len(decode_reqs),
            prefill_len_list=[x.current_prefill_lens for x in prefill_items],
            decode_len_list=[x.current_context_len for x in decode_reqs],
            power=power,
        )

        if not get_skip_prefill():
            # Get prefill time wrt total number of tokens.
            delay = get_prefill_time_tree(
                num_tokens,
                bs=len(prefill_items),
                decode_bs=len(decode_reqs),
                pp=self.cluster.PP_prefill,
                model_type=self.model_type, TP=self.TP_Prefill,
                # prefill_len_list=[x.current_prefill_lens for x in prefill_items],
                prefill_len_list=[0.95 * x.current_prefill_lens + 0.05 * (x.prefill_lens - x.remain_prefill_lens) for x in prefill_items],
                # prefill_len_list=[x.prefill_lens - x.remain_prefill_lens for x in prefill_items],
                # prefill_len_list=[x.prefill_lens for x in prefill_items],
                engine_type=self.engine_type,
                time_since_last_batch=self.env.now - self.last_prefill_end_time_env,
                # __prefill_reqs=prefill_items,
                # __decode_reqs=decode_reqs,
                freq=self.freq,
            )
            # delay += 5 # 5ms for scheduling
            num_tokens = sum(x.current_context_len for x in (prefill_items + decode_reqs))
            if self.is_first_in_pipeline:
                delay += self.add_ray_overhead(num_tokens)
            
        else:
            delay = 0
        
        # TODO: (Yunzhao) do something about power
        # Set the number of prefills in progress such that the scheduler get proper information about the worker.
        self._prefill_ips = len(prefill_items)
        yield self.env.timeout(delay)
        self._prefill_ips = 0
        self._exit_prefill(prefill_items)
        self._exit_decode(decode_reqs)
        return

    def do_decode(self):
        decode_reqs, num_recompute_req = self._enter_decodes(self.decode_max_tokens)
        batch_size = len(decode_reqs)

        _token_generated_list = [x.current_context_len + 1 for x in decode_reqs]
        
        power = get_decode_power_tree(
            batch_size, pp=self.cluster.PP_decode,
            model_type=self.model_type, TP=self.TP_Decode,
            token_generated_list=_token_generated_list,
            engine_type=self.engine_type,
            freq=self.freq,
        )

        self._log_event(
            "do_decode", num_tokens=batch_size, decode_bs=batch_size,
            decode_len_list=[x.current_context_len for x in decode_reqs],
            power=power,
        )
        if not get_skip_decode():
            delay = get_decode_time_tree(
                batch_size, pp=self.cluster.PP_decode,
                model_type=self.model_type, TP=self.TP_Decode,
                token_generated_list=_token_generated_list,
                engine_type=self.engine_type,
                freq=self.freq,
            )
            num_tokens = sum(x.current_context_len for x in decode_reqs)
            if self.is_first_in_pipeline:
                delay += self.add_ray_overhead(num_tokens)
        else:
            delay = 0
        yield self.env.timeout(delay + 10000 * num_recompute_req)
        self._exit_decode(decode_reqs)
        return

    pass
