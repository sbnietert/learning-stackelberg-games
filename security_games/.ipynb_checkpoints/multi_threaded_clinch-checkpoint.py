import numpy as np
import math
from security_games.utils import RepeatedSSG, RepeatedSSGAlg
from security_games.clinch import BatchedClinchThenCommit
from typing import Self, Optional


class MultiThreadedClinch(RepeatedSSGAlg):
    num_threads: int
    threads: list[BatchedClinchThenCommit]
    best_search_result: Optional[np.ndarray] = None
    highest_completed_thread: int = -1
    completed_thread_queue: set[int] = set()
    exploit_leader_utility: float = 0

    def __init__(self,
                 game: RepeatedSSG,
                 minimum_width:float,
                 time_horizon: int, 
                 is_simplex: bool,
                 get_response: RepeatedSSGAlg.FollowerResponseOracle,
                 accuracy_exponent: float = 1.0,
                 verbose: bool = False):
        self.num_threads = math.floor(math.log2(time_horizon)) + 1
        #search_accuracy = game.ssg.n_targets / time_horizon
        self.threads = []
        thread_get_response = self.gen_response_oracle_for_threads()
        # when any thread makes a query, it will update clock and utilities for this object

        for _ in range(self.num_threads):
            thread = BatchedClinchThenCommit(game,
                                            minimum_width,
                                            time_horizon,
                                            is_simplex,
                                            thread_get_response,
                                            batch_size=1,
                                            accuracy_exponent=accuracy_exponent,
                                            verbose=verbose)
            self.threads.append(thread)

        super().__init__(game, time_horizon, get_response)

    def gen_response_oracle_for_threads(self):
        that = self
        def thread_get_response(x: np.ndarray, _: RepeatedSSGAlg):
            nonlocal that # not sure if this is necessary or if I could just use self below
            return that.query_follower(x)
        return thread_get_response

    def copy(self, get_response: RepeatedSSGAlg.FollowerResponseOracle, reset_follower_utility=False) -> Self:
        obj = super().copy(get_response, reset_follower_utility)
        copy_thread_get_response = obj.gen_response_oracle_for_threads()
        obj.threads = []
        for thread in self.threads:
            obj.threads.append(thread.copy(copy_thread_get_response))
        if obj.best_search_result is not None:
            obj.best_search_result = obj.best_search_result.copy()
        obj.completed_thread_queue = obj.completed_thread_queue.copy()
        return obj
    
    def __str__(self) -> str:
        s = ""
        for i,t in enumerate(self.threads):
            s += f'thread:{i}\n' + str(t) + "\n"
        s += super().__str__()
        return s

    def step(self) -> bool:
        t = self.game.time
        #print(f'time:{t}')
        curr_thread_index = max([k for k in range(1, int(math.log2(t+1)+2)) if (t+1) % 2**(k-1) == 0])-1
        #print(f'curr_thread_index:{curr_thread_index}')
        if curr_thread_index < len(self.threads):
            curr_thread = self.threads[curr_thread_index]
            #print(f'curr_thread:{str(curr_thread)}')
            if curr_thread.search_result is None:
                curr_thread.step()
                if curr_thread.search_result is not None:
                    self.completed_thread_queue.add(curr_thread_index) # this does not actually advance clock for some reason, just sets result to be queried next
                    self.query_follower(curr_thread.search_result) # go ahead and query, so that "else" case below is not reached at next step
            else:
                if curr_thread_index in self.completed_thread_queue:
                    self.completed_thread_queue.remove(curr_thread_index)
                    if curr_thread_index > self.highest_completed_thread:
                        self.best_search_result = curr_thread.search_result
                        assert self.best_search_result is not None
                        self.highest_completed_thread = curr_thread_index
                #print('exploiting')
                u0 = self.game.leader_utility
                #print(f'curr leader_util:{u0}')
                #print(f'curr_best_result:{self.best_search_result}')
                if self.best_search_result is None:
                    print('What?')
                assert self.best_search_result is not None
                self.query_follower(self.best_search_result)
                #print(f'new leader_util:{self.game.leader_utility}')
                self.exploit_leader_utility += self.game.leader_utility - u0
        else: 
            assert False # should not be reached with full number of threads, correct this later
            return False
            u0 = self.game.leader_utility
            self.query_follower(self.best_search_result)
            self.exploit_leader_utility += self.game.leader_utility - u0
        return self.game.time < self.time_horizon