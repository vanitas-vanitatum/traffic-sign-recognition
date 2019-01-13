import time
from typing import *


class Step:
    def __init__(self, name: str):
        self.name = name
        self.required_keys: List[str] = []

    def perform(self, data: Dict) -> Dict:
        raise NotImplementedError

    def check_for_necessary_keys(self, data: Dict) -> List[str]:
        not_found_keys = []
        for key in self.required_keys:
            if key not in data:
                not_found_keys.append(key)
        if len(not_found_keys) > 0:
            raise ValueError("Required keys %s not found in input data" % str(not_found_keys))
        return not_found_keys


class Pipeline:
    def __init__(self, stages: List[Step]):
        self._pipeline: List[Step] = stages
        self._intermediate_results = {}
        self.timers = {}
        self._start_time = 0
        self._end_time = 0

    def perform(self, data: Dict, timeit: bool, steps_names_to_omit: List[str] = ()) -> Dict:
        self._intermediate_results.update(data)
        for step in self._pipeline:
            if step.name in steps_names_to_omit:
                continue
            if timeit:
                start_time = time.time()
            result = step.perform(self._intermediate_results)
            if timeit:
                end_time = time.time()
                self.timers[step.name] = end_time - start_time
            self._intermediate_results.update(result)
        return self._intermediate_results

    def get_intermediate_output(self, key: str):
        if key not in self._intermediate_results:
            raise AttributeError("Not found in aggregated data")
        return self._intermediate_results[key]

    def __add__(self, other: List[Step]):
        self._pipeline += other
        return self
