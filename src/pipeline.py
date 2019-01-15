import time
from queue import Queue
from threading import Thread
from typing import *

import cv2
import imutils
import numpy as np

import common


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

    def __add__(self, other: List[Step]) -> "Pipeline":
        new_stages = []
        for stage in self._pipeline:
            new_stages.append(stage)
        new_stages.extend(other)
        pipeline = Pipeline(new_stages)
        return pipeline


class MultiThreadedProcessor:
    def __init__(self, file_path: str, pipeline: Pipeline, queue_size: int = 128):
        self.stream = cv2.VideoCapture(file_path)
        self.stopped = False
        self.read_queue = Queue(maxsize=1)
        self.predicted_queue = Queue(maxsize=1)
        self.pipeline = pipeline

        self.prediction_thread = None

    def start(self):
        t = Thread(target=self.update, args=(), daemon=True)
        t.start()

        prediction_thread = Thread(target=self.predict, args=(), daemon=True)
        prediction_thread.start()
        return self

    def update(self):
        print('lulz')
        while True:
            if self.stopped:
                return
            if self.read_queue.empty():
                (grabbed, frame) = self.stream.read()
                if not grabbed:
                    self.stop()
                    return
                self.read_queue.put(frame)

    def predict(self):
        while True:
            if not self.read_queue.empty():
                print('kekz')
                frame = self.read_queue.get()
                frame = imutils.resize(frame, height=common.FRAME_HEIGHT)
                frame = frame.astype(np.uint8)
                self.pipeline.perform({
                    "input": frame
                }, False)
                visualised = self.pipeline.get_intermediate_output("visualised")
                self.predicted_queue.put(visualised)

    def read(self):
        if not self.predicted_queue.empty():
            return self.predicted_queue.get()
        return None

    def stop(self):
        self.stopped = True
