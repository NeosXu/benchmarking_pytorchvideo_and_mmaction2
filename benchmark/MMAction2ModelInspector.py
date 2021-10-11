from abc import ABC
from typing import Any

import torch
import numpy as np

from mmaction.apis import init_recognizer, inference_recognizer

from benchmark.BaseModelInspector import BaseModelInspector


class MMAction2ModelInspector(BaseModelInspector, ABC):
    """
    sub-class of BaseModelInspector for running MMAction2's models inference with metrics testing.
    User can call the the method to run and test the model and return the tested latency and 
    throughput.
    Args:
        repeat_data: data unit to repeat.
        device: the desired device, e.g., "cuda", "cpu", etc.
        config_file: config file path for model, you can find it in the folder "mmaction2/config"
        checkpoint_file: checkpoint file path for model, download it from model zoo of mmaction2
        batch_num: the number of batches you want to run
        batch_size: batch size you want
        percentile: Default is 30.
    """

    def __init__(
            self,
            repeat_data,
            device: str,
            config_file: str,
            checkpoint_file: str,
            batch_num: int = 20,
            batch_size: int = 1,
            percentile: int = 30,
    ):
        BaseModelInspector.__init__(self, repeat_data, device, batch_num, batch_size, percentile)

        self.config_file = config_file
        self.checkpoint_file = checkpoint_file

        self.model = self.__load_model()

    def __load_model(self):
        """build the model from a config file and a checkpoint file"""
        return init_recognizer(self.config_file, self.checkpoint_file, device=self.device)

    def data_preprocess(self, raw_data):
        return raw_data

    def make_request(self, input_batch) -> Any:
        return input_batch

    def infer(self, request):
        # request contains a batch of video
        # but the api seems to require a single video
        for data in request:
            results = inference_recognizer(self.model, data)
        return results
