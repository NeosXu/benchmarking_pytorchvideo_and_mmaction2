from abc import ABC
from typing import Any

import torch
import numpy as np

from mmaction.apis import init_recognizer, inference_recognizer

from benchmark.BaseModelInspector import BaseModelInspctor


class MMAction2ModelInspector(BaseModelInspctor, ABC):
    def __init__(
            self,
            repeat_data,
            device: str,
            config_file: str,
            checkpoint_file: str,
            batch_num: int = 20,
            batch_size: int = 1,
            percentile: int = 95,
    ):
        BaseModelInspctor.__init__(self, repeat_data, device, batch_num, batch_size, percentile)

        self.config_file = config_file
        self.checkpoint_file = checkpoint_file

        self.model = self.__load_model()

    def __load_model(self):
        return init_recognizer(self.config_file, self.checkpoint_file, device=self.device)

    def data_preprocess(self, raw_data):
        return raw_data

    def make_request(self, input_batch) -> Any:
        return input_batch[0]

    def infer(self, request):
        results = inference_recognizer(self.model, request) # request只能是path
        return results
