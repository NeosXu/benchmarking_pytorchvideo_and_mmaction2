import datetime
import time
from abc import abstractmethod, ABCMeta
from typing import Any
import numpy as np


class BaseModelInspctor(metaclass=ABCMeta):
    """
    A class for running the model inference with metrics testing. User can
    call the the method to run and test the model and return the tested
    latency and throughput.
    Args:
        batch_num: the number of batches you want to run
        batch_size: batch size you want
        repeat_data: data unit to repeat.
        percentile: The SLA percentile. Default is 95.
    """

    def __init__(
            self,
            repeat_data,
            device,
            batch_num: int = 20,
            batch_size: int = 1,
            percentile: int = 95,
    ):
        self.throughput_list = []
        self.latencies = []

        self.percentile = percentile

        self.device = device

        self.batch_num = batch_num
        self.batch_size = batch_size

        self.raw_data = repeat_data
        self.processed_data = self.data_preprocess(self.raw_data)

        self.batches = self.__client_batch_request()

    @abstractmethod
    def data_preprocess(self, raw_data):
        """Handle raw data, after preprocessing we can get the processed_data, which is using for benchmarking."""
        return raw_data

    def set_batch_size(self, new_batch_size):
        """
        update the batch size here.

        Args:
            new_batch_size: new batch size you want to use.
        """
        self.batch_size = new_batch_size
        self.batches = self.__client_batch_request()

    def __client_batch_request(self):
        """Batching input data according to the specific batch size."""
        batches = []
        for i in range(self.batch_num):
            batch = []
            for j in range(self.batch_size):
                batch.append(self.processed_data)
            batches.append(batch)
        return batches

    @abstractmethod
    def make_request(self, input_batch) -> Any:
        """Function for sub-class to implement before inferring, to create the `self.request` can be
            overridden if needed.
        """
        return input_batch

    @abstractmethod
    def infer(self, request):
        """Abstract function for sub-class to implement the detailed infer function.

        Args:
            request: The batch data in the request.
        """
        pass

    def start_infer_with_time(self, batch_input):
        request = self.make_request(batch_input)
        start_time = time.time()
        self.infer(request)
        end_time = time.time()
        return end_time - start_time

    def run_model(self):
        """Running the benchmarking for the specific model on the specific server.
        """
        # reset the results
        self.throughput_list = []
        self.latencies = []

        # warm-up
        if self.batch_num > 10:
            warm_up_batches = self.batches[:10]
            for batch in warm_up_batches:
                self.start_infer_with_time(batch)
        else:
            raise ValueError("Not enough test values, try to make more testing data.")

        pass_start_time = time.time()
        for batch in self.batches:
            a_batch_latency = self.start_infer_with_time(batch)
            self.latencies.append(a_batch_latency)
            a_batch_throughput = self.batch_size / a_batch_latency
            self.throughput_list.append(a_batch_throughput)
            print(f' latency: {a_batch_latency:.4f} sec throughput: {a_batch_throughput:.4f} req/sec')

        while len(self.latencies) != len(self.batches):
            pass

        pass_end_time = time.time()
        all_data_latency = pass_end_time - pass_start_time
        all_data_throughput = (self.batch_size * self.batch_num) / (pass_end_time - pass_start_time)
        custom_percentile = np.percentile(self.latencies, self.percentile)

        return self.print_results(all_data_throughput, all_data_latency, custom_percentile)

    def print_results(
            self,
            throughput,
            latency,
            custom_percentile,
    ):
        """Export the testing results to local JSON file.

        Args:
            throughput: The tested overall throughput for all batches.
            latency: The tested latency for all batches.
            custom_percentile: The custom percentile you want to check for latencies.
        """
        percentile_50 = np.percentile(self.latencies, 50)
        percentile_95 = np.percentile(self.latencies, 95)
        percentile_99 = np.percentile(self.latencies, 99)
        complete_time = datetime.datetime.now()

        print('\n')
        print(f'total batches: {len(self.batches)}, batch_size: {self.batch_size}')
        print(f'total latency: {latency} s')
        print(f'total throughput: {throughput} req/sec')
        print(f'50th-percentile latency: {percentile_50} s')
        print(f'95th-percentile latency: {percentile_95} s')
        print(f'99th-percentile latency: {percentile_99} s')
        # print(f'{self.percentile}th-percentile latency: {custom_percentile} s')
        print(f'completed at {complete_time}')

        return {
            'total_batches': len(self.batches),
            'batch_size': self.batch_size,
            'total_latency': latency,
            'total_throughput': throughput,
            'completed_time': complete_time,
        }
