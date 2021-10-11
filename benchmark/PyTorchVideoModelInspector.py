from abc import ABC
from typing import Any

import copy

import torch
import json
from torchvision import transforms
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
)
from typing import Dict

import torchvision

from benchmark.BaseModelInspector import BaseModelInspector


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors. 
    """
    def __init__(self, alpha=None):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, frames: torch.Tensor):
        if self.alpha is not None:
            fast_pathway = frames
            # Perform temporal sampling from the fast pathway.
            slow_pathway = torch.index_select(
                frames,
                1,
                torch.linspace(
                    0, frames.shape[1] - 1, frames.shape[1] // self.alpha
                ).long(),
            )
            frame_list = [slow_pathway, fast_pathway]
            return frame_list
        else:
            return frames


class PyTorchVideoModelInspector(BaseModelInspector, ABC):
    """
    sub-class of BaseModelInspector for running PyTorchVideo's models inference with metrics testing.
    User can call the the method to run and test the model and return the tested latency and 
    throughput.
    Args:
        repeat_data: data unit to repeat.
        device: the desired device, e.g., "cuda", "cpu", etc.
        model_name: the name of model selected from model zoo
        side_size: int, parameter of input transforms, specific to the model
        mean: list, parameter of input transforms, specific to the model
        std: list, parameter of input transforms, specific to the model
        crop_size: int, parameter of input transforms, specific to the model
        num_frames: int, parameter of input transforms, specific to the model
        sampling_rate: int, parameter of input transforms, specific to the model
        frames_per_second: int, parameter of input transforms, specific to the model
        slowfast_alpha: int, default None, only set it when selecting slowfast model
        batch_num: the number of batches you want to run
        batch_size: batch size you want
        percentile: Default is 30.
    """

    def __init__(
            self,
            repeat_data: str,
            device: str,
            model_name: str,
            side_size: int, 
            mean: list, 
            std: list, 
            crop_size: int, 
            num_frames: int, 
            sampling_rate: int, 
            frames_per_second: int, 
            slowfast_alpha: int = None, 
            batch_num: int = 20,
            batch_size: int = 1,
            percentile: int = 30,
    ):
        self.side_size = side_size
        self.mean =  mean
        self.std = std
        self.crop_size = crop_size
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.frames_per_second = frames_per_second
        self.slowfast_alpha = slowfast_alpha

        BaseModelInspector.__init__(self, repeat_data, device, batch_num, batch_size, percentile)

        self.model_name = model_name

        self.model = self.__load_model()

    def __load_model(self):
        """Load model selected from model zoo"""
        # Remove this may cause a bug
        # This is a bug introduced in pytorch 1.9
        # This bug seems to only appear on colab
        # https://stackoverflow.com/questions/68901236/urllib-error-httperror-http-error-403-rate-limit-exceeded-when-loading-resnet1
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        # remove main may cause bug
        # the default may be master, but recently, the default branch of github migrate to main
        model = torch.hub.load("facebookresearch/pytorchvideo:main", model=self.model_name, pretrained=True)

        # Set to eval mode and move to desired device
        model = model.to(self.device)
        model = model.eval()

        return model

    def data_preprocess(self, raw_data):
        # transform is specific to model
        # I am not sure if this process is applicable to models other than slowfast, x3d and slow
        transform = ApplyTransformToKey(
            key='video', 
            transform=Compose(
                [
                    UniformTemporalSubsample(self.num_frames), 
                    Lambda(lambda x: x / 255.0), 
                    NormalizeVideo(self.mean, self.std), 
                    ShortSideScale(
                        size=self.side_size
                    ), 
                    CenterCropVideo(crop_size=self.crop_size if self.slowfast_alpha is not None else (self.crop_size, self.crop_size)), 
                    PackPathway(alpha=self.slowfast_alpha)
                ]
            )
        )

        clip_duration = (self.num_frames * self.sampling_rate) / self.frames_per_second

        # Select the duration of the clip to load by specifying the start and end duration
        # The start_sec should correspond to where the action occurs in the video
        start_sec = 0
        end_sec = start_sec + clip_duration

        # Initialize an EncodedVideo helper class and load the video
        video = EncodedVideo.from_path(raw_data) 

        # Load the desired clip
        video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)

        # Apply a transform to normalize the video input
        video_data = transform(video_data)

        # Move the inputs to the desired device
        inputs = video_data["video"]
        if self.slowfast_alpha is not None:
            inputs = [i.to(self.device)[None, ...] for i in inputs]
        else:
            inputs = inputs.to(self.device)

        return inputs

    def make_request(self, input_batch) -> Any:
        return input_batch

    def infer(self, request):
        pred_classes = []
        # request contains a batch of video
        # but the method seems to require a single video
        for data in request:
            # Pass the input clip through the model
            # Warning: this method will change "request"
            preds = self.model(copy.deepcopy(data) if self.slowfast_alpha is not None else copy.deepcopy(data)[None, ...])

            # Get the predicted classes
            post_act = torch.nn.Softmax(dim=1)
            preds = post_act(preds)
            pred_class = preds.topk(k=5).indices[0]
            pred_classes.append(pred_class)

        return pred_classes
