from abc import ABC
from typing import Any

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

from benchmark.BaseModelInspector import BaseModelInspctor


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors. 
    """
    def __init__(self, alpha):
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
        return frames


class PyTorchVideoModelInspector(BaseModelInspctor, ABC):
    def __init__(
            self,
            repeat_data: str,
            model: str,
            device: str,
            model_name: str,
            side_size: int, 
            mean: list, 
            std: list, 
            crop_size: int, 
            num_frames: int, 
            sampling_rate: int, 
            frames_per_second: int, 
            num_clips: int, 
            num_crops: int, 
            slowfast_alpha: int = None, 
            batch_num: int = 20,
            batch_size: int = 1,
            percentile: int = 95,
    ):
        BaseModelInspctor.__init__(self, repeat_data, model, device, batch_num, batch_size, percentile)

        self.side_size = side_size
        self.mean =  mean
        self.std = std
        self.crop_size = crop_size
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.frames_per_second = frames_per_second
        self.slowfast_alpha = slowfast_alpha
        self.num_clips = num_clips
        self.num_crops = num_crops

        self.model_name = model_name

        self.model = self.__load_model()

    def __load_model(self):
        # This is a bug introduced in pytorch 1.9
        # 这个bug似乎只会出现在colab上
        # https://stackoverflow.com/questions/68901236/urllib-error-httperror-http-error-403-rate-limit-exceeded-when-loading-resnet1
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        # 代码默认主分支是master，github最近的默认主分支都从master迁移到了main
        model = torch.hub.load("facebookresearch/pytorchvideo:main", model=self.model_name, pretrained=True)

        # Set to eval mode and move to desired device
        model = model.to(self.device)
        model = model.eval()

        return model

    def data_preprocess(self, raw_data):
        # 暂时只针对slowfast, x3d, slow
        transforms = ApplyTransformToKey(
            key='video', 
            transforms=Compose(
                [
                    UniformTemporalSubsample(self.num_frames), 
                    Lambda(lambda x: x / 255.0), 
                    NormalizeVideo(self.mean, self.std), 
                    ShortSideScale(
                        size=self.side_size
                    ), 
                    CenterCropVideo(crop_size=self.crop_size if self.slowfast_alpha is not None else (self.crop_size, self.crop_size)), 
                    PackPathway(self.slowfast_alpha)
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
        video_data = transforms(video_data)

        # Move the inputs to the desired device
        inputs = video_data["video"]
        inputs = inputs.to(self.device)

        return inputs

    def make_request(self, input_batch) -> Any:
        return super().make_request(input_batch)

    def infer(self, request):
        # Pass the input clip through the model
        preds = self.model(request[None, ...])

        # Get the predicted classes
        post_act = torch.nn.Softmax(dim=1)
        preds = post_act(preds)
        pred_classes = preds.topk(k=5).indices[0]

        return pred_classes
