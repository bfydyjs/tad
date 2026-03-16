from .builder import DATASETS
from .dataset import PaddingDataset, ResizeDataset, SlidingWindowDataset


def _time_to_frame(segment, video_info):
    return [int(segment[i] / video_info["duration"] * video_info["frame"]) for i in range(2)]


@DATASETS.register_module()
class Ego4DPaddingDataset(PaddingDataset):
    def get_gt(self, video_info, thresh=0.0):
        return self.parse_and_filter_gt(
            video_info, thresh, lambda segment: _time_to_frame(segment, video_info)
        )


@DATASETS.register_module()
class Ego4DResizeDataset(ResizeDataset):
    def get_gt(self, video_info, thresh=0.0):
        def custom_valid(start, end, scale):
            return (not self.filter_gt) or (scale / float(video_info["duration"]) > thresh)

        return self.parse_and_filter_gt(
            video_info,
            thresh,
            lambda segment: [int(s) for s in segment],
            custom_valid_func=custom_valid,
        )


@DATASETS.register_module()
class Ego4DSlidingDataset(SlidingWindowDataset):
    def get_gt(self, video_info, thresh=0.0):
        return self.parse_and_filter_gt(
            video_info, thresh, lambda segment: _time_to_frame(segment, video_info)
        )
