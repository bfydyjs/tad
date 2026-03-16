from .builder import DATASETS
from .dataset import PaddingDataset, SlidingWindowDataset


def _time_to_frame(segment, video_info):
    return [int(segment[i] / video_info["duration"] * video_info["frame"]) for i in range(2)]


def _ignore_ambiguous(label):
    return label == "Ambiguous"


@DATASETS.register_module()
class ThumosSlidingDataset(SlidingWindowDataset):
    def get_gt(self, video_info, thresh=0.0):
        return self.parse_and_filter_gt(
            video_info,
            thresh,
            lambda segment: _time_to_frame(segment, video_info),
            _ignore_ambiguous,
        )


@DATASETS.register_module()
class ThumosPaddingDataset(PaddingDataset):
    def get_gt(self, video_info, thresh=0.0):
        return self.parse_and_filter_gt(
            video_info,
            thresh,
            lambda segment: _time_to_frame(segment, video_info),
            _ignore_ambiguous,
        )
