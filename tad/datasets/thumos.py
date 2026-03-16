from .builder import DATASETS
from .dataset import PaddingDataset, SlidingWindowDataset


def _time_to_frame(segment, info):
    return [int(segment[i] / info["duration"] * info["frame"]) for i in range(2)]


def _ignore_ambiguous(label):
    return label == "Ambiguous"


@DATASETS.register_module()
class ThumosSlidingDataset(SlidingWindowDataset):
    def get_gt(self, video_info, thresh=0.0):
        return self.parse_and_filter_gt(
            video_info, thresh, lambda seg: _time_to_frame(seg, video_info), _ignore_ambiguous
        )


@DATASETS.register_module()
class ThumosPaddingDataset(PaddingDataset):
    def get_gt(self, video_info, thresh=0.0):
        return self.parse_and_filter_gt(
            video_info, thresh, lambda seg: _time_to_frame(seg, video_info), _ignore_ambiguous
        )
