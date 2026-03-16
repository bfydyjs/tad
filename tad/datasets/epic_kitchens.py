from .builder import DATASETS
from .dataset import PaddingDataset, SlidingWindowDataset


@DATASETS.register_module()
class EpicKitchensPaddingDataset(PaddingDataset):
    def get_gt(self, video_info, thresh=0.0):
        return self.parse_and_filter_gt(
            video_info, thresh, lambda segment: [int(s * self.fps) for s in segment]
        )


@DATASETS.register_module()
class EpicKitchensSlidingDataset(SlidingWindowDataset):
    def get_gt(self, video_info, thresh=0.0):
        return self.parse_and_filter_gt(
            video_info, thresh, lambda segment: [int(s * self.fps) for s in segment]
        )
