from .builder import DATASETS
from .dataset import PaddingDataset, ResizeDataset, SlidingWindowDataset


@DATASETS.register_module()
class AnetResizeDataset(ResizeDataset):
    def get_gt(self, video_info, thresh=0.01):
        def custom_valid(start, end, scale):
            return (not self.filter_gt) or (scale / float(video_info["duration"]) > thresh)

        return self._parse_and_filter_gt(
            video_info, thresh, lambda seg: [float(s) for s in seg], custom_valid_func=custom_valid
        )


@DATASETS.register_module()
class AnetPaddingDataset(PaddingDataset):
    def get_gt(self, video_info, thresh=0.0):
        fps = self.get_fps(video_info)

        def custom_valid(start, end, scale):
            return (not self.filter_gt) or (
                (scale > thresh)
                and (end - self.offset_frames > 0)
                and (start - self.offset_frames <= float(video_info["duration"]) * fps)
            )

        return self._parse_and_filter_gt(
            video_info,
            thresh,
            lambda seg: [float(s * fps) for s in seg],
            custom_valid_func=custom_valid,
        )


@DATASETS.register_module()
class AnetSlidingDataset(SlidingWindowDataset):
    def get_gt(self, video_info, thresh=0.0):
        fps = self.get_fps(video_info)

        def custom_valid(start, end, scale):
            return (not self.filter_gt) or (
                (scale > thresh)
                and (end - self.offset_frames > 0)
                and (start - self.offset_frames <= float(video_info["duration"]) * fps)
            )

        return self._parse_and_filter_gt(
            video_info,
            thresh,
            lambda seg: [float(s * fps) for s in seg],
            custom_valid_func=custom_valid,
        )
