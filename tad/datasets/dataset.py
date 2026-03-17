import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from .builder import DATASETS, Pipeline
from .util import generate_class_map


class BaseDataset:
    """
    Data building pipeline:
    Dataset.__init__()
       └── self.build_data_list()
              └── for video in database:
                     └── video_anno = self.get_gt(video_info)
                            └── (Implemented by subclass) self.parse_and_filter_gt(...)
                                  └── Outputs a dictionary with gt_segments_s / gt_segments_frame
                     └── self.add_to_data_list(video_name, ..., video_anno)
    """

    def __init__(
        self,
        ann_file,  # path of the annotation json file
        subset_name,  # name of the subset, such as training, validation, testing
        data_path,  # folder path of the raw video / pre-extracted feature
        pipeline,  # data pipeline
        class_map,  # path of the class map, convert the class id to category name
        filter_gt=False,  # if True, filter out those gt has the scale smaller than 0.01
        class_agnostic=False,  # if True, the class index will be replaced by 0
        # some videos might be missed in the features or videos, we need to block them
        block_list=None,
        test_mode=False,  # if True, running on test mode with no annotation
        logger=None,
    ):
        self.data_path = data_path
        self.block_list = block_list
        self.ann_file = ann_file
        self.subset_name = subset_name
        self.logger = logger.info if logger is not None else print
        self.class_map = self._get_class_map(class_map)
        self.class_agnostic = class_agnostic
        self.filter_gt = filter_gt
        self.test_mode = test_mode
        self.pipeline = Pipeline(pipeline)
        self.data_list = []

    def __getitem__(self, idx):
        pass

    def __len__(self):
        return len(self.data_list)

    def get_gt(self, video_info):
        pass

    def build_data_list(self):
        anno_database, blocked_videos = self._load_annotation_database()

        self.data_list = []
        for video_name, video_info in anno_database.items():
            if (video_name in blocked_videos) or (video_info["subset"] not in self.subset_name):
                continue

            # get the ground truth annotation
            if self.test_mode:
                video_anno = {}
            else:
                video_anno = self.get_gt(video_info)
                if video_anno is None:  # have no valid gt
                    continue

            self.add_to_data_list(video_name, video_info, video_anno)
        assert len(self.data_list) > 0, f"No data found in {self.subset_name} subset."

    def get_fps(self, video_info):
        if self.fps > 0:
            return self.fps
        return float(video_info["frame"]) / float(video_info["duration"])

    def parse_and_filter_gt(
        self,
        video_info,
        thresh,
        segment_converter=None,
        ignore_label_func=None,
        custom_valid_func=None,
    ):
        gt_segment_s = []
        gt_segment_frame = []
        gt_label = []
        for anno in video_info["annotations"]:
            if ignore_label_func and ignore_label_func(anno["label"]):
                continue

            gt_start, gt_end = (
                segment_converter(anno["segment"]) if segment_converter else anno["segment"]
            )
            gt_scale = gt_end - gt_start

            if custom_valid_func:
                valid_gt = custom_valid_func(gt_start, gt_end, gt_scale)
            else:
                valid_gt = (not self.filter_gt) or (gt_scale > thresh)

            if valid_gt:
                gt_segment_s.append(anno["segment"])
                gt_segment_frame.append([gt_start, gt_end])
                if getattr(self, "class_agnostic", False):
                    gt_label.append(0)
                else:
                    gt_label.append(self.class_map.index(anno["label"]))

        if len(gt_segment_frame) == 0:
            return None
        unique_segments_s = []
        unique_segments_frame = []
        unique_labels = []
        seen = set()

        for frame, s, label in zip(gt_segment_frame, gt_segment_s, gt_label, strict=True):
            identifier = (tuple(frame), label)
            if identifier not in seen:
                unique_segments_frame.append(frame)
                unique_segments_s.append(s)
                unique_labels.append(label)
                seen.add(identifier)

        return dict(
            gt_segments_s=np.array(unique_segments_s, dtype=np.float32).reshape(-1, 2),
            gt_segments_frame=np.array(unique_segments_frame, dtype=np.float32).reshape(-1, 2),
            gt_labels=np.array(unique_labels, dtype=np.int32),
        )

    def add_to_data_list(self, video_name, video_info, video_anno):
        self.data_list.append([video_name, video_info, video_anno])

    def _get_class_map(self, class_map_path):
        if not Path(class_map_path).exists():
            class_map = generate_class_map(self.ann_file, class_map_path)
            self.logger(f"Class map is saved in {class_map_path}, total {len(class_map)} classes.")
        else:
            with open(class_map_path, encoding="utf8") as f:
                lines = f.readlines()
            class_map = [item.strip() for item in lines if item.strip()]
        return class_map

    def _load_annotation_database(self):
        with open(self.ann_file) as f:
            anno_database = json.load(f)["database"]

        # some videos might be missed in the features or videos, we need to block them
        if self.block_list is not None:
            if isinstance(self.block_list, list):
                blocked_videos = self.block_list
            else:
                with open(self.block_list) as f:
                    blocked_videos = [line.rstrip("\n") for line in f]
        else:
            blocked_videos = []
        return anno_database, blocked_videos


@DATASETS.register_module()
class PaddingDataset(BaseDataset):
    def __init__(
        self,
        feature_stride=-1,  # the frames between two adjacent features, such as 4 frames
        sample_stride=1,  # if you want to extract the feature[::sample_stride]
        offset_frames=0,  # the start offset frame of the input feature
        fps=-1,  # some annotations are based on video-seconds
        **kwargs,
    ):
        super().__init__(**kwargs)

        # feature settings
        self.feature_stride = feature_stride
        self.sample_stride = sample_stride
        self.offset_frames = int(offset_frames)
        self.snippet_stride = int(feature_stride * sample_stride)
        self.fps = fps

        self.build_data_list()
        self.logger(f"{self.subset_name} subset: {len(self.data_list)} videos")

    def __getitem__(self, index):

        video_name, video_info, video_anno = self.data_list[index]
        if video_anno:
            video_anno = deepcopy(video_anno)
            video_anno["gt_segments_feat"] = (
                video_anno["gt_segments_frame"] - self.offset_frames
            ) / self.snippet_stride
        return self.pipeline(
            dict(
                video_name=video_name,
                data_path=self.data_path,
                sample_stride=self.sample_stride,
                snippet_stride=self.snippet_stride,
                fps=self.get_fps(video_info),
                duration=float(video_info["duration"]),
                offset_frames=self.offset_frames,
                **(video_anno or {}),
            )
        )


@DATASETS.register_module()
class ResizeDataset(BaseDataset):
    def __init__(
        self,
        resize_length=128,  # the length of the resized video
        sample_stride=1,  # if you want to extract the feature[::sample_stride]
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.resize_length = resize_length
        self.sample_stride = sample_stride

        self.build_data_list()
        self.logger(f"{self.subset_name} subset: {len(self.data_list)} videos")

    def __getitem__(self, index):

        video_name, video_info, video_anno = self.data_list[index]
        if video_anno:
            video_anno = deepcopy(video_anno)
        return self.pipeline(
            dict(
                video_name=video_name,
                data_path=self.data_path,
                resize_length=self.resize_length,
                sample_stride=self.sample_stride,
                fps=-1,
                duration=float(video_info["duration"]),
                **(video_anno or {}),
            )
        )


@DATASETS.register_module()
class SlidingWindowDataset(BaseDataset):
    def __init__(
        self,
        # for feature setting
        feature_stride=-1,  # the frames between two adjacent features, such as 4 frames
        sample_stride=1,  # if you want to extract the feature[::sample_stride]
        offset_frames=0,  # the start offset frame of the input feature
        # for sliding window setting
        window_size=-1,  # the number of features in a window
        window_overlap_ratio=0.25,  # the overlap ratio of two adjacent windows
        ioa_thresh=0.75,  # the threshold of the completeness of the gt inside the window
        fps=-1,  # some annotations are based on video-seconds
        **kwargs,
    ):
        super().__init__(**kwargs)

        # feature settings
        self.feature_stride = int(feature_stride)
        self.sample_stride = int(sample_stride)
        self.offset_frames = int(offset_frames)
        self.snippet_stride = int(feature_stride * sample_stride)
        self.fps = fps

        # window settings
        self.window_size = int(window_size)
        self.window_stride = int(window_size * (1 - window_overlap_ratio))
        self.ioa_thresh = ioa_thresh

        self.build_data_list()
        self.logger(
            f"{self.subset_name} subset: {len(set([data[0] for data in self.data_list]))} videos, "
            f"truncated as {len(self.data_list)} windows."
        )

    def __getitem__(self, index):

        video_name, video_info, video_anno, window_snippet_centers = self.data_list[index]
        if video_anno:
            video_anno = deepcopy(video_anno)
            video_anno["gt_segments_feat"] = (
                video_anno["gt_segments_frame"] - window_snippet_centers[0] - self.offset_frames
            ) / self.snippet_stride
        return self.pipeline(
            dict(
                video_name=video_name,
                data_path=self.data_path,
                window_size=self.window_size,
                feature_start_idx=int(window_snippet_centers[0] / self.snippet_stride),
                feature_end_idx=int(window_snippet_centers[-1] / self.snippet_stride),
                sample_stride=self.sample_stride,
                fps=self.get_fps(video_info),
                snippet_stride=self.snippet_stride,
                window_start_frame=window_snippet_centers[0],
                duration=video_info["duration"],
                offset_frames=self.offset_frames,
                **(video_anno or {}),
            )
        )

    def add_to_data_list(self, video_name, video_info, video_anno):
        tmp_data_list = self.split_video_to_windows(video_name, video_info, video_anno)
        self.data_list.extend(tmp_data_list)

    def split_video_to_windows(self, video_name, video_info, video_anno):
        # need: video frame, video duration, video fps
        if self.fps > 0:
            num_frames = int(video_info["duration"] * self.fps)
        else:
            num_frames = video_info["frame"]

        video_snippet_centers = np.arange(0, num_frames, self.snippet_stride)
        snippet_num = len(video_snippet_centers)

        data_list = []
        last_window = False  # whether it is the last window

        for idx in range(max(1, snippet_num // self.window_stride)):  # at least one window
            window_start = idx * self.window_stride
            window_end = window_start + self.window_size

            if window_end > snippet_num:  # this is the last window
                window_end = snippet_num
                window_start = max(0, window_end - self.window_size)
                last_window = True

            window_snippet_centers = video_snippet_centers[window_start:window_end]
            window_start_frame = window_snippet_centers[0]
            window_end_frame = window_snippet_centers[-1]

            if video_anno and self.ioa_thresh > 0:
                gt_segments_frame = video_anno["gt_segments_frame"]
                gt_labels = video_anno["gt_labels"]
                anchor = np.array([window_start_frame, window_end_frame])

                # truncate the gt segments inside the window and compute the completeness
                gt_completeness, truncated_gt = compute_gt_completeness(gt_segments_frame, anchor)
                valid_idx = gt_completeness > self.ioa_thresh

                # only append window who has gt
                if np.sum(valid_idx) > 0:
                    window_anno = dict(
                        gt_segments_frame=truncated_gt[valid_idx],
                        gt_labels=gt_labels[valid_idx],
                    )
                    data_list.append(
                        [
                            video_name,
                            video_info,
                            window_anno,
                            window_snippet_centers,
                        ]
                    )
            else:
                data_list.append(
                    [
                        video_name,
                        video_info,
                        video_anno,
                        window_snippet_centers,
                    ]
                )

            if last_window:  # the last window
                break

        return data_list


def compute_gt_completeness(gt_boxes, anchors):
    """Compute the completeness of the gt_bboxes.
       GT will be first truncated by the anchor start/end,
       then the completeness is defined as the ratio of the
       truncated_gt_len / original_gt_len.
       If this ratio is too small, it means this gt is not complete enough to be used for training.
    Args:
        gt_boxes: np.array shape [N, 2]
        anchors:  np.array shape [2]
    """

    scores = np.zeros(gt_boxes.shape[0])  # initialized as 0
    valid_idx = np.logical_and(gt_boxes[:, 0] < anchors[1], gt_boxes[:, 1] > anchors[0])  # valid gt
    valid_gt_boxes = gt_boxes[valid_idx]

    truncated_valid_gt_len = np.minimum(valid_gt_boxes[:, 1], anchors[1]) - np.maximum(
        valid_gt_boxes[:, 0], anchors[0]
    )
    original_valid_gt_len = np.maximum(valid_gt_boxes[:, 1] - valid_gt_boxes[:, 0], 1e-6)
    scores[valid_idx] = truncated_valid_gt_len / original_valid_gt_len

    # also truncated gt
    truncated_gt_boxes = np.stack(
        [np.maximum(gt_boxes[:, 0], anchors[0]), np.minimum(gt_boxes[:, 1], anchors[1])], axis=1
    )
    return scores, truncated_gt_boxes  # shape [N]
