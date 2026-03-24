from torch.nn.functional import max_pool1d


def boundary_choose(score):
    mask_high = score > score.max(dim=1, keepdim=True)[0] * 0.5
    mask_peak = score == max_pool1d(score, kernel_size=3, stride=1, padding=1)
    mask = mask_peak | mask_high
    return mask


def convert_to_seconds(segments, meta):
    if meta["fps"] == -1:  # resize setting, like in anet / hacs
        segments = segments / meta["resize_length"] * meta["duration"]
    else:  # sliding window / padding setting, like in thumos / ego4d
        snippet_stride = meta["snippet_stride"]
        offset_frames = meta["offset_frames"]
        window_start_frame = (
            meta["window_start_frame"] if "window_start_frame" in meta.keys() else 0
        )
        segments = (segments * snippet_stride + window_start_frame + offset_frames) / meta["fps"]

    # truncate all boundaries within [0, duration]
    if segments.shape[0] > 0:
        segments[segments <= 0.0] *= 0.0
        segments[segments >= meta["duration"]] = (
            segments[segments >= meta["duration"]] * 0.0 + meta["duration"]
        )
    return segments
