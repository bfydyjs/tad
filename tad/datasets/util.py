import json

import numpy as np


def generate_class_map(gt_json_path, class_map_path):
    with open(gt_json_path) as f:
        anno = json.load(f)

    anno = anno["database"]
    class_map = set()
    for video_name, video_info in anno.items():  # noqa: B007
        if "annotations" in video_info:
            for tmpp_data in video_info["annotations"]:
                class_map.add(tmpp_data["label"])

    class_list = sorted(list(class_map), key=lambda x: str(x))
    with open(class_map_path, "w") as f2:
        for name in class_list:
            f2.write(name + "\n")
    return class_list


def filter_same_annotation(annotation):
    gt_segments = []
    gt_labels = []
    seen = set()
    for gt_segment, gt_label in zip(
        annotation["gt_segments"].tolist(), annotation["gt_labels"].tolist(), strict=True
    ):
        identifier = (tuple(gt_segment), gt_label)
        if identifier not in seen:
            gt_segments.append(gt_segment)
            gt_labels.append(gt_label)
            seen.add(identifier)

    annotation = dict(
        gt_segments=np.array(gt_segments, dtype=np.float32).reshape(-1, 2)
        if gt_segments
        else np.empty((0, 2), dtype=np.float32),
        gt_labels=np.array(gt_labels, dtype=np.int32),
    )
    return annotation


if __name__ == "__main__":
    anno1 = dict(gt_segments=np.array([[3, 5], [3, 6], [3, 5]]), gt_labels=np.array([0, 1, 0]))
    print(filter_same_annotation(anno1))
    # output should be:
    # 'gt_segments': array([[3., 5.], [3., 6.]], dtype=float32),
    # 'gt_labels': array([0, 1], dtype=int32)}

    anno2 = dict(gt_segments=np.array([[3, 5], [3, 6], [3, 5]]), gt_labels=np.array([0, 1, 2]))
    print(filter_same_annotation(anno2))
    # output should be:
    # 'gt_segments': array([[3., 5.], [3., 6.], [3., 5.]], dtype=float32),
    # 'gt_labels': array([0, 1, 2], dtype=int32)}

    anno3 = dict(gt_segments=np.array([[3, 5], [3, 5], [3, 5]]), gt_labels=np.array([0, 1, 1]))
    print(filter_same_annotation(anno3))
    # output should be:
    # 'gt_segments': array([[3., 5.], [3., 5.]], dtype=float32),
    # 'gt_labels': array([0, 1], dtype=int32)}
