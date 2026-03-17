import json


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
