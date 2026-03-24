import pickle
from pathlib import Path

import torch


def save_features(features, metas, folder):
    save_dir = Path(folder) / "features"
    save_dir.mkdir(parents=True, exist_ok=True)

    for i in range(len(metas)):
        video_name = metas[i]["video_name"]

        # Detach and move to CPU to avoid memory leak
        if isinstance(features, (list, tuple)):
            feat_to_save = [feat[i].detach().cpu() for feat in features]
        else:
            feat_to_save = features[i].detach().cpu()

        file_path = save_dir / f"{video_name}.pkl"
        with open(file_path, "wb") as outfile:
            pickle.dump(feat_to_save, outfile, pickle.HIGHEST_PROTOCOL)


def save_raw_predictions(predictions, metas, folder):
    save_dir = Path(folder) / "raw_prediction"
    save_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(len(metas)):
        video_name = metas[idx]["video_name"]

        file_path = save_dir / f"{video_name}.pkl"
        prediction = [data[idx] for data in predictions]
        with open(file_path, "wb") as outfile:
            pickle.dump(prediction, outfile, pickle.HIGHEST_PROTOCOL)


def load_single_prediction(metas, folder):
    """Should not be used for sliding window. Since we saved the files with
    video name, and sliding window will have multiple files with the same
    name.
    """
    predictions = []
    for idx in range(len(metas)):
        video_name = metas[idx]["video_name"]
        file_path = Path(folder) / f"{video_name}.pkl"
        with open(file_path, "rb") as infile:
            prediction = pickle.load(infile)
        predictions.append(prediction)

    batched_predictions = []
    for i in range(len(predictions[0])):
        data = torch.stack([prediction[i] for prediction in predictions])
        batched_predictions.append(data)
    return batched_predictions


def load_predictions(metas, infer_cfg):
    if "fuse_list" in infer_cfg.keys():
        predictions = []
        predictions_list = [load_single_prediction(metas, folder) for folder in infer_cfg.fuse_list]
        for i in range(len(predictions_list[0])):
            predictions.append(torch.stack([pred[i] for pred in predictions_list]).mean(dim=0))
        return predictions
    else:
        return load_single_prediction(metas, infer_cfg.work_dir)
