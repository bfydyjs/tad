import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .builder import EVALUATORS, remove_duplicate_annotations
from .map import segment_iou


@EVALUATORS.register_module()
class Recall:
    def __init__(
        self,
        ground_truth_file,
        prediction_file,
        subset,
        tiou_thresholds,
        topk=None,
        max_avg_proposals_per_video=100,
        blocked_videos=None,
    ):
        if topk is None:
            topk = [1, 5, 10, 100]
        super().__init__()

        if not ground_truth_file:
            raise OSError("Please input a valid ground truth file.")
        if not prediction_file:
            raise OSError("Please input a valid prediction file.")

        self.subset = subset
        self.tiou_thresholds = tiou_thresholds
        self.max_avg_proposals_per_video = max_avg_proposals_per_video
        self.topk = [int(k) for k in topk]
        self.gt_fields = ["database"]
        self.pred_fields = ["results"]

        # Get blocked videos
        if blocked_videos is None:
            self.blocked_videos = list()
        else:
            with open(blocked_videos) as json_file:
                self.blocked_videos = json.load(json_file)

        # Import ground truth and proposals.
        self.ground_truth, self.activity_index = self._import_ground_truth(ground_truth_file)
        self.proposal = self._import_proposal(prediction_file)

    def _import_ground_truth(self, ground_truth_file):
        """Reads ground truth file, checks if it is well formatted, and returns
           the ground truth instances and the activity classes.
        Parameters
        ----------
        ground_truth_file : str
            Full path to the ground truth json file.
        Outputs
        -------
        ground_truth : df
            Data frame containing the ground truth instances.
        activity_index : dict
            Dictionary containing class index.
        """
        with open(ground_truth_file) as fobj:
            data = json.load(fobj)
        # Checking format
        if not all([field in list(data.keys()) for field in self.gt_fields]):
            raise OSError("Please input a valid ground truth file.")

        # Read ground truth data.
        activity_index, cidx = {}, 0
        video_lst, t_start_lst, t_end_lst, label_lst = [], [], [], []
        for videoid, v in data["database"].items():
            if self.subset != v["subset"]:
                continue
            if videoid in self.blocked_videos:
                continue

            # remove duplicated instances following ActionFormer
            v_anno = remove_duplicate_annotations(v["annotations"])

            for ann in v_anno:
                if ann["label"] not in activity_index:
                    activity_index[ann["label"]] = cidx
                    cidx += 1
                video_lst.append(videoid)
                t_start_lst.append(float(ann["segment"][0]))
                t_end_lst.append(float(ann["segment"][1]))
                label_lst.append(activity_index[ann["label"]])

        ground_truth = pd.DataFrame(
            {
                "video-id": video_lst,
                "t-start": t_start_lst,
                "t-end": t_end_lst,
                "label": label_lst,
            }
        )
        return ground_truth, activity_index

    def _import_proposal(self, proposal_file):
        """Reads proposal file, checks if it is well formatted, and returns
           the proposal instances.
        Parameters
        ----------
        proposal_file : str
            Full path to the proposal json file.
        Outputs
        -------
        proposal : df
            Data frame containing the proposal instances.
        """
        # if prediction_file is a string, then json load
        if isinstance(proposal_file, str):
            with open(proposal_file) as fobj:
                data = json.load(fobj)
        elif isinstance(proposal_file, dict):
            data = proposal_file
        else:
            raise OSError(f"Type of prediction file is {type(proposal_file)}.")

        # Checking format...
        if not all([field in list(data.keys()) for field in self.pred_fields]):
            raise OSError("Please input a valid proposal file.")

        # Read predictions.
        video_lst, t_start_lst, t_end_lst = [], [], []
        score_lst = []
        for videoid, v in data["results"].items():
            if videoid in self.blocked_videos:
                continue
            for result in v:
                video_lst.append(videoid)
                t_start_lst.append(float(result["segment"][0]))
                t_end_lst.append(float(result["segment"][1]))
                score_lst.append(result["score"])
        proposal = pd.DataFrame(
            {
                "video-id": video_lst,
                "t-start": t_start_lst,
                "t-end": t_end_lst,
                "score": score_lst,
            }
        )
        return proposal

    def evaluate(self):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        """
        recall, avg_recall, proposals_per_video = average_recall_vs_avg_nr_proposals(
            self.ground_truth,
            self.proposal,
            max_avg_nr_proposals=self.max_avg_proposals_per_video,
            tiou_thresholds=self.tiou_thresholds,
        )

        area_under_curve = np.trapz(avg_recall, proposals_per_video)

        self.recall = recall
        self.avg_recall = avg_recall
        self.proposals_per_video = proposals_per_video
        self.average_auc = float(area_under_curve) / proposals_per_video[-1]

        # Calculate AUC for each tIoU threshold
        self.auc_per_tiou = []
        for i, _ in enumerate(self.tiou_thresholds):
            tiou_area = np.trapz(recall[i, :], proposals_per_video)
            tiou_auc = float(tiou_area) / proposals_per_video[-1]
            self.auc_per_tiou.append(tiou_auc)

        metric_dict = dict(average_AUC=self.average_auc)
        # Add per tIoU AUC to metric_dict
        for _, (tiou, auc) in enumerate(zip(self.tiou_thresholds, self.auc_per_tiou, strict=False)):
            metric_dict[f"AUC@{tiou}"] = auc
        # Add per tIoU AR@k to metric_dict
        for k in self.topk:
            metric_dict[f"Average Recall@{k}"] = self.avg_recall[k - 1]
            # Add per tIoU AR@k
            for i, tiou in enumerate(self.tiou_thresholds):
                metric_dict[f"Recall@{tiou}@{k}"] = self.recall[i, k - 1]
        self.plot_recall_curves
        return metric_dict

    def logging(self, logger=None):
        if logger is None:
            pprint = print
        else:
            pprint = logger.info

        pprint(f"Loaded annotations from {self.subset} subset.")
        pprint(f"Number of ground truth instances: {len(self.ground_truth)}")
        pprint(f"Number of predictions: {len(self.proposal)}")
        pprint(f"Fixed threshold for tiou score: {self.tiou_thresholds}")
        pprint(f"average_AUC: {self.average_auc * 100:>4.2f} (%)")
        # Print per tIoU AUC
        for _, (tiou, auc) in enumerate(zip(self.tiou_thresholds, self.auc_per_tiou, strict=False)):
            pprint(f"AUC@{tiou:.2f} is {auc * 100:>4.2f}%")
        pprint("")
        # Print average AR@k
        for k in self.topk:
            pprint(f"Average Recall@{k:3d} is {self.avg_recall[k - 1] * 100:>4.2f}%")
        # Print per tIoU Recall@k
        for i, tiou in enumerate(self.tiou_thresholds):
            pprint("")
            pprint(f"tIoU={tiou:.2f}:")
            for k in self.topk:
                pprint(f"Recall@{k:3d} is {self.recall[i, k - 1] * 100:>4.2f}%")

    def plot_recall_curves(self):
        """Plots recall curves for each k value, with tiou_thresholds on x-axis"""
        # Define colors for different k values
        colors = ["r", "g", "b", "c", "m", "y", "k"]

        # Create subplots for different k values
        num_k = len(self.topk)
        fig, axes = plt.subplots(1, num_k, figsize=(5 * num_k, 4))

        # Ensure axes is always a list
        if num_k == 1:
            axes = [axes]

        # Plot each k value
        for k_idx, k in enumerate(self.topk):
            ax = axes[k_idx]

            # For each tiou threshold, get the recall value at k
            recall_values = self.recall[:, k - 1]

            # Plot the curve
            ax.plot(self.tiou_thresholds, recall_values, "o-", color=colors[k_idx % len(colors)], label=f"Recall@{k}")

            # Set labels and title
            ax.set_xlabel("tIoU Thresholds")
            ax.set_ylabel(f"per tIoU Recall@{k}")
            ax.set_title(f"Recall@{k} vs tIoU Thresholds")
            ax.grid(True)
            ax.legend()

            # Set x-axis limits
            ax.set_xlim([min(self.tiou_thresholds) - 0.05, max(self.tiou_thresholds) + 0.05])
            # Set y-axis limits
            ax.set_ylim([0, 1.05])

        # Adjust layout
        plt.tight_layout()

        # Set default save path if not provided
        output_path = Path(__file__).resolve().parent.parent.parent / "output" / "figures" / "recall_curves.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Recall curves saved to: {output_path}")

        # Show the figure
        plt.show()


def average_recall_vs_avg_nr_proposals(
    ground_truth,
    proposals,
    max_avg_nr_proposals=None,
    tiou_thresholds=None,
):
    """Computes the average recall given an average number
        of proposals per video.
    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    proposal : df
        Data frame containing the proposal instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        array with tiou thresholds.
    Outputs
    -------
    recall : 2darray
        recall[i,j] is recall at ith tiou threshold at the jth average number of average number of proposals per video.
    average_recall : 1darray
        recall averaged over a list of tiou threshold. This is equivalent to recall.mean(axis=0).
    proposals_per_video : 1darray
        average number of proposals per video.
    """
    if tiou_thresholds is None:
        tiou_thresholds = np.linspace(0.5, 0.95, 10)

    # Get list of videos.
    video_lst = ground_truth["video-id"].unique()

    if not max_avg_nr_proposals:
        max_avg_nr_proposals = float(proposals.shape[0]) / video_lst.shape[0]

    ratio = max_avg_nr_proposals * float(video_lst.shape[0]) / proposals.shape[0]

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby("video-id")
    proposals_gbvn = proposals.groupby("video-id")

    # For each video, computes tiou scores among the retrieved proposals.
    score_lst = []
    total_nr_proposals = 0
    for videoid in video_lst:
        # Get ground-truth instances associated to this video.
        ground_truth_videoid = ground_truth_gbvn.get_group(videoid)
        this_video_ground_truth = ground_truth_videoid.loc[:, ["t-start", "t-end"]].values

        # Get proposals for this video.
        try:
            proposals_videoid = proposals_gbvn.get_group(videoid)
            this_video_proposals = proposals_videoid.loc[:, ["t-start", "t-end"]].values

            # Sort proposals by score.
            sort_idx = proposals_videoid["score"].argsort()[::-1]
            this_video_proposals = this_video_proposals[sort_idx, :]
        except:
            n = this_video_ground_truth.shape[0]
            score_lst.append(np.zeros((n, 1)))
            continue

        if this_video_proposals.shape[0] == 0:
            n = this_video_ground_truth.shape[0]
            score_lst.append(np.zeros((n, 1)))
            continue

        if this_video_proposals.ndim != 2:
            this_video_proposals = np.expand_dims(this_video_proposals, axis=0)
        if this_video_ground_truth.ndim != 2:
            this_video_ground_truth = np.expand_dims(this_video_ground_truth, axis=0)

        nr_proposals = np.minimum(int(this_video_proposals.shape[0] * ratio), this_video_proposals.shape[0])
        total_nr_proposals += nr_proposals
        this_video_proposals = this_video_proposals[:nr_proposals, :]

        # Compute tiou scores.
        tiou = wrapper_segment_iou(this_video_proposals, this_video_ground_truth)
        score_lst.append(tiou)

    # Given that the length of the videos is really varied, we
    # compute the number of proposals in terms of a ratio of the total
    # proposals retrieved, i.e. average recall at a percentage of proposals
    # retrieved per video.

    # Computes average recall.
    # pcn_lst = np.arange(1, 101) / 100.0 * (max_avg_nr_proposals * float(video_lst.shape[0]) / total_nr_proposals)
    pcn_lst = (
        np.arange(1, max_avg_nr_proposals + 1)
        / max_avg_nr_proposals
        * (max_avg_nr_proposals * float(video_lst.shape[0]) / total_nr_proposals)
    )
    matches = np.empty((video_lst.shape[0], pcn_lst.shape[0]))
    positives = np.empty(video_lst.shape[0])
    recall = np.empty((len(tiou_thresholds), pcn_lst.shape[0]))
    # Iterates over each tiou threshold.
    for ridx, tiou in enumerate(tiou_thresholds):
        # Inspect positives retrieved per video at different
        # number of proposals (percentage of the total retrieved).
        for i, score in enumerate(score_lst):
            # Total positives per video.
            positives[i] = score.shape[0]
            # Find proposals that satisfies minimum tiou threshold.
            true_positives_tiou = score >= tiou
            # Get number of proposals as a percentage of total retrieved.
            pcn_proposals = np.minimum((score.shape[1] * pcn_lst).astype(np.int), score.shape[1])

            for j, nr_proposals in enumerate(pcn_proposals):
                # Compute the number of matches for each percentage of the proposals
                matches[i, j] = np.count_nonzero((true_positives_tiou[:, :nr_proposals]).sum(axis=1))

        # Computes recall given the set of matches per video.
        recall[ridx, :] = matches.sum(axis=0) / positives.sum()

    # Recall is averaged.
    avg_recall = recall.mean(axis=0)

    # Get the average number of proposals per video.
    proposals_per_video = pcn_lst * (float(total_nr_proposals) / video_lst.shape[0])

    return recall, avg_recall, proposals_per_video


def wrapper_segment_iou(target_segments, candidate_segments):
    """Compute intersection over union btw segments
    Parameters
    ----------
    target_segments : ndarray
        2-dim array in format [m x 2:=[init, end]]
    candidate_segments : ndarray
        2-dim array in format [n x 2:=[init, end]]
    Outputs
    -------
    tiou : ndarray
        2-dim array [n x m] with IOU ratio.
    Note: It assumes that candidate-segments are more scarce that target-segments
    """
    if candidate_segments.ndim != 2 or target_segments.ndim != 2:
        raise ValueError("Dimension of arguments is incorrect")

    n, m = candidate_segments.shape[0], target_segments.shape[0]
    tiou = np.empty((n, m))
    for i in range(m):
        tiou[:, i] = segment_iou(target_segments[i, :], candidate_segments)

    return tiou
