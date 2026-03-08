import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 导入通用工具函数
from ..utils import save_figure, setup_paper_style


def load_cka_data() -> dict[str, tuple[np.ndarray, str]]:
    """加载 CKA 相似度数据。

    Returns:
        包含预编码器和后编码器数据的字典
    """
    cka_a_pre = np.array(
        [
            [1.000, 0.925, 0.832, 0.669, 0.557],
            [0.925, 1.000, 0.925, 0.721, 0.603],
            [0.832, 0.925, 1.000, 0.810, 0.679],
            [0.669, 0.721, 0.810, 1.000, 0.877],
            [0.557, 0.603, 0.679, 0.877, 1.000],
        ]
    )

    cka_a_post = np.array(
        [
            [1.000, 0.896, 0.801, 0.762, 0.752],
            [0.896, 1.000, 0.928, 0.844, 0.736],
            [0.801, 0.928, 1.000, 0.927, 0.722],
            [0.762, 0.844, 0.927, 1.000, 0.749],
            [0.752, 0.736, 0.722, 0.749, 1.000],
        ]
    )

    return {
        "pre_encoder": (cka_a_pre, "(a) Pre-Encoder"),
        "post_encoder": (cka_a_post, "(b) Post-Encoder"),
    }


def plot_heatmaps(data_dict: dict[str, tuple[np.ndarray, str]]) -> None:
    """绘制热力图。

    Args:
        data_dict: 包含 CKA 数据和标题的字典
    """
    setup_paper_style(
        textwidth=440,
        ratio=1.618,
        fraction=0.98,
        font_size_tex=5,
        font_size_main=4.5,
        line_width_axis=0.5,
    )
    _, axes = plt.subplots(1, 2, figsize=(10, 4))

    # 绘制预编码器热力图
    sns.heatmap(
        data_dict["pre_encoder"][0],
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        cbar=False,
        ax=axes[0],
    )
    axes[0].set_title(data_dict["pre_encoder"][1])
    axes[0].set_xlabel("Scale Level")
    axes[0].set_ylabel("Scale Level")

    # 绘制后编码器热力图
    sns.heatmap(
        data_dict["post_encoder"][0],
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        cbar=False,
        ax=axes[1],
    )
    axes[1].set_title(data_dict["post_encoder"][1])
    axes[1].set_xlabel("Scale Level")
    axes[1].set_ylabel("Scale Level")

    plt.tight_layout()


def main():
    data_dict = load_cka_data()
    plot_heatmaps(data_dict)
    save_figure("layer-wise_cka_similarity_heatmap")
    plt.show()


if __name__ == "__main__":
    main()
