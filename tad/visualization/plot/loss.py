from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from setup_paper_style import setup_paper_style

df = pd.read_csv(r"C:\Users\yanho\Downloads\wandb_export_2026-01-27T15_29_05.168+08_00.csv")
step_col = "Step"
loss_col = "0125_1101 - train/loss"

if loss_col not in df.columns:
    for col in df.columns:
        if "train/loss" in col:
            loss_col = col
            break
    else:
        raise KeyError("未能找到包含 'train/loss' 的列")

setup_paper_style(440 / 2, ratio=1.618, fraction=0.98, font_size_tex=5, font_size_main=4.5, line_width_axis=0.5)
plt.figure()
plt.plot(df[step_col], df[loss_col], label="Training Loss", color="tab:blue")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid(True, linestyle="--")
plt.legend()
plt.tight_layout()

output_path = Path(__file__).resolve().parent.parent.parent.parent / "output" / "figures" / "training_loss_curve.png"
print(f"Saving figure to: {output_path}")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path)

plt.show()
