import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from setup_paper_style import setup_paper_style

# 读取 wandb 下载的 CSV 文件
df = pd.read_csv(r"C:\Users\yanho\Downloads\wandb_export_2026-01-27T15_29_05.168+08_00.csv")

# 提取 Step 和 loss 列
# 注意：列名中包含引号和空格，pandas 通常会自动去除外层引号，但保留内部
# 实际列名可能是：'Step', '0125_1101 - train/loss'
step_col = "Step"
loss_col = "0125_1101 - train/loss"

# 确保列存在
if loss_col not in df.columns:
    # 尝试模糊匹配（以防有空格或格式问题）
    for col in df.columns:
        if "train/loss" in col:
            loss_col = col
            break
    else:
        raise KeyError("未能找到包含 'train/loss' 的列")

# 绘图
setup_paper_style() # 配置论文风格
plt.figure()
plt.plot(df[step_col], df[loss_col], label="Training Loss", color="tab:blue")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid(True, linestyle="--")
plt.legend()
plt.tight_layout()

output_path = (Path(__file__).resolve().parent.parent.parent.parent / "output" / "figures" / "training_loss_curve.png")
print(f"Saving figure to: {output_path}")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path)

plt.show()