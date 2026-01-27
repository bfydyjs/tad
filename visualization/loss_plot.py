import pandas as pd
import matplotlib.pyplot as plt
import os

# 读取 CSV 文件
df = pd.read_csv(r"C:\Users\yanho\Downloads\wandb_export_2026-01-27T15_29_05.168+08_00.csv")

# 查看列名（可选，用于调试）
print("Columns:", df.columns.tolist())

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
plt.figure(figsize=(10, 6))
plt.plot(df[step_col], df[loss_col], label="Training Loss", color="tab:blue")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "..", "output", "figures", "loss_plot.png")
output_path = os.path.normpath(output_path)
print(f"Saving figure to: {output_path}")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300)

plt.show()