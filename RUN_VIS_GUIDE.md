# 可视化脚本运行指南

## 🚀 快速开始

### 方法 1：使用运行器（最简单）⭐

**直接点击 VS Code 右上角的运行按钮**，然后选择配置：
- 选择 **"Python: Visualize Scripts"**

或者在终端运行：
```bash
python run_vis.py
```

这会显示交互式菜单，让你选择要运行的脚本。

---

### 方法 2：指定脚本名称

在终端运行：
```bash
python run_vis.py layer-wise_cka_similarity_heatmap
python run_vis.py loss
python run_vis.py map_heatmap
```

---

### 方法 3：使用 -m 参数（标准方式）

在终端运行：
```bash
python -m tad.visualization.plot.layer-wise_cka_similarity_heatmap
python -m tad.visualization.plot.loss
python -m tad.visualization.plot.map_heatmap
```

---

## ⚙️ VS Code 配置说明

### 运行配置

`.vscode/launch.json` 提供了两种配置：

#### 1. **Python: Visualize Scripts** （推荐）
- 自动处理模块导入
- 支持所有可视化脚本
- 使用统一的入口 `run_vis.py`

#### 2. **Python: Current File (Module Mode)**
- 直接运行当前打开的文件
- 以模块模式运行（支持相对导入）

### 切换运行配置

1. 按 `F5` 或点击运行按钮
2. 点击运行配置名称（顶部中间）
3. 选择想要的配置

---

## 📁 可用脚本列表

| 脚本名称 | 功能描述 |
|---------|---------|
| `layer-wise_cka_similarity_heatmap` | CKA 相似度热力图 |
| `loss` | 训练损失曲线 |
| `map_heatmap` | mAP 热力图 |

---

## 🔧 故障排除

### 问题 1：ImportError: attempted relative import with no known parent package

**原因：** 直接运行 Python 文件而不是作为模块运行

**解决：**
- ✅ 使用 `python run_vis.py <script_name>`
- ✅ 使用 `python -m tad.visualization.plot.<script_name>`
- ✅ 在 VS Code 中使用 "Python: Visualize Scripts" 配置

### 问题 2：ModuleNotFoundError: No module named 'tad'

**原因：** PYTHONPATH 未设置

**解决：**
```bash
# Windows PowerShell
$env:PYTHONPATH="C:\Users\yanho\Desktop\git\tad"

# Linux/Mac
export PYTHONPATH=/path/to/tad
```

或者直接使用 `run_vis.py`（已自动处理路径）

---

## 💡 最佳实践

1. **统一使用 `run_vis.py`** 
   - 一个命令运行所有脚本
   - 自动处理路径和导入
   - 提供友好的错误提示

2. **在 VS Code 中配置默认运行器**
   - 打开任意 `.py` 文件
   - 点击右上角运行按钮
   - 选择 "Python: Visualize Scripts"

3. **需要调试时使用 -m 方式**
   ```bash
   python -m tad.visualization.plot.loss
   ```

---

## 📝 添加新的可视化脚本

1. 在 `tad/visualization/plot/` 目录下创建新文件
2. 实现 `main()` 函数
3. 使用 `from ..utils import save_figure` 导入工具
4. 自动被 `run_vis.py` 识别

示例：
```python
from ..utils import save_figure

def main():
    # 你的绘图代码
    save_figure("my_new_plot")
    plt.show()

if __name__ == "__main__":
    # 不需要这个，run_vis.py 会调用 main()
    pass
```

---

## 🎯 总结

**最简单的使用方式：**

1. 打开 VS Code
2. 打开任意可视化脚本文件
3. 点击运行按钮（▶️）
4. 选择 "Python: Visualize Scripts"
5. 完成！✨

或者在终端：
```bash
python run_vis.py
```
