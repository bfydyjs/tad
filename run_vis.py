#!/usr/bin/env python3
"""可视化脚本运行器。

使用方法:
    python run_vis.py layer_wise_cka_similarity_heatmap
    python run_vis.py loss
    python run_vis.py map_heatmap

或者直接双击运行（会提示选择脚本）
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))


def run_script(script_name: str) -> None:
    """运行指定的可视化脚本。

    Args:
        script_name: 脚本名称（不含 .py 扩展名）
    """
    try:
        # 动态导入模块
        module_path = f"tad.visualization.plot.{script_name}"
        module = __import__(module_path, fromlist=["main"])

        # 调用 main 函数
        if hasattr(module, "main"):
            print(f"Running {script_name}...")
            module.main()
            print(f"✓ {script_name} completed successfully!")
        else:
            print(f"✗ Error: {script_name} has no main() function")

    except ImportError as e:
        print(f"✗ Error: Cannot import {script_name}")
        print(f"  Details: {e}")
    except Exception as e:
        print(f"✗ Error running {script_name}")
        print(f"  Details: {e}")


def list_available_scripts() -> list[str]:
    """列出所有可用的可视化脚本。"""
    plot_dir = project_root / "tad" / "visualization" / "plot"
    scripts = []

    if plot_dir.exists():
        for file in plot_dir.glob("*.py"):
            if not file.name.startswith("_") and file.name != "__init__.py":
                scripts.append(file.stem)

    return sorted(scripts)


def interactive_mode() -> None:
    """交互模式：让用户选择要运行的脚本。"""
    scripts = list_available_scripts()

    if not scripts:
        print("No visualization scripts found!")
        return

    print("\n=== Available Visualization Scripts ===\n")
    for i, script in enumerate(scripts, 1):
        print(f"  {i}. {script}")

    print("\nEnter the number of the script to run (or 'q' to quit):")

    while True:
        choice = input("> ").strip()

        if choice.lower() == "q":
            return

        try:
            index = int(choice) - 1
            if 0 <= index < len(scripts):
                run_script(scripts[index])
                break
            else:
                print(f"Please enter a number between 1 and {len(scripts)}")
        except ValueError:
            print("Invalid input. Please enter a number.")


def main():
    """主函数。"""
    if len(sys.argv) > 1:
        # 命令行参数模式
        script_name = sys.argv[1]
        run_script(script_name)
    else:
        # 交互模式
        interactive_mode()


if __name__ == "__main__":
    main()
