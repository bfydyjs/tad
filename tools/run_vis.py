#!/usr/bin/env python3
"""可视化脚本运行器。

使用方法:
    python run_vis.py layer_wise_cka_similarity_heatmap
    python run_vis.py loss
    python run_vis.py map_heatmap

"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))


def run_script(script_name: str) -> None:

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
    plot_dir = project_root / "tad" / "visualization" / "plot"
    scripts = []

    if plot_dir.exists():
        for file in plot_dir.glob("*.py"):
            # Skip special scripts that shouldn't be run directly
            if file.name.startswith("_") or file.name == "__init__.py":
                continue

            # Skip recall.py - it should be called from eval.py with --plot-recall
            # if file.stem == "recall":
            #     continue

            scripts.append(file.stem)

    return sorted(scripts)


def run_all_scripts() -> None:
    """Run all visualization scripts in order."""
    scripts = list_available_scripts()

    if not scripts:
        print("No visualization scripts found!")
        return

    print(f"\n=== Running All {len(scripts)} Visualization Scripts ===\n")

    for i, script in enumerate(scripts, 1):
        print(f"\n[{i}/{len(scripts)}] Running {script}...")
        try:
            module_path = f"tad.visualization.plot.{script}"
            module = __import__(module_path, fromlist=["main"])

            if hasattr(module, "main"):
                module.main()
                print(f"✓ {script} completed!")
            else:
                print(f"✗ {script}: No main() function")
        except Exception as e:
            print(f"✗ {script} failed: {e}")

    print("\n=== All Scripts Completed ===")


def interactive_mode() -> None:
    scripts = list_available_scripts()

    if not scripts:
        print("No visualization scripts found!")
        return

    print("\n=== Available Visualization Scripts ===\n")
    for i, script in enumerate(scripts, 1):
        print(f"  {i}. {script}")

    print("\nEnter the number of the script to run (or 'all' to run all, 'q' to quit):")

    while True:
        choice = input("> ").strip()

        if choice.lower() == "q":
            return
        elif choice.lower() == "all":
            run_all_scripts()
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
    if len(sys.argv) > 1:
        script_name = sys.argv[1]
        if script_name == "all":
            run_all_scripts()
        else:
            run_script(script_name)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
