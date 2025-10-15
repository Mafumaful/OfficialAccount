#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import subprocess
from pathlib import Path
import argparse

REPO_ROOT = Path(__file__).resolve().parents[1]


def discover_scenes():
    """
    Discover manim scenes in project subdirectories.
    A scene is identified as a Python class inheriting from Scene and a file path.
    Returns list of tuples: (display_name, workdir, file_path, scene_class)
    """
    candidates = []

    # Heuristic: look for python files directly under subdirectories (one level deep)
    for sub in REPO_ROOT.iterdir():
        if not sub.is_dir():
            continue
        # skip virtual envs and dot folders
        if sub.name.startswith('.') or sub.name in {"venv", "env", "__pycache__", "scripts"}:
            continue
        for py in sub.glob("*.py"):
            try:
                text = py.read_text(encoding="utf-8")
            except Exception:
                continue
            # Find classes that look like `class Something(Scene):`
            for m in re.finditer(r"class\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*Scene\s*\)\s*:\s*", text):
                scene = m.group(1)
                display = f"{sub.name} / {py.name} :: {scene}"
                candidates.append((display, sub, py.name, scene))

    # Sort for stable menu order
    candidates.sort(key=lambda x: x[0].lower())
    return candidates


def choose(candidates):
    print("可运行的项目如下：\n")
    for idx, (display, _, _, _) in enumerate(candidates, start=1):
        print(f"{idx}. {display}")
    print()
    while True:
        try:
            sel = input("请输入要运行的序号: ").strip()
        except EOFError:
            sys.exit(1)
        if not sel.isdigit():
            print("请输入数字编号。")
            continue
        n = int(sel)
        if 1 <= n <= len(candidates):
            return candidates[n - 1]
        print("超出范围，请重新输入。")


def run_manim(workdir: Path, file_name: str, scene_class: str, *, debug: bool = False):
    # 根据是否为 debug 模式，选择不同清晰度
    # -pql: 预览 + 低清晰度；-pqh: 预览 + 高清晰度
    quality_flag = "-pql" if debug else "-pqh"
    cmd = [
        "bash",
        "-lc",
        f"cd '{workdir}' && manim {quality_flag} '{file_name}' {scene_class}"
    ]
    print("\n正在执行: ", cmd[-1])
    proc = subprocess.run(cmd)
    return proc.returncode


def main():
    parser = argparse.ArgumentParser(description="Manim 场景运行菜单")
    parser.add_argument("--debug", action="store_true", help="调试模式，使用低分辨率/低清晰度渲染")
    args = parser.parse_args()

    scenes = discover_scenes()
    if not scenes:
        print("未发现可运行的Manim场景。请在子目录中添加包含 Scene 派生类的 .py 文件。")
        return 1
    display, workdir, file_name, scene_class = choose(scenes)
    return run_manim(workdir, file_name, scene_class, debug=args.debug)


if __name__ == "__main__":
    sys.exit(main())


