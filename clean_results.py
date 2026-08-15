# -*- coding: utf-8 -*-
"""
清空 results/ 目录内的所有文件（预测结果清理）。

用途：
    在 predict.py 完成后运行，删除 results/ 里的旧预测图。
    注意：清空不可恢复，如需保留请先备份 results/。

用法：
    python clean_results.py            # 询问确认后清空
    python clean_results.py --yes      # 跳过确认直接清空
    python clean_results.py --dir <路径>  # 自定义目标目录（默认 src.config OUTPUT_DIR）
"""
import argparse
import os
import shutil
import sys

from src.config import OUTPUT_DIR


def main():
    parser = argparse.ArgumentParser(description="清空 results/ 目录")
    parser.add_argument("--dir", default=OUTPUT_DIR, help="目标目录（默认 config.OUTPUT_DIR）")
    parser.add_argument("--yes", action="store_true", help="跳过确认")
    args = parser.parse_args()

    target = os.path.abspath(args.dir)

    # ======================
    # 安全校验：防止误删非结果目录
    # ======================
    basename = os.path.basename(os.path.normpath(target))
    project_root = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(target):
        print(f"目标目录不存在，无需清理: {target}")
        sys.exit(0)

    forbidden = {
        os.path.abspath(os.path.join(project_root, "data")),
        os.path.abspath(os.path.join(project_root, "checkpoints")),
        os.path.abspath(os.path.join(project_root, "src")),
        os.path.abspath(os.path.join(project_root, "experiments")),
        os.path.abspath(os.path.join(project_root, "logs")),
        project_root,
    }
    if target in forbidden:
        print(f"错误：拒绝清空受保护目录: {target}")
        sys.exit(1)
    if basename not in ("results", "predict"):
        print(f"警告：目标目录名不是 results/predict，请确认: {target}")
        if not args.yes:
            confirm = input("目录名异常，是否仍要清空？(y/N): ").strip().lower()
            if confirm not in ("y", "yes"):
                print("已取消。")
                sys.exit(0)

    # ======================
    # 统计
    # ======================
    files = [f for f in os.listdir(target) if os.path.isfile(os.path.join(target, f))]
    subdirs = [d for d in os.listdir(target) if os.path.isdir(os.path.join(target, d))]
    n_files = len(files)
    n_bytes = sum(os.path.getsize(os.path.join(target, f)) for f in files)

    print("=" * 50)
    print(f"目标目录: {target}")
    print(f"将删除: {n_files} 个文件, {subdirs if subdirs else '无子目录'}, 共 {n_bytes / 1024 / 1024:.2f} MB")
    print("=" * 50)

    if not args.yes:
        confirm = input("确认清空以上内容？此操作不可恢复 (y/N): ").strip().lower()
        if confirm not in ("y", "yes"):
            print("已取消。")
            sys.exit(0)

    # ======================
    # 删除（保留 results/ 目录本身）
    # ======================
    removed = 0
    for f in files:
        os.remove(os.path.join(target, f))
        removed += 1
    for d in subdirs:
        shutil.rmtree(os.path.join(target, d))
        removed += len(os.listdir(os.path.join(target, d))) if os.path.exists(os.path.join(target, d)) else 0

    print(f"清理完成：共删除 {n_files} 个文件。")
    print("重新生成结果请运行: python predict.py")


if __name__ == "__main__":
    main()
