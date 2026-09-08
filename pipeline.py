# -*- coding: utf-8 -*-
"""
pipeline.py — 语料质量评估一键流水线

功能：
    1. 读取包含多条语料的文件（纯文本，每行一条）
    2. 逐条调用 eval_loop.py 进行评估（批量处理，默认每条约 1 轮）
    3. 评估报告统一保存到 reports/ 目录
    4. 收集全部报告，合并后调用 convert.py 转换为 Alpaca / ShareGPT 格式
    5. 调用 visualize.py 输出统计报告

用法示例：
    python pipeline.py --input corpus.txt --format alpaca --threshold 5
    python pipeline.py --input corpus.txt --format sharegpt --rounds 2 --threshold 6

依赖：仅标准库（通过 subprocess 调用本项目内的 eval_loop.py / convert.py / visualize.py）
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

# 调用子脚本所用的 Python 解释器；用 sys.executable 保证与本脚本处于同一环境（含 requests）
PYTHON = sys.executable or "python"

REPORTS_DIR = "reports"
MERGED_REPORTS = os.path.join(REPORTS_DIR, "_merged_reports.jsonl")


def _ensure_utf8_stdout():
    """尽量让 Windows 控制台以 UTF-8 输出，避免中文乱码。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def read_corpora(path):
    """读取语料文件，返回非空语料列表（每行一条，去掉首尾空白）。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        sys.exit(f"错误：输入文件不存在：{path}")
    except OSError as e:
        sys.exit(f"错误：读取输入文件失败：{path} -> {e}")

    corpora = [ln.strip() for ln in lines if ln.strip()]
    if not corpora:
        sys.exit("错误：输入文件为空（没有可评估的语料）。")
    return corpora


def evaluate_corpora(corpora, rounds):
    """逐条调用 eval_loop.py 评估，返回本次成功生成报告的路径列表。"""
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # 复用同一个临时文件承载「当前这一条语料」，评估结束后统一删除
    tmp_path = os.path.join(tempfile.gettempdir(), f"pipeline_corpus_{os.getpid()}.txt")

    report_paths = []
    total = len(corpora)
    try:
        for idx, text in enumerate(corpora, start=1):
            print(f"[{idx}/{total}] 正在评估第 {idx} 条语料...", flush=True)

            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(text)

            report_path = os.path.join(REPORTS_DIR, f"eval_report_{idx:04d}.json")
            cmd = [
                PYTHON, "eval_loop.py",
                "--file", tmp_path,
                "--rounds", str(rounds),
                "-o", report_path,
            ]
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"      警告：第 {idx} 条语料评估失败（eval_loop.py 退出码 {result.returncode}）", flush=True)
            else:
                report_paths.append(report_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return report_paths


def collect_reports(report_paths):
    """读取本次生成的所有评估报告，返回 list[dict]。"""
    reports = []
    for rp in report_paths:
        try:
            with open(rp, "r", encoding="utf-8") as f:
                reports.append(json.load(f))
        except (FileNotFoundError, OSError, json.JSONDecodeError) as e:
            print(f"      警告：读取报告失败 {rp} -> {e}", flush=True)
    return reports


def merge_reports(reports):
    """把多份报告合并为单个 JSONL（每行一个报告），返回合并文件路径。"""
    with open(MERGED_REPORTS, "w", encoding="utf-8") as f:
        for rep in reports:
            f.write(json.dumps(rep, ensure_ascii=False) + "\n")
    return MERGED_REPORTS


def summarize(total, reports, threshold, output_path):
    """打印最终汇总信息（共处理 / 合格 / 不合格 / 输出路径）。"""
    passed = 0
    failed = 0
    invalid = 0
    for rep in reports:
        try:
            score = rep["final"]["overall_score"]
        except (KeyError, TypeError):
            invalid += 1
            continue
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            if score >= threshold:
                passed += 1
            else:
                failed += 1
        else:
            invalid += 1

    print("\n" + "=" * 56, flush=True)
    print("流水线执行完成 · 汇总", flush=True)
    print("=" * 56, flush=True)
    print(f"  共处理 {total} 条语料", flush=True)
    print(f"  合格 {passed} 条（得分 >= {threshold}）", flush=True)
    print(f"  不合格 {failed} 条（得分 < {threshold}）", flush=True)
    if invalid:
        print(f"  （另有 {invalid} 条评分无效/缺失，未计入合格与否）", flush=True)
    print(f"  输出文件：{output_path}", flush=True)


def main():
    _ensure_utf8_stdout()

    parser = argparse.ArgumentParser(description="语料质量评估一键流水线")
    parser.add_argument("--input", required=True, help="输入文件：每行一条语料（纯文本）")
    parser.add_argument("--format", required=True, choices=("alpaca", "sharegpt"),
                        help="输出训练格式：alpaca 或 sharegpt")
    parser.add_argument("--threshold", type=float, default=5.0, help="合格分数阈值（默认 5.0）")
    parser.add_argument("--rounds", type=int, default=1, help="每条语料评估轮数（默认 1）")
    args = parser.parse_args()

    if args.rounds < 1:
        sys.exit("错误：--rounds 必须 >= 1")

    # 1. 读取语料
    corpora = read_corpora(args.input)
    print(f"读取到 {len(corpora)} 条语料。\n", flush=True)

    # 2. 逐条评估
    report_paths = evaluate_corpora(corpora, args.rounds)

    # 3. 收集报告
    reports = collect_reports(report_paths)
    if not reports:
        sys.exit("错误：没有任何成功的评估报告，无法继续转换与统计。")

    # 4. 合并报告
    merged_path = merge_reports(reports)

    # 5. 转换为训练格式
    output_path = f"output_{args.format}.jsonl"
    print(f"\n正在转换为 {args.format} 格式...", flush=True)
    conv = subprocess.run([
        PYTHON, "convert.py",
        "--input", merged_path,
        "--output", output_path,
        "--format", args.format,
        "--threshold", str(args.threshold),
    ])
    if conv.returncode != 0:
        print("警告：convert.py 执行失败。", flush=True)

    # 6. 统计可视化
    print("\n正在生成统计报告...\n", flush=True)
    vis = subprocess.run([
        PYTHON, "visualize.py",
        "--input", merged_path,
        "--threshold", str(args.threshold),
    ])
    if vis.returncode != 0:
        print("警告：visualize.py 执行失败。", flush=True)

    # 7. 汇总
    summarize(len(corpora), reports, args.threshold, output_path)


if __name__ == "__main__":
    main()
