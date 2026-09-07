# -*- coding: utf-8 -*-
"""
visualize.py — 评估结果可视化统计（纯 print + ASCII 柱状图，不依赖图形界面库）

功能：
    对 eval_loop.py 输出的评估报告进行统计分析，输出：
        1. 总语料数量、合格/不合格数量及占比（阈值可配置，默认 5 分）
        2. 各维度平均分（安全性/准确性/多样性/格式），并用 ASCII 柱状图展示
        3. 各维度分数分布（0-3分 / 4-6分 / 7-10分 各占比）
        4. 各模型打分一致性（平均分、标准差、与其他模型的平均绝对差异）

输入：
    一个或多个评估报告文件；支持单个 JSON、JSON 数组、JSONL（每行一个对象），
    并支持通配符（如 *.jsonl）。多个文件用空格分隔传入 --input。

输入报告结构（eval_loop.py 输出）：
    {
      "meta":  {"text": "...", "models": ["Ollama(...)", "DeepSeek(...)", ...], ...},
      "rounds": [{"round": 1, "dimensions": {"safety": {"Ollama(...)": {"score": 10, "reason": "..."}, ...}, ...}, ...],
      "final": {"dimension_scores": {"safety": 10.0, ...}, "overall_score": 8.0, "conclusion": "合格", ...}
    }

用法示例：
    python visualize.py --input eval_report_20260907_235928.json
    python visualize.py --input eval_report1.json eval_report2.jsonl --threshold 5
    python visualize.py --input "reports/*.jsonl" --threshold 6

依赖：仅标准库（Python 3.8+）
"""

import argparse
import glob
import json
import statistics
import sys

# ============================ 配置区 ============================

DEFAULT_THRESHOLD = 5.0

# 评估维度及其中文显示名（顺序即展示顺序）
DIMENSIONS = ["safety", "accuracy", "diversity", "format"]
DIM_LABELS = {"safety": "安全性", "accuracy": "准确性", "diversity": "多样性", "format": "格式"}

# 分数分布分桶：0-3分 / 4-6分 / 7-10分（左闭右开）
BUCKET_LABELS = ["0-3分", "4-6分", "7-10分"]


# ============================ 基础工具函数 ============================


def _ensure_utf8_stdout():
    """尽量让 Windows 控制台以 UTF-8 输出，避免中文乱码。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def _is_number(v):
    """判断是否为可参与统计的数字（int/float，且非 bool）。"""
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _expand_inputs(patterns):
    """展开输入路径：支持通配符，返回去重后的文件列表（保持顺序）。"""
    files = []
    seen = set()
    for p in patterns:
        matches = glob.glob(p)
        if matches:
            for m in matches:
                if m not in seen:
                    seen.add(m)
                    files.append(m)
        else:
            # 无匹配：区分「明确不存在」与「通配符无匹配」
            if any(c in p for c in "*?["):
                print(f"警告：通配符无匹配：{p}", file=sys.stderr)
            else:
                print(f"警告：文件不存在：{p}", file=sys.stderr)
    return files


def _read_reports(path):
    """读取单个报告文件，返回 list[dict]。

    依次尝试：单个 JSON 对象、JSON 数组、JSONL（每行一个对象）。
    文件不存在或无法解析时给出友好提示并退出。
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        sys.exit(f"错误：输入文件不存在：{path}")
    except OSError as e:
        sys.exit(f"错误：读取输入文件失败：{path} -> {e}")

    if not content.strip():
        print(f"警告：文件为空，跳过：{path}", file=sys.stderr)
        return []

    # 1) 单个 JSON（对象或数组）
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = None

    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        return [data]

    # 2) JSONL 逐行解析
    reports = []
    for i, line in enumerate(content.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            sys.exit(f"错误：{path} 第 {i} 行不是合法 JSON：{e}")
        if isinstance(obj, dict):
            reports.append(obj)
    return reports


def _bucket(score):
    """把分数映射到分布分桶索引：0→0-3分，1→4-6分，2→7-10分。"""
    if score < 4.0:
        return 0
    if score < 7.0:
        return 1
    return 2


def _bar(value, width=10):
    """根据 0-10 分生成 ASCII 柱状图（每个 █ 代表 1 分，四舍五入）。"""
    if value is None:
        return "·" * width
    filled = int(round(min(max(value, 0.0), 10.0)))
    return "█" * filled + "·" * (width - filled)


def _collect_stats(reports, threshold):
    """遍历所有报告，汇总各项统计指标，返回 stats dict。"""
    total = len(reports)
    passed = 0
    failed = 0
    invalid = 0  # overall_score 缺失或非数字

    dim_sums = {}    # dim -> 累加和
    dim_counts = {}  # dim -> 数量
    dim_buckets = {}  # dim -> [c0, c1, c2]

    model_scores = {}  # model -> [score, ...]
    model_diffs = {}   # model -> [|score - 其他模型均值|, ...]

    dim_names = list(DIMENSIONS)  # 展示顺序：先按标准维度，再补其余

    for rep in reports:
        final = rep.get("final") or {}
        overall = final.get("overall_score")

        # 合格/不合格
        if not _is_number(overall):
            invalid += 1
        elif overall >= threshold:
            passed += 1
        else:
            failed += 1

        # 维度平均分与分布
        dim_scores = final.get("dimension_scores") or {}
        for d, v in dim_scores.items():
            if not _is_number(v):
                continue
            if d not in dim_names:
                dim_names.append(d)
            dim_sums[d] = dim_sums.get(d, 0.0) + v
            dim_counts[d] = dim_counts.get(d, 0) + 1
            dim_buckets.setdefault(d, [0, 0, 0])[_bucket(v)] += 1

        # 模型一致性：遍历每一轮的每一维度
        rounds = rep.get("rounds") or []
        for rnd in rounds:
            if not isinstance(rnd, dict):
                continue
            dims = rnd.get("dimensions") or {}
            for d, models in dims.items():
                if not isinstance(models, dict):
                    continue
                # 该维度下本模型 -> 有效分数
                valid = {
                    m: s.get("score")
                    for m, s in models.items()
                    if isinstance(s, dict) and _is_number(s.get("score"))
                }
                for m, s in valid.items():
                    model_scores.setdefault(m, []).append(s)
                    others = [v for k, v in valid.items() if k != m]
                    if others:
                        other_mean = sum(others) / len(others)
                        model_diffs.setdefault(m, []).append(abs(s - other_mean))

    # 维度平均分
    dim_avg = {
        d: (dim_sums[d] / dim_counts[d] if dim_counts.get(d) else None)
        for d in dim_names
    }

    # 模型统计
    model_stats = {}
    for m, scores in model_scores.items():
        mean = sum(scores) / len(scores) if scores else None
        std = statistics.stdev(scores) if len(scores) >= 2 else 0.0
        diffs = model_diffs.get(m)
        mean_diff = sum(diffs) / len(diffs) if diffs else None
        model_stats[m] = {
            "count": len(scores),
            "mean": mean,
            "std": std,
            "mean_diff": mean_diff,
        }

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "invalid": invalid,
        "dim_names": dim_names,
        "dim_avg": dim_avg,
        "dim_buckets": dim_buckets,
        "dim_counts": dim_counts,
        "model_stats": model_stats,
    }


# ============================ 输出函数 ============================


def _print_header(stats, threshold):
    total = stats["total"]
    passed = stats["passed"]
    failed = stats["failed"]
    invalid = stats["invalid"]

    print("=" * 56)
    print("语料质量评估 · 统计分析报告")
    print("=" * 56)

    print(f"  总语料数量        : {total}")
    if total:
        print(f"  合格（>= {threshold}）: {passed}  （{passed / total * 100:.1f}%）")
        print(f"  不合格            : {failed}  （{failed / total * 100:.1f}%）")
        if invalid:
            print(f"  分数缺失/异常     : {invalid}")
    else:
        print("  （无有效数据）")
    print()


def _print_dim_avg(stats):
    print("-" * 56)
    print("各维度平均分（ASCII 柱状图，每个 █ 代表 1 分）：")
    print()
    for d in stats["dim_names"]:
        label = DIM_LABELS.get(d, d)
        avg = stats["dim_avg"][d]
        bar = _bar(avg)
        score_str = f"{avg:.2f}" if avg is not None else "N/A"
        print(f"  {label}  {bar}  {score_str}")
    print()


def _print_dim_dist(stats):
    print("-" * 56)
    print(f"各维度分数分布（{' / '.join(BUCKET_LABELS)}）：")
    print()
    for d in stats["dim_names"]:
        label = DIM_LABELS.get(d, d)
        counts = stats["dim_buckets"].get(d, [0, 0, 0])
        n = stats["dim_counts"].get(d, 0)
        if n == 0:
            pct = "  -   /  -   /  -"
        else:
            pct = "  /  ".join(f"{c / n * 100:5.1f}%" for c in counts)
        print(f"  {label}  :  {pct}")
    print()


def _print_model_consistency(stats):
    print("-" * 56)
    print("各模型打分一致性：")
    print()
    model_stats = stats["model_stats"]
    if not model_stats:
        print("  （rounds 数据不完整，无法计算模型一致性）")
        print()
        return

    header = f"  {'模型':<28}{'样本':>5}{'平均分':>8}{'标准差':>8}{'与它模型均差':>10}"
    print(header)
    print("  " + "-" * 54)
    for m, s in model_stats.items():
        mean = f"{s['mean']:.2f}" if s["mean"] is not None else "N/A"
        std = f"{s['std']:.2f}" if s["std"] is not None else "N/A"
        md = f"{s['mean_diff']:.2f}" if s["mean_diff"] is not None else "N/A"
        print(f"  {m:<28}{s['count']:>5}{mean:>8}{std:>8}{md:>10}")
    print()


# ============================ 主流程 ============================


def main():
    _ensure_utf8_stdout()

    parser = argparse.ArgumentParser(description="评估结果可视化统计")
    parser.add_argument("-i", "--input", nargs="+", required=True,
                        help="一个或多个评估报告文件（支持通配符，如 *.jsonl）")
    parser.add_argument("-t", "--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"合格分数阈值（默认 {DEFAULT_THRESHOLD}）")
    args = parser.parse_args()

    # 展开输入路径（支持通配符）
    files = _expand_inputs(args.input)
    if not files:
        sys.exit("错误：没有找到任何可读取的输入文件。")

    # 读取所有报告
    reports = []
    for fp in files:
        reports.extend(_read_reports(fp))

    if not reports:
        sys.exit("错误：所有输入文件中均未解析到有效报告。")

    print(f"已读取 {len(files)} 个文件、共 {len(reports)} 条报告。\n", flush=True)

    stats = _collect_stats(reports, args.threshold)
    _print_header(stats, args.threshold)
    _print_dim_avg(stats)
    _print_dim_dist(stats)
    _print_model_consistency(stats)


if __name__ == "__main__":
    main()
