# -*- coding: utf-8 -*-
"""
compare_models.py — 评价生成语料的大模型：DeepSeek / GLM / Qwen 生成质量横向对比

按 prompts/model_comparison.md 的评估设计执行：
    1. 使用同一组种子指令，分别让 deepseek / glm / qwen 各生成一批 指令-响应对
    2. 按模板『输入格式』组装 {"datasets": {deepseek:[...], glm:[...], qwen:[...]}}
    3. 三台大模型各自担任评审（LLM-as-Judge），按模板对三家生成数据
       在 安全性/准确性/多样性/格式规范性 四维打分，输出 model_scores / ranking
    4. 汇总排名，并额外给出「排除自评」口径 —— 某家的生成数据只由
       “另外两家”评审打分（评审团恰好包含被评模型自己，避免自吹自擂偏差）

用法示例：
    python compare_models.py                        # 默认配置跑全流程
    python compare_models.py --per-seed 3           # 每条种子生成 3 对（默认 2）
    python compare_models.py --seeds "种子A|种子B|种子C"
    python compare_models.py --models glm qwen      # 只对比指定生成模型
    python compare_models.py --judges deepseek glm  # 只让指定模型当评审
    python compare_models.py --dry-run              # 不调用 API，预览流程

输出（data/compare/<时间戳>/）：
    gen_deepseek.jsonl / gen_glm.jsonl / gen_qwen.jsonl   三家各自生成的原始数据
    judge_<评审模型>.json                                  每位评审的完整报告
    summary.json                                           汇总投票与两种口径排名
    控制台直接打印对比表格与最终排名

依赖：requests、python-dotenv（pip install requests python-dotenv）
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import llm
import jsonx
import promptio

# 默认种子：与 model_comparison.md 末尾『开始执行』给出的 5 条种子一致
DEFAULT_SEEDS = [
    "将以下英文翻译成中文：Machine learning is a subset of artificial intelligence",
    "用 Python 写一个快速排序算法",
    "撰写一封正式的商务邮件",
    "生成一个常见问题的FAQ",
    "总结这篇文章的主要观点",
]

# 参与对比的生成模型（顺序即表格列顺序）
COMPARE_MODELS = ["deepseek", "glm", "qwen"]

# 生成模型 中文简称（打印/存档用）
DISPLAY_NAME = {
    "deepseek": "DeepSeek",
    "glm": "GLM-5.3",
    "qwen": "Qwen3.6",
}

GEN_SYSTEM = (
    "你是一名高质量数据生成助手。请严格遵循用户要求生成指令-响应对，"
    "只输出指定的 JSON 数据本体，不要 ```json 代码块围栏，不要任何多余文字。"
)

JUDGE_SYSTEM = (
    "你是一名数据质量评估专家（LLM-as-Judge）。请严格遵循用户给出的评审模板执行，"
    "对三个模型生成的数据逐项打分、排名并给出分析，最终只输出模板要求的 JSON，"
    "不要输出任何多余文字。"
)


def _ensure_utf8_stdout():
    """尽量让 Windows 控制台以 UTF-8 输出，避免中文乱码。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


# ============================ 第一步：三家模型生成 ============================

def build_gen_message(seeds, per_seed):
    """构造“同种子 → 每家生成 N 对”的生成提示词，保证三家输入完全一致。"""
    lines = "\n".join(f"{i}. {s}" for i, s in enumerate(seeds, 1))
    user = (
        f"请针对下面每条种子指令，各生成 {per_seed} 个与该种子语义相关的"
        "指令-响应对。\n\n"
        "要求：\n"
        "- instruction 为用户指令，response 为模型的完整回答；\n"
        "- 回答准确、完整、具体，代码类指令给出可直接运行的代码；\n"
        "- 全部用中文撰写（代码、专有名词除外）；内容安全合规，不生成敏感信息；\n"
        "- 三家模型使用完全相同的种子与数量，保证对比公平。\n\n"
        "种子指令：\n"
        f"{lines}\n\n"
        '只输出如下结构的 JSON（不要任何其他内容）：\n'
        '{"items": [{"instruction": "...", "response": "..."}, ...]}'
    )
    return [
        {"role": "system", "content": GEN_SYSTEM},
        {"role": "user", "content": user},
    ]


def generate_items(provider, seeds, per_seed, max_tokens):
    """让 provider 模型按种子生成数据对；返回 items 列表（失败抛异常）。"""
    messages = build_gen_message(seeds, per_seed)
    raw = llm.call_chat(provider, messages, temperature=0.7, max_tokens=max_tokens)
    obj = jsonx.extract_json(raw)
    if obj is None:
        raise ValueError(f"生成输出未解析出合法 JSON。片段：{raw[:200]}...")
    if isinstance(obj, dict) and isinstance(obj.get("items"), list):
        return [it for it in obj["items"]
                if isinstance(it, dict) and it.get("instruction") and it.get("response")]
    # 模型未按 items 包装时兜底：递归找所有指令-响应对
    pairs = jsonx.walk_pairs(obj)
    return [{"instruction": p["instruction"], "response": p["response"]} for p in pairs]


# ============================ 第二步：模板组装 + 评审 ============================

def build_judge_message(template, datasets):
    """把三家生成数据按模板『输入格式』组装成一次评审请求。"""
    data_json = json.dumps({"datasets": datasets}, ensure_ascii=False, indent=1)
    user = template.rstrip() + (
        "\n\n[待评估数据]\n"
        "以下为三个模型在相同种子、相同数量条件下生成的真实数据，"
        "请直接以上述数据集为输入，按模板『评估流程』与『输出要求』"
        "完成打分、排名与分析，只输出最终 JSON：\n"
        f"{data_json}"
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]


def parse_judge_report(raw, models):
    """解析评审输出，返回 {"totals": {生成模型: 综合分}, "ranking": ...}；解析失败返回 None。"""
    obj = jsonx.extract_json(raw)
    if obj is None:
        return None
    node = jsonx.deep_find(obj, "model_scores")
    totals = {}
    if isinstance(node, dict):
        ms = node["model_scores"]
        if isinstance(ms, dict):
            for m in models:
                d = ms.get(m)
                if not isinstance(d, dict):
                    continue
                t = d.get("total")
                if not isinstance(t, (int, float)):
                    nums = [d.get(k) for k in ("safety", "accuracy", "diversity", "format")]
                    nums = [x for x in nums if isinstance(x, (int, float))]
                    t = sum(nums) / len(nums) if nums else None
                totals[m] = round(float(t), 2) if isinstance(t, (int, float)) else None
    if not totals:
        return None
    ranking = jsonx.deep_find(obj, "ranking")
    return {"totals": totals, "ranking": ranking, "obj": obj}


# ============================ 第三步：汇总 ============================

def _mean(vals):
    """求均值，忽略 None / 空列表；全空返回 None。"""
    nums = [v for v in vals if v is not None]
    return round(sum(nums) / len(nums), 2) if nums else None


def main():
    _ensure_utf8_stdout()
    parser = argparse.ArgumentParser(
        description="评价生成语料的大模型：DeepSeek / GLM / Qwen 生成质量横向对比")
    parser.add_argument("--seeds", default=None,
                        help="种子指令，多个用 | 分隔；不给则用模板自带的 5 条默认种子")
    parser.add_argument("--models", nargs="+", choices=COMPARE_MODELS,
                        default=COMPARE_MODELS, help="参与对比的生成模型")
    parser.add_argument("--judges", nargs="+", choices=COMPARE_MODELS,
                        default=COMPARE_MODELS, help="担任评审的模型（默认三者互评）")
    parser.add_argument("--per-seed", type=int, default=2,
                        help="每家生成模型对每条种子的生成对数（默认 2，共 种子数×2 条）")
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="单次调用最大 token 数（默认 8192）")
    parser.add_argument("--output-dir",
                        default=str(Path(__file__).resolve().parent / "data" / "compare"),
                        help="输出目录（默认 data/compare）")
    parser.add_argument("--dry-run", action="store_true",
                        help="不调用 API，仅预览流程与各步骤提示词大小")
    args = parser.parse_args()

    seeds = [s.strip() for s in args.seeds.split("|") if s.strip()] if args.seeds \
        else DEFAULT_SEEDS
    # 与模板保持一致：生成模型缺 Key 自动跳过（但评审判定仍覆盖全部）
    gen_models, _missing = llm.check_available(args.models)
    judge_models, _missing2 = llm.check_available(args.judges)
    if args.dry_run:
        gen_models, judge_models = args.models, args.judges
    if not gen_models or not judge_models:
        sys.exit("错误：生成模型或评审模型均不可用，请先配置 .env（参考 .env.example）。")
    if not set(gen_models) & set(judge_models):
        print("⚠ 评审模型与生成模型完全无交集：无法计算『排除自评』口径，将只输出全量投票。")

    per_model_count = args.per_seed * len(seeds)
    print(f"种子 {len(seeds)} 条 | 每家生成 {per_model_count} 对 | "
          f"生成：{', '.join(gen_models)} | 评审：{', '.join(judge_models)}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 生成阶段 ----------
    datasets = {}
    for m in gen_models:
        if args.dry_run:
            msgs = build_gen_message(seeds, args.per_seed)
            print(f"[dry-run] {m} 生成提示词 {len(msgs[1]['content'])} 字符")
            datasets[m] = []
            continue
        try:
            items = generate_items(m, seeds, args.per_seed, args.max_tokens)
        except Exception as e:
            print(f"[生成 {m}] ✗ 失败：{e}")
            continue
        datasets[m] = items
        with open(out_dir / f"gen_{m}.jsonl", "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"[生成 {m}] 成功 {len(items)} 条 → {out_dir / f'gen_{m}.jsonl'}")

    if not datasets:
        sys.exit("错误：所有生成模型均失败，无法对比。")
    print(f"生成完成：{ {k: len(v) for k, v in datasets.items()} }")

    # ---------- 评审阶段 ----------
    template = promptio.load_prompt("model_comparison")
    votes = {}  # {评审模型: {生成模型: 综合分}}
    judge_reports = {}
    for j in judge_models:
        msgs = build_judge_message(template, datasets)
        if args.dry_run:
            print(f"[dry-run] 评审 {j} 提示词 {len(msgs[1]['content'])} 字符")
            continue
        try:
            raw = llm.call_chat(j, msgs, temperature=0.2, max_tokens=args.max_tokens)
        except Exception as e:
            print(f"[评审 {j}] ✗ 失败：{e}")
            continue
        report = parse_judge_report(raw, COMPARE_MODELS)
        if report is None:
            print(f"[评审 {j}] ✗ 输出解析失败，原文已存盘供人工查看")
            (out_dir / f"judge_{j}_raw.txt").write_text(raw, encoding="utf-8")
            continue
        votes[j] = report["totals"]
        judge_reports[j] = report
        with open(out_dir / f"judge_{j}.json", "w", encoding="utf-8") as f:
            json.dump({"judge": j, "report": report["obj"]}, f,
                      ensure_ascii=False, indent=1)
        print(f"[评审 {j}] 完成 → {out_dir / f'judge_{j}.json'}")

    if args.dry_run:
        print("dry-run 完成：未真正调用任何 API。")
        return

    if not votes:
        sys.exit("错误：所有评审均失败，无结果。")
    print(f"评审完成：{ {k: len(v) for k, v in votes.items()} }")

    # ---------- 汇总与打印 ----------
    judge_keys = list(votes.keys())
    lines = [["生成模型"] + [DISPLAY_NAME[j] for j in judge_keys]
             + ["自评均分", "非自评均分*"]]
    ranking_data = []  # (生成模型, 非自评均分, 自评均分)
    for g in gen_models:
        row_votes = [votes[j].get(g) for j in judge_keys]
        self_vote = votes.get(g, {}).get(g) if g in votes else None
        non_self = [votes[j].get(g) for j in judge_keys if j != g]
        non_self_mean = _mean(non_self)
        self_mean = _mean([v for v in row_votes if v is not None])
        cells = [f"{v:.2f}" if v is not None else "—" for v in row_votes]
        cells += [f"{self_mean:.2f}" if self_mean is not None else "—",
                  f"{non_self_mean:.2f}" if non_self_mean is not None else "—"]
        lines.append([DISPLAY_NAME[g]] + cells)
        ranking_data.append((g, non_self_mean, self_mean))

    # ASCII 表格
    widths = [max(len(lines[r][c]) for r in range(len(lines)))
              for c in range(len(lines[0]))]
    print("\n================ 生成模型质量对比（LLM-as-Judge） ================")
    for r, row in enumerate(lines):
        print("  " + " | ".join(
            cell.ljust(widths[c]) for c, cell in enumerate(row)))
        if r == 0:
            print("  " + "-+-".join("-" * w for w in widths))
    print("\n排名（按非自评均分* 降序）：")
    for rank, (g, nsm, sm) in enumerate(
            sorted(ranking_data, key=lambda x: (x[1] if x[1] is not None else -1),
                   reverse=True), 1):
        print(f"  {rank}. {DISPLAY_NAME[g]}\t非自评 {nsm}\t（自评口径 {sm}）")

    best = max(ranking_data,
               key=lambda x: (x[1] if x[1] is not None else -1, x[2] if x[2] is not None else -1))
    worst = min(ranking_data, key=lambda x: (x[1] if x[1] is not None else 11))
    print(f"\n结论：{DISPLAY_NAME[best[0]]} 生成数据质量综合最高；"
          f"{DISPLAY_NAME[worst[0]]} 相对较弱。"
          f"详细单条理由见各 judge_*.json 的 details 字段。")
    print("* 非自评 = 排除该生成模型自己当评审时的打分（评审团包含其本人，避免自吹自擂偏差）")
    print(f"完整报告目录：{out_dir}")

    # 存档汇总
    summary = {
        "seeds": seeds,
        "per_seed": args.per_seed,
        "generated_counts": {k: len(v) for k, v in datasets.items()},
        "judges": judge_keys,
        "votes": {j: votes[j] for j in judge_keys},
        "ranking": [
            {"model": g, "non_self_mean": nsm, "self_mean": sm}
            for g, nsm, sm in sorted(
                ranking_data,
                key=lambda x: (x[1] if x[1] is not None else -1), reverse=True)
        ],
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
