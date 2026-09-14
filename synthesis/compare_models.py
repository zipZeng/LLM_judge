# -*- coding: utf-8 -*-
"""
compare_models.py — 评价生成语料的大模型：多模型生成质量横向对比

按 prompts/model_comparison.md 的评估设计执行：
    1. 使用同一组种子指令，分别让各家生成模型各生成一批 指令-响应对
    2. 按模板『输入格式』组装 {"datasets": {各生成模型:[...]}}
    3. 各模型各自担任评审（LLM-as-Judge），按模板对各家的生成数据
       在 安全性/准确性/多样性/格式规范性 四维打分，输出 model_scores / ranking
    4. 汇总排名，并额外给出「排除自评」口径 —— 某家的生成数据只由
       “另外几家”评审打分（评审团恰好包含被评模型自己，避免自吹自擂偏差）

用法示例：
    python compare_models.py                        # 默认：2 条种子 × 每家 3 对
    python compare_models.py --seeds 3 --per-seed 2 # 3 条种子，每条 2 对
    python compare_models.py --models <模型键1> <模型键2>  # 只对比指定生成模型
    python compare_models.py --judges <模型键1> <模型键2>  # 只让指定模型当评审
    python compare_models.py --dry-run              # 不调用 API，预览流程

    python compare_models.py --from-report data/compare/20260911_104142
        # 不调用 API，重放已存档的对比结果，秒级打印同一张表。
        # 现场演示用：真实跑一次要 25–45 分钟（推理模型单次约 250 秒），等不起。

输出（data/compare/<时间戳>/）：
    gen_<模型键>.jsonl                                     各家各自生成的原始数据
    judge_<评审模型>.json                                  每位评审的完整报告
    summary.json                                           汇总投票与两种口径排名
    控制台直接打印对比表格与最终排名

`--dry-run` 与 `--from-report` 都不需要 API Key，也不写任何文件。
`--from-report` 与正常跑**共用同一段渲染代码**（`render_report`），
故重放出来的表与现跑出来的完全一致，不是另写一套。

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

# 参与对比的生成模型（顺序即表格列顺序）—— 从 models.json 动态取，不写死具体模型名
COMPARE_MODELS = list(llm.PROVIDERS.keys())

# 生成模型显示名（打印/存档用）—— 直接复用 llm.PROVIDERS 的 label
DISPLAY_NAME = {k: p["label"] for k, p in llm.PROVIDERS.items()}

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


def generate_items(provider, seeds, per_seed, max_tokens, timeout=1800):
    """让 provider 模型按种子生成数据对；返回 items 列表（失败抛异常）。"""
    messages = build_gen_message(seeds, per_seed)
    raw = llm.call_chat(provider, messages, temperature=0.7,
                        max_tokens=max_tokens, timeout=timeout)
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


def render_report(gen_models, judge_keys, votes, source=None):
    """把 votes 渲染成对比表 + 排名 + 两项方法学检验，返回 ranking_data。

    **现跑与 `--from-report` 重放共用这一段** —— 两条路径共用才能保证
    「重放出来的表」和「现跑出来的表」是同一套口径。抽出来的直接原因：
    答辩现场跑一次真实对比要 25–45 分钟（推理模型单次约 250 秒），等不起；
    用存档 summary.json 重放是秒级的。
    """
    lines = [["生成模型"] + [DISPLAY_NAME.get(j, j) for j in judge_keys]
             + ["全量均分", "非自评均分*"]]
    ranking_data = []
    for g in gen_models:
        row_votes = [votes[j].get(g) for j in judge_keys]
        non_self = [votes[j].get(g) for j in judge_keys if j != g]
        non_self_mean = _mean(non_self)
        # ⚠ 这一列是全量均分（含自评），不是自评分。原表头写「自评均分」是错的：
        # 09-11 那次 DeepSeek 全量 8.67，但它给自己打的是 8.25。
        all_mean = _mean([v for v in row_votes if v is not None])
        self_score = votes[g].get(g) if g in votes else None
        cells = [f"{v:.2f}" if v is not None else "—" for v in row_votes]
        cells += [f"{all_mean:.2f}" if all_mean is not None else "—",
                  f"{non_self_mean:.2f}" if non_self_mean is not None else "—"]
        lines.append([DISPLAY_NAME.get(g, g)] + cells)
        ranking_data.append((g, non_self_mean, all_mean, self_score))

    # ASCII 表格（宽度按各列最大宽度对齐）
    widths = [max(len(lines[r][c]) for r in range(len(lines)))
              for c in range(len(lines[0]))]
    print("\n================ 生成模型质量对比（LLM-as-Judge） ================")
    for r, row in enumerate(lines):
        print("  " + " | ".join(
            cell.ljust(widths[c]) for c, cell in enumerate(row)))
        if r == 0:
            print("  " + "-+-".join("-" * w for w in widths))

    print("\n排名（按非自评均分* 降序）：")
    for rank, (g, nsm, allm, ss) in enumerate(
            sorted(ranking_data, key=lambda x: (x[1] if x[1] is not None else -1),
                   reverse=True), 1):
        print(f"  {rank}. {DISPLAY_NAME.get(g, g)}\t非自评 {nsm}\t（全量口径 {allm}）")

    if len(gen_models) >= 2:
        # 这两项是需求「评价生成语料的大模型」的核心方法学检验，现跑与重放都要有：
        # 评委尺度决定分数能不能跨评审横向比；自评偏好决定要不要排除自评。
        print("\n评委尺度（该评委给出的均分；越高越宽松。差距大 ⇒ 跨评审比绝对分不可靠）：")
        for j in judge_keys:
            vals = [votes[j].get(g) for g in gen_models if votes[j].get(g) is not None]
            print(f"  {DISPLAY_NAME.get(j, j)}\t{_mean(vals)}")
        # 自评偏好 = 该模型给自己的分 − 别人给它的均分。
        # ⚠ 不能用「全量均分 − 非自评均分」：全量均分本身就含自评，减出来只有真值的一半左右。
        # 实测 09-11 DeepSeek 真值 −0.63（8.25 − 8.88），错法算出 −0.21。
        print("\n自评偏好（自评分 − 他人均分；正 = 给自己打高分）：")
        for g, nsm, allm, ss in ranking_data:
            d = "—" if (ss is None or nsm is None) else f"{ss - nsm:+.2f}"
            extra = "" if ss is None else f"（自评 {ss}）"
            print(f"  {DISPLAY_NAME.get(g, g)}\t{d}{extra}")

        best = max(ranking_data,
                   key=lambda x: (x[1] if x[1] is not None else -1,
                                  x[2] if x[2] is not None else -1))
        worst = min(ranking_data, key=lambda x: (x[1] if x[1] is not None else 11))
        print(f"\n结论：{DISPLAY_NAME.get(best[0], best[0])} 生成数据质量综合最高；"
              f"{DISPLAY_NAME.get(worst[0], worst[0])} 相对较弱。")
    print("单条理由见各 judge_*.json 的 details 字段。")
    print("* 非自评 = 排除该生成模型自己当评审时的打分（评审团包含其本人，避免自吹自擂偏差）")
    print(f"报告目录：{source}")
    return ranking_data


def load_summary(path):
    """读存档的 summary.json；也接受 `data/compare/<时间戳>/` 目录。"""
    p = Path(path)
    if p.is_dir():
        p = p / "summary.json"
    if not p.is_file():
        sys.exit(f"错误：找不到 {p}。--from-report 需要 summary.json 本身或其所在目录。")
    with open(p, encoding="utf-8") as f:
        return json.load(f), p


def main():
    _ensure_utf8_stdout()
    parser = argparse.ArgumentParser(
        description="评价生成语料的大模型：多模型生成质量横向对比")
    parser.add_argument("--seeds", type=int, default=2,
                        help="种子数量（默认 2）：取模板自带的默认种子前 N 条")
    parser.add_argument("--models", nargs="+", choices=COMPARE_MODELS,
                        default=COMPARE_MODELS, help="参与对比的生成模型")
    parser.add_argument("--judges", nargs="+", choices=COMPARE_MODELS,
                        default=COMPARE_MODELS, help="担任评审的模型（默认三者互评）")
    parser.add_argument("--per-seed", type=int, default=3,
                        help="每家生成模型对每条种子的生成对数（默认 3，共 种子数×3 条）")
    parser.add_argument("--max-tokens", type=int, default=16384,
                        help="单次调用最大 token 数（默认 16384；推理模型的思维链也占用该预算）")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="单次调用读超时秒数（默认 1800；推理模型生成大 JSON 较慢，"
                             "实测 Kimi/Qwen 单次约 250s，勿低于 300）")
    parser.add_argument("--output-dir",
                        default=str(Path(__file__).resolve().parent / "data" / "compare"),
                        help="输出目录（默认 data/compare）")
    parser.add_argument("--dry-run", action="store_true",
                        help="不调用 API，仅预览流程与各步骤提示词大小")
    parser.add_argument("--from-report", metavar="路径", default=None,
                        help="不调用 API，直接重放已存档的对比结果并打印同一张表。"
                             "传 summary.json 本身或 data/compare/<时间戳>/ 目录。"
                             "现场演示用：真实跑一次要 25–45 分钟，重放是秒级。")
    args = parser.parse_args()

    # ---------- 只读重放：必须在任何 Key 检查之前返回 ----------
    # 演示机器可能没配 .env，重放不该因为缺 Key 而失败。
    if args.from_report:
        data, path = load_summary(args.from_report)
        votes = data.get("votes") or {}
        if not votes:
            sys.exit(f"错误：{path} 里没有 votes 字段，无法重放（该存档可能是失败运行）。")
        judge_keys = data.get("judges") or list(votes.keys())
        # 行序优先用 generated_counts（即当年生成阶段的模型顺序）；旧存档缺该字段时
        # 回退到从 votes 里收集，并丢掉「一条票都没有」的模型，避免打出空行。
        gen_models = list((data.get("generated_counts") or {}).keys()) \
            or list({g for v in votes.values() for g in v})
        gen_models = [g for g in gen_models
                      if any(g in votes.get(j, {}) for j in judge_keys)]
        if not gen_models:
            sys.exit(f"错误：{path} 里没有任何生成模型拿到评分，无法重放。")
        n_seeds, per_seed = len(data.get("seeds") or []), data.get("per_seed")
        print(f"重放存档：{path}")
        if n_seeds and per_seed:
            print(f"种子 {n_seeds} 条 | 每家生成 {n_seeds * per_seed} 对 | "
                  f"评审：{', '.join(DISPLAY_NAME.get(j, j) for j in judge_keys)}")
        render_report(gen_models, judge_keys, votes, source=path.parent)
        print(f"（--from-report 为只读重放：未调用任何 API，未写入任何文件）")
        return

    if args.seeds < 1:
        sys.exit("错误：--seeds 必须 >= 1")
    if args.seeds > len(DEFAULT_SEEDS):
        print(f"⚠ 种子数 {args.seeds} 超过模板自带默认种子 {len(DEFAULT_SEEDS)} 条，"
              f"按 {len(DEFAULT_SEEDS)} 条执行。")
    seeds = DEFAULT_SEEDS[:args.seeds]
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

    # 进度计数：生成 + 评审的全部调用，用来显示"第几次/共几次"。
    # 推理模型单次 4–7.5 分钟，没有进度的话现场分不清在跑还是卡死。
    n_calls = 0
    planned_calls = len(gen_models) + len(judge_models)

    # ---------- 生成阶段 ----------
    datasets = {}
    for m in gen_models:
        if args.dry_run:
            msgs = build_gen_message(seeds, args.per_seed)
            print(f"[dry-run] {m} 生成提示词 {len(msgs[1]['content'])} 字符")
            datasets[m] = []
            continue
        n_calls += 1
        try:
            with llm.long_call(f"[{n_calls}/{planned_calls}] [生成 {m}]"):
                items = generate_items(m, seeds, args.per_seed, args.max_tokens,
                                       args.timeout)
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
        n_calls += 1
        try:
            with llm.long_call(f"[{n_calls}/{planned_calls}] [评审 {j}]"):
                raw = llm.call_chat(j, msgs, temperature=0.2,
                                    max_tokens=args.max_tokens, timeout=args.timeout)
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
    # 与 --from-report 共用 render_report，保证「现跑的」和「重放的」是同一套口径
    judge_keys = list(votes.keys())
    ranking_data = render_report(gen_models, judge_keys, votes, source=out_dir)

    # 存档汇总
    summary = {
        "seeds": seeds,
        "per_seed": args.per_seed,
        "generated_counts": {k: len(v) for k, v in datasets.items()},
        "judges": judge_keys,
        "votes": {j: votes[j] for j in judge_keys},
        "ranking": [
            {"model": g, "non_self_mean": nsm, "all_mean": allm, "self_score": ss}
            for g, nsm, allm, ss in sorted(
                ranking_data,
                key=lambda x: (x[1] if x[1] is not None else -1), reverse=True)
        ],
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
