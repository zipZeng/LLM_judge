# -*- coding: utf-8 -*-
"""
generate.py — 合成数据生成器（Self-Instruct / Evol-Instruct / Magpie / 种子改写增强）

读取 prompts/ 目录下对应范式的提示词模板，调用大模型生成合成指令数据，
把模型输出的 JSON 规整为统一的 JSONL 数据集，每行一条：
    {"paradigm": "self_instruct", "category": "...", "source_model": "deepseek",
     "instruction": "...", "response": "..."}

用法示例：
    # 四个范式 × 所有已配置 Key 的模型
    python generate.py

    # 只跑指定范式 + 指定模型
    python generate.py --paradigm self_instruct magpie --models deepseek glm

    # 追加自己的种子指令（多个用 | 分隔）
    python generate.py --paradigm magpie --seeds "帮我写一封请假邮件|总结一篇论文的核心观点"

    # 不调用 API，只预览将发送的提示词
    python generate.py --dry-run

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

PARADIGMS_ALL = ["self_instruct", "evol_instruct", "magpie", "seed_rewrite"]
MODELS_ALL = list(llm.PROVIDERS.keys())

# 解析体检阈值：平均每个数据对占多少字符算"不合常理"。
# 正常数据对（指令 + 回答）约 200-400 字符；实测 magpie 一次返回 11262 字符 / 32 对 = 352。
# 超过该阈值说明返回体量远大于解析出的数据量，多半是漏解析（见 jsonx.walk_pairs 注释）。
SUSPICIOUS_CHARS_PER_PAIR = 800

# 各范式无 --seeds 时的兜底种子：
# evol_instruct / magpie 模板末尾自带默认种子（模板内执行），无需再给；
# self_instruct 模板未给出明确默认种子，这里补一组通用种子供模型按第一步扩展。
DEFAULT_SEEDS_SELF_INSTRUCT = [
    "写一封申请调休的请假邮件",
    "把一段中文翻译成英文",
    "用 Python 实现冒泡排序并说明原理",
    "为一篇科技新闻写 100 字以内的摘要",
    "根据商品名称与价格生成一张购物清单",
]

SYSTEM_GEN = (
    "你是一名专业的大模型合成数据工程师。请严格遵循用户给出的提示词模板执行："
    "模板要求分步时按步骤完成，最终只输出模板『输出要求 / 输出格式』规定的完整 JSON"
    "数据本体——不要 ```json 代码块围栏，不要任何解释、评论或多余文字。"
)


def _ensure_utf8_stdout():
    """尽量让 Windows 控制台以 UTF-8 输出，避免中文乱码。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def build_messages(paradigm, template, seeds):
    """组装发送给模型的 system + user 消息。

    种子指令只作“补充说明”附在模板之后；若用户未给种子且模板自带
    （evol/magpie 的『开始执行』节有默认种子），则原样使用模板内容。
    """
    user = template
    if seeds:
        user += "\n\n[本次执行的补充说明]\n"
        user += "下面额外给定本次要用的种子指令。若与模板内示例种子不同，" \
                "一律以本补充为准，从模板第一步开始处理这些种子：\n"
        for i, s in enumerate(seeds, 1):
            user += f"{i}. {s}\n"
    user += "\n\n请只输出最终 JSON 数据本体。"
    return [
        {"role": "system", "content": SYSTEM_GEN},
        {"role": "user", "content": user},
    ]


def generate_one(provider, paradigm, template, seeds, max_tokens, timeout=600):
    """调用单个模型生成一轮合成数据。

    返回 (pairs, raw)：pairs 为展平后的数据对列表，raw 为模型原始输出。
    解析失败时抛 ValueError，由调用方记入错误日志。
    """
    messages = build_messages(paradigm, template, seeds)
    raw = llm.call_chat(provider, messages, temperature=0.8,
                        max_tokens=max_tokens, timeout=timeout)
    obj = jsonx.extract_json(raw)
    if obj is None:
        raise ValueError(f"输出中未解析出合法 JSON。片段：{raw[:200]}...")
    pairs = jsonx.walk_pairs(obj)
    if not pairs:
        raise ValueError(f"JSON 中未找到 instruction/response 数据对。片段：{raw[:200]}...")
    return pairs, raw


def main():
    _ensure_utf8_stdout()
    parser = argparse.ArgumentParser(
        description="合成数据生成器：Self-Instruct / Evol-Instruct / Magpie / 种子改写增强")
    parser.add_argument("--paradigm", nargs="+", choices=PARADIGMS_ALL,
                        default=PARADIGMS_ALL, help="生成范式（可多个）")
    parser.add_argument("--models", nargs="+", choices=MODELS_ALL,
                        default=MODELS_ALL, help="生成用模型（可多个，缺 Key 自动跳过）")
    parser.add_argument("--seeds", default=None,
                        help="种子指令，多个用 | 分隔；不给则用模板自带/内置默认种子。"
                             "注意：seed_rewrite 模板自带的是『指令+回答』数据对，"
                             "此处传入的只是纯指令（回答由模型生成）")
    parser.add_argument("--max-tokens", type=int, default=16384,
                        help="单次生成最大 token 数（默认 16384；推理模型的思维链也占用该预算）")
    parser.add_argument("--timeout", type=int, default=600,
                        help="单次调用读超时秒数（默认 600；推理模型生成大 JSON 较慢，"
                             "实测 GLM/Qwen 单次约 250s，勿低于 300）")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "data" / "syn"),
                        help="输出目录（默认 data/syn）")
    parser.add_argument("--dry-run", action="store_true",
                        help="不调用模型，仅打印将发送的提示词长度与开头片段")
    args = parser.parse_args()

    seeds = [s.strip() for s in args.seeds.split("|") if s.strip()] if args.seeds else None

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"syn_data_{ts}.jsonl"
    err_path = out_dir / f"errors_{ts}.log"

    # 过滤可用模型（缺 Key 提示后跳过）；dry-run 展示全部计划，不真正调用
    models, _missing = llm.check_available(args.models)
    if args.dry_run:
        models = args.models
    if not models:
        sys.exit("错误：没有可用的模型。请按提示配置 .env（参考 .env.example）。")

    total_added = 0
    total_calls = 0
    seen = set()
    stats = []  # 每轮结果的统计，便于最后汇总打印

    for paradigm in args.paradigm:
        template = promptio.load_prompt(paradigm)
        # 范式无默认种子时补内置种子
        eff_seeds = seeds
        if eff_seeds is None and paradigm == "self_instruct":
            eff_seeds = DEFAULT_SEEDS_SELF_INSTRUCT

        for model in models:
            total_calls += 1
            tag = f"[{paradigm} / {model}]"
            if args.dry_run:
                messages = build_messages(paradigm, template, eff_seeds)
                user_text = messages[1]["content"]
                seed_src = f"--seeds 指定 {len(eff_seeds)} 条" if eff_seeds else "模板内置"
                print(f"{tag} 提示词 {len(user_text)} 字符，种子：{seed_src}。开头预览：")
                print("    " + user_text[:220].replace("\n", " "))
                continue
            try:
                pairs, raw = generate_one(model, paradigm, template, eff_seeds,
                                          args.max_tokens, args.timeout)
            except Exception as e:
                print(f"{tag} ✗ 失败：{e}", flush=True)
                with open(err_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"paradigm": paradigm, "model": model,
                                        "error": str(e)}, ensure_ascii=False) + "\n")
                continue

            # 展平结果 → 统一 JSONL，做整体去重
            added = 0
            with open(out_path, "a", encoding="utf-8") as f:
                for rec in pairs:
                    key = (rec["instruction"], rec["response"])
                    if key in seen:
                        continue
                    seen.add(key)
                    line = {
                        "paradigm": paradigm,
                        "category": rec["category"],
                        "source_model": model,
                        "instruction": rec["instruction"],
                        "response": rec["response"],
                    }
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
                    added += 1
            total_added += added
            stat_line = f"{tag} 返回 {len(raw)} 字符 → 解析 {len(pairs)} 对 → 新增 {added} 对"
            # 返回体量与解析出的数据量严重不匹配时给出提示，避免丢数据无声无息
            per_pair = len(raw) // max(len(pairs), 1)
            if per_pair > SUSPICIOUS_CHARS_PER_PAIR:
                stat_line += (f"  ⚠ 平均每对 {per_pair} 字符，远超常理"
                              f"（阈值 {SUSPICIOUS_CHARS_PER_PAIR}），疑似漏解析")
            stats.append(stat_line)

    # 汇总输出
    print("\n==================== 生成结果汇总 ====================")
    if args.dry_run:
        print(f"dry-run：共 {total_calls} 次调用计划（未真正调用 API）")
    else:
        for s in stats:
            print("  " + s)
        print(f"本次共 {total_calls} 次调用，产出去重后 {total_added} 条数据")
        print(f"数据集文件：{out_path}")
        if out_path.exists() and out_path.stat().st_size == 0:
            print("⚠ 文件为空：请检查 prompts/ 模板与模型返回，或查看错误日志 "
                  f"{err_path}")


if __name__ == "__main__":
    main()
