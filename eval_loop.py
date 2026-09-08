# -*- coding: utf-8 -*-
"""
eval_loop.py — 语料质量多模型迭代评估脚本（基于 FastChat llm_judge 提示词）

功能：
    1. 加载 fastchat/llm_judge/data/judge_prompts.jsonl 中的 4 条评估提示词
       （safety / accuracy / diversity / format）
    2. 支持两种输入方式：命令行直接传入文本，或用 --file 读取文件
    3. 核心流程：2 个模型（Ollama + DeepSeek）分别对 4 个维度打分，
       每轮汇总双方意见后发回修正，共迭代 3 轮
    4. 输出完整 JSON 评估报告（含每轮详情与最终合格/不合格结论，阈值 5 分）

用法示例：
    python eval_loop.py "这是一段待评估的语料文本"
    python eval_loop.py --file corpus.txt
    python eval_loop.py --file corpus.txt --rounds 3 --threshold 5 --output report.json

依赖：requests、python-dotenv（若未安装请执行 pip install requests python-dotenv）
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

import requests
from dotenv import load_dotenv

load_dotenv()

# ============================ 配置区（按需修改） ============================

# Ollama 本地服务
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3.5:4b"

# DeepSeek 在线 API
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "YOUR_DEEPSEEK_KEY")  # 从 .env 文件或环境变量读取

# 评估提示词文件（相对脚本运行目录）
JUDGE_PROMPTS_PATH = "data/judge_prompts.jsonl"

# 迭代轮数、合格阈值、单次请求超时（秒）、最大重试次数
ROUNDS = 3
PASS_THRESHOLD = 5.0
TIMEOUT = 120
MAX_RETRY = 3

# 需要评估的维度（需与 judge_prompts.jsonl 中的 name 一致）
DIMENSIONS = ["safety", "accuracy", "diversity", "format"]

# 分数提取正则：匹配 "[[8]]"、"[[8.5]]" 等（与 FastChat common.py 保持一致）
SCORE_PATTERN = re.compile(r"\[\[(\d+(?:\.\d+)?)\]\]")


# ============================ 基础工具函数 ============================


def _ensure_utf8_stdout():
    """尽量让 Windows 控制台以 UTF-8 输出，避免中文乱码。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def load_judge_prompts(path):
    """加载 judge_prompts.jsonl，返回 {name: 完整提示词 dict}。"""
    prompts = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                prompts[obj["name"]] = obj
    except FileNotFoundError:
        sys.exit(f"错误：找不到评估提示词文件 {path}")
    except json.JSONDecodeError as e:
        sys.exit(f"错误：评估提示词文件 JSON 解析失败 {path} -> {e}")
    return prompts


def _post_json(url, payload, headers=None, timeout=TIMEOUT):
    """POST JSON 请求，带超时与重试；成功返回解析后的 JSON dict，否则抛 RuntimeError。

    覆盖异常：请求超时（Timeout）、HTTP 错误（RequestException）、响应 JSON 解析失败。
    """
    last_err = None
    for attempt in range(1, MAX_RETRY + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            last_err = f"请求超时（{timeout}s）"
        except requests.exceptions.RequestException as e:
            # 含连接错误、HTTP 状态码错误、以及 requests 抛出的 JSON 解码错误
            last_err = f"请求异常：{e}"
        except ValueError as e:
            # resp.json() 解析失败的兜底
            last_err = f"响应 JSON 解析失败：{e}"

        if attempt < MAX_RETRY:
            wait = 2 * attempt
            print(f"      [重试 {attempt}/{MAX_RETRY - 1}] {last_err}，{wait}s 后重试...", flush=True)
            time.sleep(wait)

    raise RuntimeError(last_err)


def call_ollama(system_prompt, user_prompt):
    """调用 Ollama /api/generate，返回模型输出的原始文本。"""
    full_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": full_prompt,
        "stream": False,
        "options": {"temperature": 0},
    }
    data = _post_json(OLLAMA_URL, payload)
    if "response" not in data:
        raise RuntimeError(f"Ollama 响应缺少 response 字段：{data}")
    return data["response"].strip()


def call_deepseek(system_prompt, user_prompt):
    """调用 DeepSeek /chat/completions（OpenAI 兼容），返回模型输出的原始文本。"""
    if DEEPSEEK_API_KEY == "YOUR_DEEPSEEK_KEY":
        raise RuntimeError("DeepSeek API Key 仍是占位符，请在 .env 文件或环境变量中设置 DEEPSEEK_API_KEY")

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 2048,
        "stream": False,
    }
    data = _post_json(DEEPSEEK_URL, payload, headers=headers)
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"DeepSeek 响应结构异常：{data}")


# 两个模型的统一注册表（key 用于报告，label 用于展示，call 为统一调用函数）
MODELS = [
    {"key": "ollama", "label": f"Ollama({OLLAMA_MODEL})", "call": call_ollama},
    {"key": "deepseek", "label": f"DeepSeek({DEEPSEEK_MODEL})", "call": call_deepseek},
]


def extract_score(text):
    """从模型输出中提取 0-10 的分数，解析失败返回 None。"""
    m = SCORE_PATTERN.search(text)
    if m:
        score = float(m.group(1))
        return min(10.0, max(0.0, score))

    # 兜底：寻找文本中最后一个位于 0-10 之间的数字
    fallback = None
    for num in re.findall(r"\d+(?:\.\d+)?", text):
        try:
            score = float(num)
        except ValueError:
            continue
        if 0.0 <= score <= 10.0:
            fallback = score
    return fallback


def build_refine_prompt(dim, text, prev_dim_results):
    """构造第 2、3 轮的修正提示词：在原始评判指令后附加上一轮双方意见。

    dim              当前维度的提示词 dict
    text             待评估语料
    prev_dim_results {label: {"score": float|None, "reason": str}} 上一轮该维度的结果
    """
    # 用 replace 而非 format，避免语料中包含 { } 时导致 format 报错
    base = dim["prompt_template"].replace("{text}", text)

    lines = []
    for label, res in prev_dim_results.items():
        score_str = f"{res['score']}" if res.get("score") is not None else "未评分"
        reason = (res.get("reason") or "").strip().replace("\n", " ")
        reason = reason[:400]
        lines.append(f"- {label}：评分 {score_str}，理由：{reason}")
    feedback = "\n".join(lines)

    refine = (
        "[上一轮评估结果与修正要求]\n"
        f"{feedback}\n\n"
        "请参考上一轮各模型的评分与理由，重新独立评估本条语料。"
        "若你认为其他模型指出的问题更合理，请修正你的分数；若坚持原判，请给出更有力的理由。"
        "仍然严格按照格式输出分数：\"[[rating]]\"，例如 \"[[8]]\"。"
    )
    return base + "\n\n" + refine


def compute_final(last_round_dim):
    """根据最后一轮结果计算各维度得分、综合得分与结论。"""
    dim_scores = {}
    for d in DIMENSIONS:
        scores = [
            v["score"]
            for v in last_round_dim[d].values()
            if isinstance(v.get("score"), (int, float))
        ]
        dim_scores[d] = round(sum(scores) / len(scores), 2) if scores else None

    valid = [s for s in dim_scores.values() if s is not None]
    overall = round(sum(valid) / len(valid), 2) if valid else None
    passed = overall is not None and overall >= PASS_THRESHOLD
    conclusion = "合格" if passed else "不合格"
    return dim_scores, overall, passed, conclusion


# ============================ 主流程 ============================


def main():
    global PASS_THRESHOLD
    _ensure_utf8_stdout()

    parser = argparse.ArgumentParser(description="语料质量多模型迭代评估")
    parser.add_argument("text", nargs="?", help="待评估的语料文本（命令行直接传入）")
    parser.add_argument("-f", "--file", help="从文件读取待评估语料")
    parser.add_argument("-o", "--output", help="评估报告输出路径（默认自动生成时间戳文件名）")
    parser.add_argument("--rounds", type=int, default=ROUNDS, help=f"迭代轮数（默认 {ROUNDS}）")
    parser.add_argument("--threshold", type=float, default=PASS_THRESHOLD, help=f"合格阈值（默认 {PASS_THRESHOLD}）")
    args = parser.parse_args()

    PASS_THRESHOLD = args.threshold
    rounds = args.rounds
    if rounds < 1:
        sys.exit("错误：--rounds 必须 >= 1")

    # ---- 1. 读取输入语料 ----
    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                text = f.read()
        except OSError as e:
            sys.exit(f"错误：读取文件失败 {args.file} -> {e}")
        src = f"file:{args.file}"
    elif args.text:
        text = args.text
        src = "cli"
    else:
        print("未提供文本，请从标准输入输入语料（Windows: 输入后 Ctrl+Z 回车；Linux/Mac: Ctrl+D）：", flush=True)
        text = sys.stdin.read()
        src = "stdin"

    text = text.strip()
    if not text:
        sys.exit("错误：待评估语料为空，请检查输入。")

    # ---- 2. 加载评估提示词并校验维度 ----
    dims = load_judge_prompts(JUDGE_PROMPTS_PATH)
    missing = [d for d in DIMENSIONS if d not in dims]
    if missing:
        sys.exit(f"错误：提示词文件缺少以下维度：{missing}（现有：{list(dims.keys())}）")

    if DEEPSEEK_API_KEY == "YOUR_DEEPSEEK_KEY":
        print("⚠ 提醒：DeepSeek API Key 仍为占位符，DeepSeek 相关请求会失败，请在 .env 文件或环境变量中设置 DEEPSEEK_API_KEY。", flush=True)

    print(
        f"开始评估：语料长度 {len(text)} 字符 | 模型 {len(MODELS)} 个 | "
        f"维度 {len(DIMENSIONS)} 个 | 轮数 {rounds}",
        flush=True,
    )

    # ---- 3. 迭代评估 ----
    rounds_results = []
    prev_dim = None  # {dim_name: {label: {"score", "reason"}}}

    for r in range(1, rounds + 1):
        print(f"\n==================== 第 {r}/{rounds} 轮 ====================", flush=True)
        round_dim = {}

        for dim_name in DIMENSIONS:
            dim = dims[dim_name]
            print(f"  ▶ 维度 [{dim_name}]", flush=True)
            round_dim[dim_name] = {}

            for model in MODELS:
                label = model["label"]
                try:
                    if r == 1:
                        user_prompt = dim["prompt_template"].replace("{text}", text)
                    else:
                        user_prompt = build_refine_prompt(dim, text, prev_dim[dim_name])

                    raw = model["call"](dim["system_prompt"], user_prompt)
                    score = extract_score(raw)
                    if score is None:
                        print(f"    - {label}: 调用成功但未提取到分数", flush=True)
                        round_dim[dim_name][label] = {"score": None, "reason": raw, "error": "分数解析失败"}
                    else:
                        print(f"    - {label}: {score} 分", flush=True)
                        round_dim[dim_name][label] = {"score": score, "reason": raw}
                except Exception as e:
                    print(f"    - {label}: 调用失败 -> {e}", flush=True)
                    round_dim[dim_name][label] = {"score": None, "reason": "", "error": str(e)}

        rounds_results.append({"round": r, "dimensions": round_dim})
        prev_dim = round_dim

    # ---- 4. 汇总最终结果 ----
    dim_scores, overall, passed, conclusion = compute_final(prev_dim)

    report = {
        "meta": {
            "input_source": src,
            "text": text,
            "text_length": len(text),
            "models": [m["label"] for m in MODELS],
            "rounds": rounds,
            "threshold": PASS_THRESHOLD,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
        "rounds": rounds_results,
        "final": {
            "dimension_scores": dim_scores,
            "overall_score": overall,
            "passed": passed,
            "conclusion": conclusion,
        },
    }

    # ---- 5. 输出 ----
    out_path = args.output or f"reports/eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n==================== 最终评估结果 ====================", flush=True)
    for d in DIMENSIONS:
        s = dim_scores[d]
        print(f"  {d:10s}: {'未评分' if s is None else s}", flush=True)
    print(f"  综合得分  : {'未评分' if overall is None else overall}", flush=True)
    print(f"  结论      : {conclusion}（阈值 {PASS_THRESHOLD}）", flush=True)
    print(f"\n完整 JSON 报告已保存：{out_path}", flush=True)


if __name__ == "__main__":
    main()
