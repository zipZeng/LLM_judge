# -*- coding: utf-8 -*-
"""
eval_loop.py — 语料质量多模型迭代评估脚本

功能：
    1. 加载 data/judge_prompts.jsonl 中的 4 条评估提示词
       （safety / accuracy / diversity / format）
    2. 支持两种输入方式：命令行直接传入文本，或用 --file 读取文件
    3. 核心流程：3 个模型（DeepSeek + 硅基流动 GLM-5.3 / Qwen3.6-35B-A3B）分别对 4 个维度打分，
       每轮汇总各方意见后发回修正，共迭代 3 轮
    4. 输出完整 JSON 评估报告（含每轮详情、各维度权重、加权综合得分、各轮得分变化）

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
from functools import partial

import requests
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# ============================ 配置区（按需修改） ============================

# DeepSeek 在线 API
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "YOUR_DEEPSEEK_KEY")  # 从 .env 文件或环境变量读取

# 硅基流动在线 API（OpenAI 兼容）
SILICONFLOW_URL = "https://api.siliconflow.cn/v1/chat/completions"
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "YOUR_SILICONFLOW_KEY")  # 从 .env 文件或环境变量读取
SILICONFLOW_MODELS = ["zai-org/GLM-5.3", "Qwen/Qwen3.6-35B-A3B"]

# 评估提示词文件（相对项目根目录）
JUDGE_PROMPTS_PATH = os.path.join(PROJECT_ROOT, "data", "judge_prompts.jsonl")

# 迭代轮数、合格阈值、单次请求超时（秒）、最大重试次数
ROUNDS = 3
PASS_THRESHOLD = 5.0
TIMEOUT = 120
MAX_RETRY = 3

# 需要评估的维度（需与 judge_prompts.jsonl 中的 name 一致）
DIMENSIONS = ["safety", "accuracy", "diversity", "format"]

# 各维度权重（用于加权综合得分，权重总和建议为 1.0；缺失维度会自动归一化）
DIMENSION_WEIGHTS = {
    "safety": 0.35,
    "accuracy": 0.30,
    "diversity": 0.20,
    "format": 0.15,
}

# 维度中文显示名（用于最终结果与各轮变化表格）
DIM_LABELS = {
    "safety": "安全性",
    "accuracy": "准确性",
    "diversity": "多样性",
    "format": "格式",
}

# 分数提取正则：匹配 "[[8]]"、"[[8.5]]" 等
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


def call_siliconflow(system_prompt, user_prompt, model):
    """调用硅基流动 /chat/completions（OpenAI 兼容），返回模型输出的原始文本。

    model 为 SILICONFLOW_MODELS 中的模型 ID。
    """
    if SILICONFLOW_API_KEY == "YOUR_SILICONFLOW_KEY":
        raise RuntimeError("硅基流动 API Key 仍是占位符，请在 .env 文件或环境变量中设置 SILICONFLOW_API_KEY")

    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json",
    }
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 2048,
        "stream": False,
    }
    data = _post_json(SILICONFLOW_URL, payload, headers=headers)
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"硅基流动响应结构异常：{data}")


# 三个模型的统一注册表（key 用于报告，label 用于展示，call 为统一调用函数）
# 硅基流动的两个模型通过 partial 绑定各自的 model ID，保持 call(system_prompt, user_prompt) 统一签名
MODELS = [
    {"key": "deepseek", "label": f"DeepSeek({DEEPSEEK_MODEL})", "call": call_deepseek},
    {"key": "glm53", "label": "GLM-5.3（硅基流动）", "call": partial(call_siliconflow, model=SILICONFLOW_MODELS[0])},
    {"key": "qwen36", "label": "Qwen3.6-35B-A3B（硅基流动）", "call": partial(call_siliconflow, model=SILICONFLOW_MODELS[1])},
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


def extract_reason(text, max_len=100):
    """从模型输出中提取「精炼理由」作为控制台显示的简短摘要（完整理由仍存于报告 reason 字段）。

    优先匹配「精炼理由：xxx」格式；若模型未按格式输出，则回退到评分前的文字截取。
    """
    # 1. 优先提取「精炼理由：」后面的内容，到「完整分析 / 分数」或行尾为止
    m = re.search(r"精炼理由[:：]\s*(.+?)(?=完整分析|分数[:：]|\[\[|$)", text, re.DOTALL)
    if m:
        reason = re.sub(r"\s+", " ", m.group(1)).strip(" 。.；;，,")
        if reason:
            return reason[:max_len] + ("…" if len(reason) > max_len else "")

    # 2. 兜底：取评分前的内容，压缩空白后截取前 max_len 字符
    m2 = SCORE_PATTERN.search(text)
    prefix = text[:m2.start()] if m2 else text
    prefix = re.sub(r"\s+", " ", prefix).strip()
    if not prefix:
        return ""
    return prefix[:max_len] + ("…" if len(prefix) > max_len else "")


def build_refine_prompt(dim, text, prev_dim_results):
    """构造第 2、3 轮的修正提示词：在原始评判指令后附加上一轮各方意见。

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
    """根据最后一轮结果计算各维度得分、加权综合得分与结论。

    返回 (dim_scores, overall, passed, conclusion)：
        dim_scores  各维度均分（各模型平均）
        overall     加权综合得分（按 DIMENSION_WEIGHTS 加权，缺失维度自动归一化）
    """
    dim_scores = {}
    for d in DIMENSIONS:
        scores = [
            v["score"]
            for v in last_round_dim[d].values()
            if isinstance(v.get("score"), (int, float))
        ]
        dim_scores[d] = round(sum(scores) / len(scores), 2) if scores else None

    # 加权综合得分：只对有效分数加权，缺失维度按剩余权重归一化
    weighted = 0.0
    weight_sum = 0.0
    for d in DIMENSIONS:
        s = dim_scores[d]
        w = DIMENSION_WEIGHTS.get(d, 0.0)
        if s is not None and w > 0:
            weighted += s * w
            weight_sum += w
    overall = round(weighted / weight_sum, 2) if weight_sum > 0 else None

    passed = overall is not None and overall >= PASS_THRESHOLD
    conclusion = "合格" if passed else "不合格"
    return dim_scores, overall, passed, conclusion


def compute_round_scores(rounds_results):
    """计算每个维度在每一轮的聚合得分（各模型均值）。

    返回 {dim: [第1轮得分, 第2轮得分, ...]}，无有效分数时为 None。
    """
    dim_rounds = {d: [] for d in DIMENSIONS}
    for rr in rounds_results:
        dims = rr.get("dimensions", {})
        for d in DIMENSIONS:
            scores = [
                v["score"]
                for v in dims.get(d, {}).values()
                if isinstance(v.get("score"), (int, float))
            ]
            avg = round(sum(scores) / len(scores), 2) if scores else None
            dim_rounds[d].append(avg)
    return dim_rounds


def trend_arrow(scores):
    """根据首尾轮得分判断趋势：↑ 上升 / ↓ 下降 / → 持平。"""
    valid = [s for s in scores if isinstance(s, (int, float))]
    if len(valid) < 2:
        return "→"
    first, last = valid[0], valid[-1]
    if last > first:
        return "↑"
    if last < first:
        return "↓"
    return "→"


def show_progress(step, total, label=""):
    """用 \\r 覆盖式打印进度条（ASCII 字符，兼容 Windows 终端）。"""
    if total <= 0:
        total = 1
    pct = step * 100.0 / total
    bar_len = 20
    filled = int(round(bar_len * step / total))
    bar = "#" * filled + "-" * (bar_len - filled)
    sys.stdout.write(f"\r  进度: [{bar}] {step}/{total} ({pct:5.1f}%) {label}   ")
    sys.stdout.flush()


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
    parser.add_argument("--progress", action="store_true", help="用进度条替代逐条详情打印（批量评估时建议开启）")
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
    if SILICONFLOW_API_KEY == "YOUR_SILICONFLOW_KEY":
        print("⚠ 提醒：硅基流动 API Key 仍为占位符，硅基流动相关请求会失败，请在 .env 文件或环境变量中设置 SILICONFLOW_API_KEY。", flush=True)

    print(
        f"开始评估：语料长度 {len(text)} 字符 | 模型 {len(MODELS)} 个 | "
        f"维度 {len(DIMENSIONS)} 个 | 轮数 {rounds}",
        flush=True,
    )

    # ---- 3. 迭代评估 ----
    rounds_results = []
    prev_dim = None  # {dim_name: {label: {"score", "reason"}}}
    total_steps = rounds * len(DIMENSIONS) * len(MODELS)
    step = 0

    for r in range(1, rounds + 1):
        if not args.progress:
            print(f"\n==================== 第 {r}/{rounds} 轮 ====================", flush=True)
        round_dim = {}

        for dim_name in DIMENSIONS:
            dim = dims[dim_name]
            if not args.progress:
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
                    step += 1

                    if score is None:
                        round_dim[dim_name][label] = {"score": None, "reason": raw, "error": "分数解析失败"}
                    else:
                        round_dim[dim_name][label] = {"score": score, "reason": raw}

                    if args.progress:
                        show_progress(step, total_steps, f"{DIM_LABELS.get(dim_name, dim_name)} · {label}")
                    else:
                        if score is None:
                            print(f"    - {label}: 调用成功但未提取到分数", flush=True)
                        else:
                            print(f"    - {label}: {score} 分", flush=True)
                            # 第 1 轮显示简短扣分理由
                            reason_short = extract_reason(raw)
                            if r == 1 and reason_short:
                                print(f"      理由：{reason_short}", flush=True)
                except Exception as e:
                    step += 1
                    round_dim[dim_name][label] = {"score": None, "reason": "", "error": str(e)}
                    if args.progress:
                        show_progress(step, total_steps, f"{DIM_LABELS.get(dim_name, dim_name)} · {label} 失败")
                    else:
                        print(f"    - {label}: 调用失败 -> {e}", flush=True)

        rounds_results.append({"round": r, "dimensions": round_dim})
        prev_dim = round_dim

    # ---- 4. 汇总最终结果 ----
    dim_scores, overall, passed, conclusion = compute_final(prev_dim)
    round_scores = compute_round_scores(rounds_results)

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
            "weights": DIMENSION_WEIGHTS,
            "overall_score": overall,
            "passed": passed,
            "conclusion": conclusion,
            "round_trend": round_scores,
        },
    }

    # ---- 5. 输出 ----
    out_path = args.output or os.path.join(
        PROJECT_ROOT, "reports", f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if args.progress:
        print()  # 进度条结束后换行，避免与最终结果粘连

    # 最终结果：各维度得分 × 权重 = 加权得分，及加权综合得分
    print("\n==================== 最终评估结果 ====================", flush=True)
    for d in DIMENSIONS:
        s = dim_scores[d]
        label = DIM_LABELS.get(d, d)
        if s is None:
            print(f"  {label:<6}: 未评分", flush=True)
        else:
            w = DIMENSION_WEIGHTS.get(d, 0.0)
            weighted = round(s * w, 2)
            pct = int(round(w * 100))
            print(f"  {label:<6}: {s:<5} × {pct}% = {weighted:.2f}", flush=True)
    print(f"  加权综合得分：{overall if overall is not None else '未评分'}（阈值 {PASS_THRESHOLD}）", flush=True)
    print(f"  结论      ：{conclusion}", flush=True)

    # 各轮得分变化表
    print("\n各轮得分变化：", flush=True)
    print("  " + "维度".ljust(6) + "  " + "  ".join(f"第{r}轮" for r in range(1, rounds + 1)) + "  趋势", flush=True)
    for d in DIMENSIONS:
        label = DIM_LABELS.get(d, d)
        scores = round_scores[d]
        cells = "  ".join(f"{s if s is not None else '--':>6}" for s in scores)
        arrow = trend_arrow(scores)
        print(f"  {label:<6}  {cells}  {arrow}", flush=True)

    print(f"\n完整 JSON 报告已保存：{out_path}", flush=True)


if __name__ == "__main__":
    main()
