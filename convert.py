# -*- coding: utf-8 -*-
"""
convert.py — 将评估通过的语料转换为标准训练格式（Alpaca / ShareGPT）

功能：
    1. 读取评估报告：支持单个 JSON 文件、JSON 数组、或每行一个 JSON 对象的 JSONL 文件
       （即 eval_loop.py 输出的评估报告，单个为 .json，批量可拼接为 .jsonl）
    2. 过滤：仅保留 final.overall_score >= 阈值（默认 5 分）的语料
    3. 转换：把 meta.text（原始语料）转成目标训练格式

输入报告结构（eval_loop.py 输出）：
    {
      "meta":  {"text": "原始语料", ...},
      "final": {"overall_score": 8.0, "conclusion": "合格", ...}
    }

语料结构自动识别规则（对 meta.text 逐行解析）：
    - 对话结构：行首出现 human/user/人/用户、gpt/assistant/ai/助手 等标记，如
          human: 你好
          assistant: 你好，有什么可以帮你？
    - Alpaca 结构：行首出现 instruction/input/output 或 指令/输入/输出 等标记
    - 普通文本：以上标记均未出现时视为普通文本

转换规则：
    Alpaca 格式  {"instruction": ..., "input": ..., "output": ...}
        - Alpaca 结构   -> 直接映射 instruction/input/output
        - 对话结构      -> 首条 human 作 instruction，其余 human 合并作 input，gpt 合并作 output
        - 普通文本      -> 全部放入 instruction（input、output 留空）
    ShareGPT 格式 {"conversations": [{"from": "human"|"gpt", "value": ...}, ...]}
        - 对话结构      -> 直接映射 human/gpt 多轮
        - Alpaca 结构   -> human = instruction(+input)，gpt = output
        - 普通文本      -> 单条 {"from": "gpt", "value": 文本}（视作一条模型回复）

用法示例：
    python convert.py --input eval_report.json --output output.jsonl --format alpaca --threshold 5
    python convert.py --input reports.jsonl  --output out.jsonl     --format sharegpt

依赖：仅标准库（Python 3.8+）
"""

import argparse
import json
import re
import sys

# ============================ 配置区 ============================

DEFAULT_THRESHOLD = 5.0
FORMATS = ("alpaca", "sharegpt")

# 对话角色标记：行首「角色:内容」
CONV_PATTERN = re.compile(
    r"^\s*(human|user|assistant|gpt|ai|人|用户|助手)\s*[:：]\s*(.*)$", re.IGNORECASE
)
# Alpaca 字段标记：行首「字段:内容」
ALPACA_PATTERN = re.compile(
    r"^\s*(instruction|input|output|指令|输入|输出)\s*[:：]\s*(.*)$", re.IGNORECASE
)


# ============================ 基础工具函数 ============================


def _ensure_utf8_stdout():
    """尽量让 Windows 控制台以 UTF-8 输出，避免中文乱码。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def _normalize_role(token):
    """把对话标记归一化为 human / gpt，无法识别返回 None。"""
    t = token.strip().lower()
    if t in ("human", "user", "人", "用户"):
        return "human"
    if t in ("gpt", "assistant", "ai", "助手"):
        return "gpt"
    return None


def _normalize_field(token):
    """把 Alpaca 字段标记归一化为 instruction / input / output，无法识别返回 None。"""
    t = token.strip().lower()
    mapping = {
        "instruction": "instruction",
        "input": "input",
        "output": "output",
        "指令": "instruction",
        "输入": "input",
        "输出": "output",
    }
    return mapping.get(t)


def _read_reports(path):
    """读取评估报告，返回 list[dict]。

    依次尝试：单个 JSON 对象、JSON 数组、JSONL（每行一个对象）。
    文件不存在或内容无法解析时给出友好提示并退出。
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        sys.exit(f"错误：输入文件不存在：{path}")
    except OSError as e:
        sys.exit(f"错误：读取输入文件失败：{path} -> {e}")

    if not content.strip():
        sys.exit("错误：输入文件为空。")

    # 1) 尝试作为单个 JSON（对象或数组）
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = None

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]

    # 2) 按 JSONL 逐行解析
    reports = []
    for i, line in enumerate(content.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            reports.append(json.loads(line))
        except json.JSONDecodeError as e:
            sys.exit(f"错误：第 {i} 行不是合法 JSON：{e}")
    return reports


def _parse_conversation(lines):
    """按对话结构解析，返回 [{"role", "value"}]；未识别到任何对话标记返回 None。"""
    turns = []       # [(role, value), ...]
    cur_role = None
    cur_buf = []

    for line in lines:
        m = CONV_PATTERN.match(line)
        if m:
            role = _normalize_role(m.group(1))
            if role is None:
                continue
            if cur_role is not None and cur_buf:
                turns.append((cur_role, "\n".join(cur_buf).strip()))
            cur_role = role
            cur_buf = [m.group(2).strip()] if m.group(2).strip() else []
        else:
            if cur_role is not None:
                cur_buf.append(line)

    if cur_role is not None and cur_buf:
        turns.append((cur_role, "\n".join(cur_buf).strip()))

    if not turns:
        return None

    # 合并连续相同角色、过滤空值
    merged = []
    for role, val in turns:
        val = val.strip()
        if not val:
            continue
        if merged and merged[-1][0] == role:
            merged[-1] = (role, merged[-1][1] + "\n" + val)
        else:
            merged.append((role, val))

    if not merged:
        return None
    return [{"role": r, "value": v} for r, v in merged]


def _parse_alpaca(lines):
    """按 Alpaca 字段结构解析，返回 {"instruction","input","output"}；未识别返回 None。"""
    fields = {"instruction": "", "input": "", "output": ""}
    current = None
    buf = []
    found = False

    for line in lines:
        m = ALPACA_PATTERN.match(line)
        if m:
            key = _normalize_field(m.group(1))
            if key is None:
                continue
            if current is not None:
                fields[current] = "\n".join(buf).strip()
            current = key
            found = True
            buf = [m.group(2).strip()] if m.group(2).strip() else []
        else:
            if current is not None:
                buf.append(line)

    if current is not None:
        fields[current] = "\n".join(buf).strip()

    return fields if found else None


def _parse_text(text):
    """把原始语料解析为统一中间结构。

    返回 dict：
        {"type": "conversation", "turns": [{"role","value"}, ...]}
        {"type": "alpaca", "instruction": ..., "input": ..., "output": ...}
        {"type": "plain", "text": ...}
    """
    lines = text.splitlines()

    conv = _parse_conversation(lines)
    if conv is not None:
        return {"type": "conversation", "turns": conv}

    alp = _parse_alpaca(lines)
    if alp is not None:
        return {"type": "alpaca", **alp}

    return {"type": "plain", "text": text.strip()}


def to_alpaca(parsed):
    """把中间结构转为 Alpaca 格式。"""
    if parsed["type"] == "alpaca":
        return {
            "instruction": parsed["instruction"],
            "input": parsed["input"],
            "output": parsed["output"],
        }
    if parsed["type"] == "conversation":
        human_parts = [t["value"] for t in parsed["turns"] if t["role"] == "human"]
        gpt_parts = [t["value"] for t in parsed["turns"] if t["role"] == "gpt"]
        instruction = human_parts[0] if human_parts else ""
        input_ = "\n".join(human_parts[1:]) if len(human_parts) > 1 else ""
        output = "\n".join(gpt_parts)
        return {"instruction": instruction, "input": input_, "output": output}
    # plain
    return {"instruction": parsed["text"], "input": "", "output": ""}


def to_sharegpt(parsed):
    """把中间结构转为 ShareGPT 格式。"""
    if parsed["type"] == "conversation":
        return {
            "conversations": [
                {"from": t["role"], "value": t["value"]} for t in parsed["turns"]
            ]
        }
    if parsed["type"] == "alpaca":
        human_value = parsed["instruction"]
        if parsed["input"]:
            human_value = (human_value + "\n" + parsed["input"]) if human_value else parsed["input"]
        conv = []
        if human_value:
            conv.append({"from": "human", "value": human_value})
        if parsed["output"]:
            conv.append({"from": "gpt", "value": parsed["output"]})
        return {"conversations": conv}
    # plain：无结构文本视作一条模型回复（gpt）
    return {"conversations": [{"from": "gpt", "value": parsed["text"]}]}


# ============================ 主流程 ============================


def main():
    _ensure_utf8_stdout()

    parser = argparse.ArgumentParser(description="将评估通过的语料转换为训练格式（Alpaca/ShareGPT）")
    parser.add_argument("-i", "--input", required=True, help="输入文件：评估报告 JSON 或 JSONL")
    parser.add_argument("-o", "--output", required=True, help="输出文件（JSONL，每行一条）")
    parser.add_argument("-f", "--format", required=True, choices=FORMATS, help="输出格式：alpaca 或 sharegpt")
    parser.add_argument("-t", "--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"合格分数阈值（默认 {DEFAULT_THRESHOLD}）")
    args = parser.parse_args()

    reports = _read_reports(args.input)

    total = len(reports)
    kept = 0
    filtered_low = 0     # 得分不足
    skipped_invalid = 0  # 字段缺失或分数异常
    plain_fallback = 0   # 普通文本（无结构）数量
    out_records = []

    for idx, rep in enumerate(reports, start=1):
        # 提取分数与原始语料
        try:
            score = rep["final"]["overall_score"]
            text = rep["meta"]["text"]
        except (KeyError, TypeError) as e:
            skipped_invalid += 1
            continue

        # 分数必须是数字
        if not isinstance(score, (int, float)):
            skipped_invalid += 1
            continue

        # 过滤：得分不足
        if score < args.threshold:
            filtered_low += 1
            continue

        # 解析并转换
        parsed = _parse_text(text)
        if parsed["type"] == "plain":
            plain_fallback += 1

        if args.format == "alpaca":
            out_records.append(to_alpaca(parsed))
        else:
            out_records.append(to_sharegpt(parsed))
        kept += 1

    # 写出结果（JSONL，每行一条）
    try:
        with open(args.output, "w", encoding="utf-8") as f:
            for rec in out_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError as e:
        sys.exit(f"错误：写入输出文件失败：{args.output} -> {e}")

    # 统计输出
    print("\n==================== 转换统计 ====================")
    print(f"  读取总数              : {total}")
    print(f"  保留（得分 >= {args.threshold}）    : {kept}")
    print(f"  过滤（得分不足）      : {filtered_low}")
    print(f"  跳过（字段缺失/异常） : {skipped_invalid}")
    if plain_fallback:
        print(f"  其中普通文本（无结构，走兜底）: {plain_fallback}")
    print(f"  输出格式              : {args.format}")
    print(f"  输出文件              : {args.output}")


if __name__ == "__main__":
    main()
