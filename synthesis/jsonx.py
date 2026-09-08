# -*- coding: utf-8 -*-
"""
jsonx.py — 模型 JSON 输出的鲁棒解析工具

大模型常会输出 ```json 代码块围栏、前后多余文字、键名漂移甚至不完整 JSON。
这里提供：
    1. extract_json()  —— 从杂文里尽力提取第一个合法 JSON 对象/数组
    2. walk_pairs()    —— 递归展平所有「指令-响应对」，兼容中英文键名
    3. deep_find()     —— 递归查找某个关键字段所在的 dict（用于评审结果）
"""

import json
import re


def strip_fences(text):
    """去掉 markdown ```json ... ``` 围栏。"""
    text = re.sub(r"```(?:json|JSON)?\s*", "", text)
    text = text.replace("```", "")
    return text.strip()


def extract_json(text):
    """从模型输出中提取第一个能成功解析的 JSON 对象/数组；失败返回 None。

    策略：去除围栏后，从每个 '{' / '[' 起点做括号配平（跳过字符串内的括号），
    在配平归零处尝试 json.loads，取第一个成功的完整片段。
    """
    if not text:
        return None
    text = strip_fences(text)
    for start, ch in enumerate(text):
        if ch not in "{[":
            continue
        open_c, close_c = ("{", "}") if ch == "{" else ("[", "]")
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == open_c:
                depth += 1
            elif c == close_c:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break  # 该起点解析失败，继续找下一个起点
    return None


def walk_pairs(node, inherited=None):
    """递归遍历解析后的 JSON，提取所有「指令-响应对」节点。

    兼容模板要求的英文键 instruction/response，
    也兜底常见中文键（指令/响应/回答），提高对模型键名漂移的容错。

    返回记录列表：[{"instruction": str, "response": str, "category": str}]
    category 由节点所在层级附近的 domain / type / task_name /
    evolution_type 等语义字段用 " | " 拼接，用于事后按类别统计。
    """

    def _walk(n, cats):
        if isinstance(n, dict):
            cur = list(cats)
            for k in ("domain", "type", "task_name", "task_type",
                      "evolution_type", "title"):
                v = n.get(k)
                if isinstance(v, str) and v.strip() and v.strip() not in cur:
                    cur.append(v.strip())
            ins = n.get("instruction") or n.get("指令")
            res = n.get("response") or n.get("响应") or n.get("回答")
            if isinstance(ins, str) and isinstance(res, str) and ins.strip():
                records.append({
                    "instruction": ins.strip(),
                    "response": res.strip(),
                    "category": " | ".join(cur),
                })
                return  # 已是数据对叶子节点，不再下钻
            for v in n.values():
                _walk(v, cur)
        elif isinstance(n, list):
            for it in n:
                _walk(it, cats)

    records = []
    _walk(node, list(inherited or []))
    return records


def deep_find(node, key):
    """递归查找「首个」包含指定 key 字段的 dict；找不到返回 None。

    例：deep_find(评审输出, "model_scores") 拿到打分表 dict。
    """
    if isinstance(node, dict):
        if key in node:
            return node
        for v in node.values():
            found = deep_find(v, key)
            if found is not None:
                return found
    elif isinstance(node, list):
        for it in node:
            found = deep_find(it, key)
            if found is not None:
                return found
    return None
