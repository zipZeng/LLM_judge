# -*- coding: utf-8 -*-
"""
promptio.py — 读取 prompts/ 目录下的提示词模板

模板文件共 5 个：
    self_instruct.md / evol_instruct.md / magpie.md / seed_rewrite.md /
    model_comparison.md

按优先级在两个位置查找（先命中者为准）：

    1. 本文件同级的 synthesis/prompts/  —— 正式位置，让 synthesis/ 自成一体，
       随模块一起提交入库（缺了它 generate.py 跑不起来）
    2. 上一级的 <项目根>/prompts/        —— 兼容「模板放在实训项目根目录」的既有布局

路径按本文件位置推导，因此从任意目录运行脚本、或把 synthesis/ 整个目录
拷到别处，都能找到模板。
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# 模板搜索目录，按优先级从高到低
PROMPTS_DIRS = [
    _HERE / "prompts",           # synthesis/prompts/（优先）
    _HERE.parent / "prompts",    # <项目根>/prompts/（兼容旧布局）
]

# 模板名 -> 文件名
PARADIGM_FILES = {
    "self_instruct": "self_instruct.md",
    "evol_instruct": "evol_instruct.md",
    "magpie": "magpie.md",
    "seed_rewrite": "seed_rewrite.md",
    "model_comparison": "model_comparison.md",
}


def load_prompt(name):
    """读取提示词模板全文（UTF-8）；模板缺失或读取失败时报错退出。

    依次在 PROMPTS_DIRS 中查找，返回第一个存在的模板；
    全部找不到时报错并列出已尝试的完整路径，便于排查。
    """
    fname = PARADIGM_FILES.get(name)
    if fname is None:
        sys.exit(f"错误：未知模板名 {name}，可选：{', '.join(PARADIGM_FILES)}")
    for d in PROMPTS_DIRS:
        path = d / fname
        if not path.exists():
            continue
        try:
            return path.read_text(encoding="utf-8")
        except OSError as e:
            sys.exit(f"错误：读取模板失败 {path} -> {e}")
    tried = "\n  ".join(str(d / fname) for d in PROMPTS_DIRS)
    sys.exit(f"错误：找不到提示词模板 {fname}，已尝试以下路径：\n  {tried}")
