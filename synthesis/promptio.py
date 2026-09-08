# -*- coding: utf-8 -*-
"""
promptio.py — 读取 prompts/ 目录下的提示词模板

模板文件位于实训项目根目录的 prompts/ 子目录（与 git_hub 同级）：
    self_instruct.md / evol_instruct.md / magpie.md / model_comparison.md

路径按本文件位置推导，因此从任意目录运行脚本都能找到模板。
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
PROMPTS_DIR = _HERE.parent / "prompts"

# 模板名 -> 文件名
PARADIGM_FILES = {
    "self_instruct": "self_instruct.md",
    "evol_instruct": "evol_instruct.md",
    "magpie": "magpie.md",
    "model_comparison": "model_comparison.md",
}


def load_prompt(name):
    """读取提示词模板全文（UTF-8）；模板缺失或读取失败时报错退出。"""
    fname = PARADIGM_FILES.get(name)
    if fname is None:
        sys.exit(f"错误：未知模板名 {name}，可选：{', '.join(PARADIGM_FILES)}")
    path = PROMPTS_DIR / fname
    if not path.exists():
        sys.exit(f"错误：找不到提示词模板 {path}")
    try:
        return path.read_text(encoding="utf-8")
    except OSError as e:
        sys.exit(f"错误：读取模板失败 {path} -> {e}")
