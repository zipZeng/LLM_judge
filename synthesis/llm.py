# -*- coding: utf-8 -*-
"""
llm.py — 云端大模型统一调用封装（OpenAI 兼容接口）

供合成数据生成（generate.py）与生成模型对比（compare_models.py）共用，
支持三家模型，与 git_hub/LLM_judge 的模型配置保持一致：

    deepseek : DeepSeek 官方 API（deepseek-chat）
    glm      : 硅基流动 SiliconFlow（zai-org/GLM-5.3）
    qwen     : 硅基流动 SiliconFlow（Qwen/Qwen3.6-35B-A3B）

API Key 读取顺序（依次尝试）：
    1. 本目录 .env（DEEPSEEK_API_KEY / SILICONFLOW_API_KEY）
    2. git_hub/LLM_judge/.env（与语料评估项目共用，若存在）
    3. 系统环境变量

依赖：requests、python-dotenv（pip install requests python-dotenv）
"""

import os
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("缺少依赖 requests，请先执行：pip install requests python-dotenv")

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(path=None):
        """未安装 python-dotenv 时降级为不加载 .env（仅读环境变量）。"""
        return False

# 依次尝试加载 .env
_HERE = Path(__file__).resolve().parent
load_dotenv(_HERE / ".env")
_judge_env = _HERE.parent / "git_hub" / "LLM_judge" / ".env"
if _judge_env.exists():
    load_dotenv(_judge_env)

# ============================ 模型注册表 ============================
# 修改模型名/地址只需改这里（如硅基流动下线某模型后换成新 tag）

PROVIDERS = {
    "deepseek": {
        "label": "DeepSeek(deepseek-chat)",
        "url": "https://api.deepseek.com/v1/chat/completions",
        "api_key_env": "DEEPSEEK_API_KEY",
        "model": "deepseek-chat",
        "key_help": "DeepSeek 开放平台 https://platform.deepseek.com",
    },
    "glm": {
        "label": "GLM-5.3(zai-org/GLM-5.3)",
        "url": "https://api.siliconflow.cn/v1/chat/completions",
        "api_key_env": "SILICONFLOW_API_KEY",
        "model": "zai-org/GLM-5.3",
        "key_help": "硅基流动 https://cloud.siliconflow.cn",
    },
    "qwen": {
        "label": "Qwen3.6(Qwen/Qwen3.6-35B-A3B)",
        "url": "https://api.siliconflow.cn/v1/chat/completions",
        "api_key_env": "SILICONFLOW_API_KEY",
        "model": "Qwen/Qwen3.6-35B-A3B",
        "key_help": "硅基流动 https://cloud.siliconflow.cn",
    },
}

_PLACEHOLDERS = ("YOUR_DEEPSEEK_KEY", "YOUR_SILICONFLOW_KEY", "YOUR_API_KEY", "")


def provider_key(provider):
    """读取某模型的 API Key（环境变量 / .env）。"""
    return os.getenv(PROVIDERS[provider]["api_key_env"], "").strip()


def check_available(models):
    """过滤出已配置 Key 的模型，返回 (可用列表, 缺 Key 列表)。

    缺 Key 的模型自动跳过并打印提示，不影响其余模型继续执行。
    """
    avail, missing = [], []
    for m in models:
        key = provider_key(m)
        if key and key not in _PLACEHOLDERS:
            avail.append(m)
        else:
            missing.append(m)
    if missing:
        labels = ", ".join(PROVIDERS[m]["label"] for m in missing)
        print(f"⚠ 以下模型未配置 API Key，本次跳过：{labels}")
        help_text = PROVIDERS[missing[0]]["key_help"]
        print(f"   请到 {help_text} 获取 Key，写入 .env（参考 .env.example）后重试。")
    return avail, missing


def call_chat(provider, messages, temperature=0.8, max_tokens=8192,
              timeout=120, max_retry=3):
    """调用一次 chat 补全；成功返回回复文本（去首尾空白），失败抛 RuntimeError。"""
    p = PROVIDERS[provider]
    key = provider_key(provider)
    if not key or key in _PLACEHOLDERS:
        raise RuntimeError(f"模型 {p['label']} 未配置 API Key（{p['api_key_env']}）")

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": p["model"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    last_err = None
    for attempt in range(1, max_retry + 1):
        try:
            resp = requests.post(p["url"], headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                choice = data["choices"][0]
                text = (choice.get("message", {}).get("content") or "").strip()
                # 截断与空响应都是 token 预算不足的确定性结果，同样预算下重试必然复现，
                # 故与 4xx 一样直接抛出，把可操作的提示交给调用方，不浪费重试时间
                if choice.get("finish_reason") == "length":
                    # 推理模型的思维链同样吃 token 预算，预算不足时正文被截断成半截 JSON
                    raise RuntimeError(
                        f"输出被 max_tokens={max_tokens} 截断（finish_reason=length），"
                        f"请调大 --max-tokens 或减少单次生成数量")
                if not text:
                    raise RuntimeError(
                        f"模型返回空内容（finish_reason={choice.get('finish_reason')}），"
                        f"推理模型可能是思维链占满 token 预算，请调大 --max-tokens")
                return text
            elif 400 <= resp.status_code < 500 and resp.status_code != 429:
                # 4xx（除限流外）多为参数/Key 错误，重试无意义，直接抛出
                raise RuntimeError(f"HTTP {resp.status_code}：{resp.text[:300]}")
            else:
                last_err = f"HTTP {resp.status_code}：{resp.text[:200]}"
        except requests.exceptions.RequestException as e:
            last_err = str(e)
        wait = 3 * attempt
        if attempt < max_retry:
            print(f"  [重试 {attempt}/{max_retry - 1}] {p['label']} 请求失败：{last_err}"
                  f"，{wait}s 后重试...", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"模型 {p['label']} 请求失败（已重试 {max_retry} 次）：{last_err}")
