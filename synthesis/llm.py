# -*- coding: utf-8 -*-
"""
llm.py — 云端大模型统一调用封装（OpenAI 兼容接口）

供合成数据生成（generate.py）与生成模型对比（compare_models.py）共用。
模型配置统一维护在项目根目录 models.json（DeepSeek 一条 + 硅基流动 N 条），
运行时读取并动态构建 PROVIDERS 注册表 —— 代码里不写死具体模型名，
增删模型只改 models.json（缺失时回退到仅 DeepSeek 的内置默认）。

API Key 读取顺序（依次尝试）：
    1. 本目录 .env（DEEPSEEK_API_KEY / SILICONFLOW_API_KEY）
    2. git_hub/LLM_judge/.env（与语料评估项目共用，若存在）
    3. 系统环境变量

依赖：requests、python-dotenv（pip install requests python-dotenv）
"""

import contextlib
import json
import os
import re
import sys
import threading
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

# ============================ 模型注册表（从 models.json 读取） ============================
# 模型名/地址不再写死在代码里：项目根目录 models.json 统一维护
# （DeepSeek 一条 + 硅基流动 models 数组）。增删模型只改 models.json；
# 缺失/损坏时友好提示并回退到仅 DeepSeek 的内置默认。

_MODELS_CONFIG_PATH = _HERE.parent / "models.json"

_DEFAULT_DEEPSEEK = {
    "url": "https://api.deepseek.com/v1/chat/completions",
    "model": "deepseek-chat",
    "api_key_env": "DEEPSEEK_API_KEY",
}


def _slug(s):
    """把模型 ID 的末段转成稳定短键：'Kimi-K2.7-Code' -> 'kimi-k2-7-code'。"""
    return re.sub(r"[^0-9a-z]+", "-", s.lower()).strip("-")


def _build_providers(config):
    """把 models.json 的原始结构组装成 PROVIDERS 注册表。

    config 为 None 表示回退：仅保留 DeepSeek，硅基流动模型列表为空。
    """
    cfg = config or {}
    deepseek = cfg.get("deepseek") or _DEFAULT_DEEPSEEK
    sf = cfg.get("siliconflow") or {}

    providers = {
        "deepseek": {
            "label": "DeepSeek",
            "url": deepseek.get("url", _DEFAULT_DEEPSEEK["url"]),
            "api_key_env": deepseek.get("api_key_env", "DEEPSEEK_API_KEY"),
            "model": deepseek.get("model", "deepseek-chat"),
            "key_help": "DeepSeek 开放平台 https://platform.deepseek.com",
        },
    }
    sf_url = sf.get("url")
    sf_key = sf.get("api_key_env", "SILICONFLOW_API_KEY")
    for model_id in sf.get("models") or []:
        base = model_id.rsplit("/", 1)[-1]
        key = _slug(base)
        n = 2
        while key in providers:  # 键名冲突时追加序号，保证稳定唯一
            key = f"{_slug(base)}-{n}"
            n += 1
        providers[key] = {
            "label": base,
            "url": sf_url,
            "api_key_env": sf_key,
            "model": model_id,
            "key_help": "硅基流动 https://cloud.siliconflow.cn",
        }
    return providers


def _load_providers():
    """读取项目根目录 models.json 构建 PROVIDERS；缺失/损坏时友好提示并回退。"""
    config = None
    if _MODELS_CONFIG_PATH.is_file():
        try:
            config = json.loads(_MODELS_CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"⚠ 读取 {_MODELS_CONFIG_PATH} 失败（{e}），回退到内置默认配置（仅 DeepSeek）。",
                  flush=True)
    else:
        print(f"⚠ 未找到 {_MODELS_CONFIG_PATH}，回退到内置默认配置（仅 DeepSeek）。"
              f"如需使用硅基流动模型，请创建 models.json（见 README「模型配置说明」）。",
              flush=True)
    return _build_providers(config)


PROVIDERS = _load_providers()

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


# ============================ 长调用进度提示 ============================
# 推理模型（Kimi-K2.7-Code / Qwen3.6）单次大 JSON 生成实测 4–7.5 分钟，这期间控制台
# 一个字都不动 —— 调试和演示时分不清"在跑"还是"卡死了"，很容易被手贱 Ctrl-C 掉。
# 故在等待期间起一个后台线程定期打印已用时。放在 llm.py 是因为这里是所有 API
# 调用的唯一收口，generate.py 和 compare_models.py 都能直接用。

HEARTBEAT_SECONDS = 30


@contextlib.contextmanager
def long_call(tag, interval=HEARTBEAT_SECONDS):
    """长调用进度提示。用法：

        with llm.long_call(f"[{n}/{total}] {tag}"):
            raw = llm.call_chat(...)

    进入时打印一行"调用中"，等待期间每 interval 秒打印一次已用时，
    退出时打印本次总用时。异常路径也会正常收尾（打印"中断"后原样抛出）。
    """
    print(f"{tag} 调用中…（推理模型单次可能 4–7.5 分钟，请勿中断）", flush=True)
    t0 = time.monotonic()
    stop = threading.Event()

    def _tick():
        # stop.wait 超时返回 False，被 set 唤醒返回 True —— 用 wait 而不是 sleep，
        # 这样调用一结束线程立刻退出，不必等满一个 interval
        while not stop.wait(interval):
            print(f"{tag} ⏳ 已等待 {time.monotonic() - t0:.0f} 秒…", flush=True)

    th = threading.Thread(target=_tick, daemon=True)
    th.start()
    try:
        yield
    except BaseException:
        stop.set()
        th.join(timeout=1)
        print(f"{tag} 中断，用时 {time.monotonic() - t0:.1f} 秒", flush=True)
        raise
    else:
        stop.set()
        th.join(timeout=1)
        print(f"{tag} 返回，用时 {time.monotonic() - t0:.1f} 秒", flush=True)


def call_chat(provider, messages, temperature=0.8, max_tokens=8192,
              timeout=1800, max_retry=3):
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
