# -*- coding: utf-8 -*-
"""llm.py 的离线回归测试 —— 不联网、不读 API Key、不花 token。

    python test_llm.py

目前只覆盖 `long_call()` 进度提示。它值得单测，因为：

1. **它靠后台线程打印**。线程的收尾如果写错，最坏情况是主流程已经退出、
   线程还在往一个关掉的控制台里写 —— 报错信息会出现在完全无关的地方，
   极难查。所以必须钉死"退出后不留线程"。
2. **它包着真正的 API 调用**，写错会把异常吞掉或换成别的异常，
   让上层 `except` 拿到错误的失败原因（比不显示进度危险得多）。
3. 它的默认间隔是个**人肉参数**（30 秒），有人顺手调成 1 秒就会把日志刷烂。

真正的网络调用不在这里测 —— 那些要花钱，且 llm.py 的正确性主要靠
`--dry-run` 与真实运行验证。
"""

import contextlib
import inspect
import io
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PASS = FAIL = 0


def check(name, got, want):
    global PASS, FAIL
    ok = got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        print(f"         期望 {want!r}\n         实际 {got!r}")
    PASS, FAIL = PASS + ok, FAIL + (not ok)


def capture(fn):
    """跑 fn()，返回 (打印文本, 抛出的异常或 None)。"""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            fn()
    except BaseException as e:          # noqa: BLE001 —— 测试要如实看到任何异常
        return buf.getvalue(), e
    return buf.getvalue(), None


def wait_threads_gone(baseline, timeout=2.0):
    """等后台线程收干净；返回最终线程数。"""
    end = time.monotonic() + timeout
    while threading.active_count() > baseline and time.monotonic() < end:
        time.sleep(0.02)
    return threading.active_count()


class _Boom(RuntimeError):
    pass


# ============================ 被测的几段代码 ============================
# ⚠ 必须定义在这里、在下面用到之前 —— 第一版把它们写在 A 组之后，
#    `capture(lambda: _run_ok())` 直接 NameError。


def _run_ok():
    with llm.long_call("[3/9] [生成 glm]", interval=60):
        pass


def _run_short():
    with llm.long_call("[1/2] [评审 qwen]", interval=60):
        time.sleep(0.05)          # 远小于 interval


def _run_beat():
    with llm.long_call("[2/2] [评审 glm]", interval=0.1):
        time.sleep(0.35)          # 应至少打 2 次


def _run_raise():
    with llm.long_call("[9/9] [生成 deepseek]", interval=60):
        raise _Boom("模型返回空内容")


def _run_keyboard():
    with llm.long_call("[1/1] [生成 glm]", interval=60):
        raise KeyboardInterrupt


print("=" * 70)
print("A. 基本形态：进入打印开始，退出打印用时")
print("=" * 70)

out, err = capture(_run_ok)
if err is not None:
    print(f"  [FAIL] 正常路径不应抛异常，实际抛出 {err!r}")
    FAIL += 1
else:
    check("进入时打印了 tag 与『调用中』", "[3/9] [生成 glm] 调用中…" in out, True)
    check("提示里写明了可能耗时（避免被误当卡死 Ctrl-C）", "4–7.5 分钟" in out, True)
    check("退出时打印『返回，用时』", "返回，用时" in out, True)
    check("用时带秒数", "秒" in out.split("返回，用时")[1][:20], True)


print()
print("=" * 70)
print("B. 心跳：等待超过 interval 才打印，没过就不打印")
print("=" * 70)

out, err = capture(_run_short)
check("块很快结束 → 不打心跳（日志不被刷屏）", "⏳" in out, False)

out, err = capture(_run_beat)
n_beat = out.count("⏳")
check("等待超过 interval → 打出心跳", n_beat >= 2, True)
check("心跳里带 tag 与『已等待』", "[2/2] [评审 glm] ⏳ 已等待" in out, True)


print()
print("=" * 70)
print("C. 异常路径：原样抛出，不能被进度提示吃掉或换掉")
print("=" * 70)

out, err = capture(_run_raise)
check("异常类型未被替换", type(err), _Boom)
check("异常消息未被替换", str(err), "模型返回空内容")
check("异常路径也打印了收尾行", "中断，用时" in out, True)
# 上层 generate.py / compare_models.py 靠 str(e) 写错误日志，
# 换掉异常等于让日志说谎 —— 这条是这组测试存在的首要理由
check("异常路径不说『返回』（会让人以为调用成功了）", "返回，用时" in out, False)

out, err = capture(_run_keyboard)
check("Ctrl-C 也能正常收尾（不被当成成功）", isinstance(err, KeyboardInterrupt), True)


print()
print("=" * 70)
print("D. 不留后台线程（写错的话报错会跑到无关的地方去）")
print("=" * 70)

def _run_raise_thread():
    try:
        with llm.long_call("[i/n] [生成 glm]", interval=0.05):
            time.sleep(0.12)
            raise _Boom("x")
    except _Boom:
        pass


def _run_many(fn):
    """跑 5 次；外面套 capture，否则这里的心跳会直接打到真实控制台。"""
    for _ in range(5):
        fn()


def _run_ok_once():
    with llm.long_call("[i/n] [生成 glm]", interval=0.05):
        time.sleep(0.12)


base = threading.active_count()
capture(lambda: _run_many(_run_ok_once))
check("连跑 5 次后线程数回到基线", wait_threads_gone(base), base)

base = threading.active_count()
capture(lambda: _run_many(_run_raise_thread))
check("异常路径同样不留线程", wait_threads_gone(base), base)


# ⚠ 第一版这里写的是 `threading.Thread(target=...).daemon` —— 那是**新建一个线程**
# 看它的默认值（恒为 False），跟 llm 里那个线程毫无关系，等于没测。
# 要在 with 块**内部**去看活着的线程。
_daemons = []


def _run_daemon_probe():
    with llm.long_call("[1/1] [生成 glm]", interval=0.05):
        time.sleep(0.12)
        _daemons[:] = [t.daemon for t in threading.enumerate()
                       if t is not threading.main_thread()]


capture(_run_daemon_probe)
check("阻塞期间确实存在后台线程（否则上面几条是空测）", len(_daemons) >= 1, True)
check("且它是 daemon（主流程退出不被挂住）", all(_daemons), True)


print()
print("=" * 70)
print("E. 默认参数（人手容易顺手改坏的地方）")
print("=" * 70)

check("默认间隔仍是 30 秒", inspect.signature(llm.long_call)
      .parameters["interval"].default, 30)
check("HEARTBEAT_SECONDS 未被改小", llm.HEARTBEAT_SECONDS, 30)


print()
print("=" * 70)
print(f"结果：{PASS} 通过 / {FAIL} 失败")
print("=" * 70)
sys.exit(1 if FAIL else 0)
