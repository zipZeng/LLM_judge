# -*- coding: utf-8 -*-
"""compare_models.py 的离线回归测试 —— 不联网、不读 API、不花 token。

    python test_compare.py

重点是**两个极易写错的公式**和 `--from-report` 重放：

1. **自评偏好 = 自评分 − 他人均分**，不是「全量均分 − 非自评均分」。
   后者减出来只有真值的一半左右 —— 实测 09-11 DeepSeek 真值 −0.63
   （自评 8.25 − 他人均分 8.88），错法算出 −0.21。写错的话它会**看起来像个合理的数**，
   不会报错，所以只能靠测试钉死。

2. **表里那列是「全量均分」（含自评），不是自评分**。原表头写「自评均分」是错的：
   09-11 DeepSeek 全量 8.67，但它给自己打的是 8.25。

断言用的期望值取自两份真实存档（data/compare/20260910_151109、20260911_104142），
与交接文档记录的实测数字一致 —— 即测试同时校验了「公式对不对」和「文档数字可复现」。

（`--from-report` 本身不需要 API Key，本测试也不碰网络。）
"""

import contextlib
import io
import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_models as cm  # noqa: E402

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


def run_report(votes, models):
    """调用 render_report 并吞掉它的打印，返回 (ranking_data, 打印文本)。"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rd = cm.render_report(models, list(votes.keys()), votes, source="(test)")
    return rd, buf.getvalue()


# ============ 真实存档数据（原样抄自 summary.json 的 votes）============
V0911 = {
    "deepseek": {"deepseek": 8.25, "kimi": 9.60, "qwen": 8.40},
    "kimi":      {"deepseek": 8.75, "kimi": 9.50, "qwen": 7.75},
    "qwen":     {"deepseek": 9.00, "kimi": 9.50, "qwen": 7.25},
}
V0910 = {
    "deepseek": {"deepseek": 7.90, "kimi": 8.85, "qwen": 8.60},
    "kimi":      {"deepseek": 8.25, "kimi": 9.00, "qwen": 8.50},
    "qwen":     {"deepseek": 8.50, "kimi": 9.75, "qwen": 8.75},
}
MODELS = ["deepseek", "kimi", "qwen"]

print("=" * 70)
print("A. 自评偏好公式 —— 必须是「自评分 − 他人均分」")
print("=" * 70)
rd, out = run_report(V0911, MODELS)
# 期望值来自交接文档记录的实测：DeepSeek −0.63 / Kimi −0.05 / Qwen −0.82
pref = {g: round(ss - nsm, 2) for g, nsm, allm, ss in rd}
check("09-11 DeepSeek 自评偏好 = −0.63", pref["deepseek"], -0.63)
check("09-11 Kimi 自评偏好 = −0.05", pref["kimi"], -0.05)
check("09-11 Qwen 自评偏好 = −0.82", pref["qwen"], -0.82)
check("打印的自评偏好与计算一致（含自评分）",
      "DeepSeek\t-0.63（自评 8.25）" in out, True)

rd2, _ = run_report(V0910, MODELS)
pref2 = {g: round(ss - nsm, 2) for g, nsm, allm, ss in rd2}
check("09-10 DeepSeek 自评偏好 = −0.48", pref2["deepseek"], -0.48)
check("09-10 Kimi 自评偏好 = −0.30", pref2["kimi"], -0.30)
check("09-10 Qwen 自评偏好 = +0.20（唯一为正的一次）", pref2["qwen"], 0.20)

# 反例：确认错法（全量 − 非自评）会给出不同的值，否则这个测试是空的
wrong = {g: round(allm - nsm, 2) for g, nsm, allm, ss in rd}
check("错法确实会算出不同的数（证明本测试有效）",
      wrong["deepseek"] != -0.63, True)


print()
print("=" * 70)
print("B. 「全量均分」不是「自评分」")
print("=" * 70)
check("表头写的是「全量均分」", "全量均分" in out, True)
# ⚠ 不能直接断言 `"自评均分" not in out` —— 「非自评均分」里**包含**「自评均分」这个子串，
# 那样写永远为假。要按列边界匹配：`| 自评均分` 这种"紧跟分隔符"的形态才算真的表头。
check("表头不再把这一列叫「自评均分」",
      re.search(r"\|\s*自评均分", out) is None, True)
check("列序为 … 全量均分 | 非自评均分*",
      re.search(r"全量均分\s*\|\s*非自评均分\*", out) is not None, True)
allm = {g: x for g, nsm, x, ss in rd}
check("09-11 DeepSeek 全量均分 = 8.67", allm["deepseek"], 8.67)
check("而它给自己的分是 8.25（两者不同，故表头必须区分）",
      dict((g, ss) for g, nsm, x, ss in rd)["deepseek"], 8.25)
check("非自评均分 = 8.88", {g: n for g, n, x, s in rd}["deepseek"], 8.88)


print()
print("=" * 70)
print("C. 评委尺度")
print("=" * 70)
scale = {j: cm._mean([V0911[j][g] for g in MODELS]) for j in MODELS}
check("09-11 DeepSeek 评委 8.75（最松）", scale["deepseek"], 8.75)
check("09-11 Qwen 评委 8.58（最严）", scale["qwen"], 8.58)
scale10 = {j: cm._mean([V0910[j][g] for g in MODELS]) for j in MODELS}
check("09-10 Qwen 评委 9.0（最松）", scale10["qwen"], 9.0)
check("09-10 DeepSeek 评委 8.45（最严）", scale10["deepseek"], 8.45)
check("两次的宽松/严格排序确实翻转了（文档里说这是稳健性证据）",
      scale["deepseek"] > scale["qwen"] and scale10["qwen"] > scale10["deepseek"], True)


print()
print("=" * 70)
print("D. load_summary 与重放入口")
print("=" * 70)
tmp = Path(tempfile.mkdtemp(prefix="cmtest_"))
summary = {"seeds": ["a", "b"], "per_seed": 2,
           "generated_counts": {"deepseek": 4, "kimi": 4, "qwen": 4},
           "judges": MODELS, "votes": V0911}
(tmp / "summary.json").write_text(json.dumps(summary, ensure_ascii=False),
                                  encoding="utf-8")

d, p = cm.load_summary(str(tmp))
check("传目录 → 自动找 summary.json", p.name, "summary.json")
check("读回的 votes 正确", d["votes"], V0911)
d2, p2 = cm.load_summary(str(tmp / "summary.json"))
check("传文件路径也可以", p2, tmp / "summary.json")

try:
    cm.load_summary(str(tmp / "不存在"))
    check("路径不存在 → 报错退出", "没退出", "SystemExit")
except SystemExit:
    check("路径不存在 → 报错退出", True, True)


def run_main(argv):
    """跑一次 main()，返回 (stdout 文本, 是否 SystemExit)。"""
    buf, old = io.StringIO(), sys.argv
    sys.argv = ["compare_models.py"] + argv
    try:
        with contextlib.redirect_stdout(buf):
            cm.main()
        return buf.getvalue(), False
    except SystemExit:
        return buf.getvalue(), True
    finally:
        sys.argv = old


# 端到端：让 check_available 直接炸。重放若在任何 Key 检查之前返回，就不会炸到 ——
# 这正好证明「演示机器没配 .env 也能重放」。
def _boom(*a, **k):
    raise AssertionError("重放路径不应触达 API Key 检查！")


_orig = cm.llm.check_available
cm.llm.check_available = _boom
try:
    out, exited = run_main(["--from-report", str(tmp)])
finally:
    cm.llm.check_available = _orig
check("重放不触达 Key 检查（无需 .env 即可演示）", exited, False)
check("重放打印了对比表", "生成模型质量对比" in out, True)
check("重放打印了存档路径", "重放存档" in out, True)
check("重放打印了只读声明", "只读重放" in out, True)
check("重放不写文件（目录里仍只有 summary.json）",
      sorted(q.name for q in tmp.iterdir()), ["summary.json"])

# 空 votes（失败运行留下的存档）要给出可读提示，而不是打一张空表
(tmp / "empty").mkdir()
(tmp / "empty" / "summary.json").write_text('{"votes": {}}', encoding="utf-8")
out, exited = run_main(["--from-report", str(tmp / "empty")])
check("空 votes 的存档 → 报错退出而不是打空表", exited, True)

# 缺 generated_counts 的旧存档要能从 votes 里推回行序
old = {"judges": MODELS, "votes": V0911}
gen = list((old.get("generated_counts") or {}).keys()) \
    or list({g for v in old["votes"].values() for g in v})
check("旧存档缺 generated_counts 时可回退出生成模型",
      sorted(gen), ["deepseek", "kimi", "qwen"])


print()
print("=" * 70)
print(f"结果：{PASS} 通过 / {FAIL} 失败")
print("=" * 70)
sys.exit(1 if FAIL else 0)
