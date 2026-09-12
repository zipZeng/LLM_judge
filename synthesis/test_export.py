# -*- coding: utf-8 -*-
"""export.py 的离线回归测试 —— 不联网、不读 API、不花 token。

    python test_export.py

钉死两类踩过的坑：

A. **规则校验的作用域**。第一版用字面子串匹配，45 条报 9 条违规/7 条误报；
   扩到 512 条报 10 条/6 条误报。误报全部同因：约束管的是**子部件**不是整篇。
   下面每条 SCOPE_CASES 都是真实数据里的原句，必须判「未核验」而不是「违规」。

B. **近重复丢弃必须报最相似的那一条**。第一版 `break` 在第一个命中，
   于是「用不超过 5 行的 Python 代码判断一个整数是否为素数」被报成
   「近重复 1.00 于『怎么用 Python 判断一个整数是不是素数？』」—— 字面并不相同。
   审计日志报错对照物 = 审计不可信。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export  # noqa: E402

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


# ============ A. 作用域：这些必须判「未核验」，绝不能报违规 ============
# 每条都是真实生成数据里的原句，括号里写明约束实际管的是哪个部件。
SCOPE_CASES = [
    ("限的是 principle 字段",
     "请用Python实现冒泡排序，输出格式为JSON，包含code字段（代码字符串）和"
     "principle字段（原理说明，不超过100字）。",
     '{"code": "def bubble_sort(arr):\\n    ...", "principle": "相邻元素两两比较，大的后移"}'),
    ("限的是每个要点（且窗口里还有『整体』）",
     "将这篇科技新闻压缩为3个要点，每个要点不超过20字，整体作为摘要输出。",
     "1. 发布新一代AI芯片。\n2. 算力升三倍功耗降四成。\n3. 明年量产降低AI成本。"),
    ("限的是需求摘要那一句",
     "假设你是资深产品需求分析师，请对以下用户需求进行多层次分析。输出要求："
     "1) 需求摘要：不超过50字；2) 需求分类：按'功能需求/非功能需求/约束条件'三类标注；"
     "3) 优先级判断：用KANO模型标注。",
     "1) 需求摘要（不超过50字）：\n寻找一款免费或低价、界面简洁、支持离线与Markdown的"
     "笔记软件，期望免费跨设备同步。\n\n2) 需求分类：\n- 功能需求：离线使用、Markdown支持"),
    ("限的是时间复杂度分析那一节",
     "写一个Python冒泡排序函数，要求时间复杂度分析不超过100字，代码需包含异常处理。",
     "def bubble_sort(arr):\n    if not isinstance(arr, list):\n        raise TypeError()\n"
     "    return arr\n\n时间复杂度：最好O(n)，最坏O(n²)，平均O(n²)，空间O(1)，"
     "稳定性好，适合小规模数据，工程上建议改用内置排序。"),
    ("限的是新闻摘要那一段",
     "先提取新闻的5W1H要素，再基于要素生成一段不超过100字的新闻摘要。",
     "5W1H：Who(某公司), What(发布新药), When(昨日), Where(全球), Why(攻克罕见病), "
     "How(临床试验成功)。\n摘要：某公司昨日宣布攻克罕见病新药研发成功，"
     "临床试验数据优异。该突破为百万患者带来新希望，预计明年上市。"),
    ("限的是末尾『总结』那一句",
     "理解并总结用户需求，要求：1. 识别显性需求与隐性需求 2. 按重要性排序 "
     "3. 输出格式为表格 4. 用不超过100字总结",
     "| 需求类型 | 内容 | 重要性 |\n|---|---|---|\n| 显性需求 | 准确理解用户表达 | 高 |\n"
     "总结：用户希望系统精准理解并高效总结其需求，优先保证准确性与响应速度。"),
]

print("=" * 70)
print("A. 作用域 —— 约束限定在子部件上，必须记「未核验」而非「违规」")
print("=" * 70)
for name, ins, out in SCOPE_CASES:
    v, b, u = export.check_constraints(ins, out)
    check(f"{name}：不报违规", v, [])
    check(f"{name}：记 1 处未核验", u, 1)


# ============ B. 字数口径：含/不含标点两种算法 ============
print()
print("=" * 70)
print("B. 字数口径 —— 含标点超但去标点达标 ⇒ 边界存疑（保留），不算违规")
print("=" * 70)
INS30 = "用不超过 30 字概括下面这段话的主要观点：远程办公提高了员工的时间灵活性。"
OUT_OK = "远程办公提升灵活性却模糊工作生活边界。"                 # 18 字，怎么数都过
OUT_EDGE = "远程办公提升灵活性却模糊工作生活边界，企业需建立清晰沟通规范。"  # 含标点 31、去标点 29
OUT_BAD = "远程办公虽然提升了员工的时间灵活性，但同时也模糊了工作与生活的边界，"\
          "长期来看企业需要建立更清晰的沟通规范来应对这些问题。"       # 怎么数都超

check("整篇约束、回答达标 → 无违规",
      export.check_constraints(INS30, OUT_OK)[0], [])
check("整篇约束、回答达标 → 无存疑",
      export.check_constraints(INS30, OUT_OK)[1], [])
check("整篇约束、含标点超/去标点达标 → 不报违规（口径有歧义）",
      export.check_constraints(INS30, OUT_EDGE)[0], [])
check("整篇约束、含标点超/去标点达标 → 记为边界存疑",
      len(export.check_constraints(INS30, OUT_EDGE)[1]), 1)
check("整篇约束、两种口径都超 → 报违规",
      len(export.check_constraints(INS30, OUT_BAD)[0]), 1)
check("『用不超过30字概括…』不含部件词，应判为整篇约束",
      export.check_constraints(INS30, OUT_BAD)[2], 0)

check("去标点计数：31 字含标点 → 29 字不含标点",
      export._count_cjk(OUT_EDGE, False), 29)
check("含标点计数：31",
      export._count_cjk(OUT_EDGE, True), 31)


# ============ C. 行数与 JSON 格式 ============
print()
print("=" * 70)
print("C. 行数约束 与 JSON 格式约束")
print("=" * 70)
check("要求 3 行、写了 5 行 → 违规",
      len(export.check_constraints("请分 3 行输出，不超过 3 行。", "a\nb\nc\nd\ne")[0]), 1)
check("要求 5 行、写了 3 行 → 通过",
      export.check_constraints("请输出，不超过 5 行。", "a\nb\nc")[0], [])

JSON_INS = ("理解并总结用户需求，要求以JSON格式输出，字段为：background, explicit_needs, "
            "implicit_needs, priorities, next_steps。")
JSON_BAD = ("首先分析用户需求，提取背景信息。然后列出显性需求，再推断隐性需求。"
            "最后提供可操作的建议。输出JSON格式，确保字段完整、内容准确。")
JSON_OK = ('{"background": "使用产品遇阻", "explicit_needs": ["解决登录失败"], '
           '"implicit_needs": ["快速恢复"], "priorities": ["登录"], "next_steps": ["排查"]}')
check("要求 JSON、回答是散文（真实违规案例）→ 报违规",
      len(export.check_constraints(JSON_INS, JSON_BAD)[0]), 1)
check("要求 JSON、回答是合法 JSON → 通过",
      export.check_constraints(JSON_INS, JSON_OK)[0], [])


# ============ D. 硬去重 / 近重复去重 ============
print()
print("=" * 70)
print("D. 去重")
print("=" * 70)


def rec(ins, res, f="f.jsonl"):
    return {"instruction": ins, "response": res, "paradigm": "p",
            "category": "c", "source_model": "m", "_file": f}


same = [rec("问", "答"), rec("问", "答"), rec("问", "答")]
kept, dropped = export.exact_dedup(same)
check("整对逐字相同 ×3 → 只留 1 条", len(kept), 1)
check("整对逐字相同 ×3 → 丢 2 条", len(dropped), 2)

# 同指令不同回答：这正是「同模板同种子重复运行」的产物，必须判重复
runs = [rec("为什么天空是蓝色的？", "天空呈蓝色是阳光大气散射的结果。" * 3),
        rec("为什么天空是蓝色的？", "天空呈现蓝色是因为瑞利散射。" * 3)]
kept, dropped = export.minhash_dedup(runs, 0.8)
check("同指令、不同回答 → 判为重复", len(dropped), 1)

# 近重复必须报【最相似】的那条，不是扫描中第一个命中的
base = "请把下面这段话翻译成英文：今天天气很好，我们一起去公园散步吧。"
mut = [base, base.replace("很好", "不错"), base.replace("散步吧。", "散步吧！"),
       base + "请直接给出译文。", base.replace("公园", "校园"),
       base.replace("今天", "昨日").replace("很好", "尚可")]
kept, dropped = export.minhash_dedup([rec(x, "答" * 40) for x in mut], 0.5)
bad = [d for d in dropped
       if f"{max(export.jaccard_estimate(export.minhash_signature(d[0]['instruction']),
                                         export.minhash_signature(k['instruction']))
                for k in kept):.2f}" not in d[1]]
check(f"丢弃原因报的是最相似项（{len(dropped)} 条被丢，逐条核对）", bad, [])


# ============ E. MinHash 本身 ============
print()
print("=" * 70)
print("E. MinHash 定标")
print("=" * 70)
check("逐字相同 → 1.00",
      export.jaccard_estimate(export.minhash_signature("完全一样的句子"),
                              export.minhash_signature("完全一样的句子")), 1.0)
check("毫不相干 → 低相似",
      export.jaccard_estimate(export.minhash_signature("红烧肉的做法大全"),
                              export.minhash_signature("Python 异步编程最佳实践")) < 0.3, True)
check("签名长度 = 置换数", len(export.minhash_signature("任意文本")), export.NUM_PERM)
check("同一文本两次签名一致（可复现）",
      export.minhash_signature("复现性检查") == export.minhash_signature("复现性检查"), True)

# 换行/空格不影响签名（_shingles 先去掉所有空白）
check("空白不影响签名",
      export.minhash_signature("a b\nc") == export.minhash_signature("abc"), True)


print()
print("=" * 70)
print(f"结果：{PASS} 通过 / {FAIL} 失败")
print("=" * 70)
sys.exit(1 if FAIL else 0)
