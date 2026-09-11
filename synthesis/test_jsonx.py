# -*- coding: utf-8 -*-
"""
test_jsonx.py — jsonx.walk_pairs 回归测试（不依赖网络，直接 python test_jsonx.py 运行）

守的是两类**静默丢数据**的坑（都实际踩过，见 jsonx.walk_pairs 注释）：

    1. 包装层带 instruction/response，真正的数据对在子列表里 —— 只登记包装层那一条。
       实测 GLM 的 evol 输出 4968 字符 / 9 条数据对，只入库了 1 条。
    2. 反过来，"子节点有产出就跳过当前节点" —— 丢掉当前节点自己那条真数据对：

           {"instruction": A, "response": B, "variants": [{"instruction": C, ...}]}

       只留下 C，A/B 无声消失。

丢数据是**无声**的：脚本照常报成功，只是数据变少。故用测试钉死。
"""

import json
import sys

import jsonx

# 控制台默认代码页在中文 Windows 上不是 UTF-8，不重设会把中文打成乱码
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _pairs(obj):
    return jsonx.walk_pairs(obj)


# (用例名, 输入对象, 期望对数, 期望的 instruction 顺序)
CASES = [
    ("纯叶子节点", {"instruction": "单", "response": "答"}, 1, ["单"]),

    ("包装层无 instruction/response",
     {"seed": "x", "variants": [{"instruction": "子", "response": "答"}]},
     1, ["子"]),

    ("包装层是子节点的回显（不应重复登记）",
     {"instruction": "同", "response": "同",
      "data": [{"instruction": "同", "response": "同"}]},
     1, ["同"]),

    ("父节点自有真数据对 + 子节点也有（两条都要留）",
     {"instruction": "父指令", "response": "父回答",
      "variants": [{"instruction": "子指令", "response": "子回答"}]},
     2, ["子指令", "父指令"]),

    ("父节点有数据对但子节点无产出",
     {"instruction": "父指令", "response": "父回答", "meta": {"note": "z"}},
     1, ["父指令"]),

    ("原始 bug 现场：包装层带字段 + 列表里 9 条",
     {"instruction": "包装", "response": "包装答",
      "items": [{"instruction": f"i{i}", "response": f"r{i}",
                 "evolution_type": "深度进化"} for i in range(9)]},
     10, [f"i{i}" for i in range(9)] + ["包装"]),

    ("空值/无效节点不产出",
     {"a": [], "b": {}, "c": None, "d": "str"}, 0, []),

    ("中文键名（指令/响应）+ 每变体带 rewrite_type",
     {"seed_rewrites": [{"original_instruction": "o",
                         "variants": [{"rewrite_type": "同义改写",
                                       "instruction": "变体", "响应": "回答"}]}]},
     1, ["变体"]),

    ("深层嵌套 + 混合无效节点",
     {"a": {"b": [None, 1, {"instruction": "深", "response": "答"}]}},
     1, ["深"]),
]

# category 断言：字段应逐层累积
CATEGORY_CASES = [
    ("rewrite_type 进 category（每变体一个，不能被种子层覆盖）",
     {"seed_rewrites": [
         {"original_instruction": "o",
          "variants": [{"rewrite_type": "同义改写", "instruction": "v1", "response": "r1"},
                       {"rewrite_type": "句式变换", "instruction": "v2", "response": "r2"}]}]},
     ["同义改写", "句式变换"]),

    ("多级字段用 ' | ' 拼接",
     {"task_name": "摘要任务", "items": [
         {"type": "中文", "instruction": "i", "response": "r"}]},
     ["摘要任务 | 中文"]),
]


def main():
    failed = 0

    for name, obj, exp_n, exp_ins in CASES:
        got = _pairs(obj)
        ins = [g["instruction"] for g in got]
        ok = len(got) == exp_n and ins == exp_ins
        failed += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            print(f"         得到 {len(got)} 对 {ins}")
            print(f"         期望 {exp_n} 对 {exp_ins}")

    for name, obj, exp_cats in CATEGORY_CASES:
        got = _pairs(obj)
        cats = [g["category"] for g in got]
        ok = cats == exp_cats
        failed += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            print(f"         得到 {cats}")
            print(f"         期望 {exp_cats}")

    total = len(CASES) + len(CATEGORY_CASES)
    print(f"\n  {total - failed}/{total} 通过")

    # extract_json 的围栏剥离也应回归（模型经常带 ```json 围栏）
    fenced = '废话\n```json\n{"instruction": "x", "response": "y"}\n```\n尾巴'
    obj = jsonx.extract_json(fenced)
    ok = isinstance(obj, dict) and obj.get("instruction") == "x"
    print(f"  [{'PASS' if ok else 'FAIL'}] extract_json 剥离 ```json 围栏与前后废话")
    failed += not ok
    print(f"\n  {'全部通过' if not failed else str(failed) + ' 项失败'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
