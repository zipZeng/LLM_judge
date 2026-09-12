# -*- coding: utf-8 -*-
"""
export.py — 精选数据集导出 + 数据卡生成

把 data/syn/ 下所有生成结果汇总成**一份可交付的精选数据集**，并据实生成数据卡。
对应需求「核心功能模块 5：导出与版本（附数据卡：来源/构成/已知偏差/适用场景）」
与验证方案「重复率（MinHash 0.8 阈值）< 5%」。

三步：
    1. 汇总   —— 读全部 syn_data_*.jsonl
    2. 去重   —— 逐字硬去重 + MinHash 近重复（**按指令判定，不按回答**）
    3. 校验   —— 字数/行数/格式约束的机械核验，剔除不合格项

## 去重口径（关键，别改错）

**按 instruction 算 MinHash，不把 response 拼进去。** 原因：种子改写增强的设计本意
就是"同一任务换多种问法、回答相同是正确的"（见 prompts/seed_rewrite.md 注意事项 7）。
把 response 拼进去会让这类**正常变体**拿到 0.74 的高相似度，被当成重复删掉。

实测印证：一对同义改写变体「请写一封向领导申请调休的邮件」/「我需要向经理申请调休一天，
能帮我写一封请假邮件吗？」—— 仅指令相似度 0.11（判为不同任务，保留），但整对相似度 0.74
（会被误删）。**所以只算 instruction。**

另做一道**逐字硬去重**：整对 (instruction, response) 完全相同的一律只留一条 ——
这是"同模板同种子重复运行"产生的真重复，与上面的口径无关。

## 规则校验

只核验**可机械核验**的约束，且**按作用域**解析。这里踩过两次同样的坑：

    第一版（45 条）报了 9 条违规，7 条误报；
    扩到 512 条又报 10 条，6 条误报 —— 误报全部同因：
    **约束管的是子部件，不是整篇。**
        「principle字段（原理说明，不超过100字）」→ 限的是那个字段
        「每个要点不超过20字」                    → 限的是每个要点
        「1) 需求摘要：不超过50字」               → 限的是摘要那句
        「时间复杂度分析不超过100字」             → 限的是分析那节
        「生成一段不超过100字的新闻摘要」         → 限的是摘要那段

所以本版只认两种：约束**明确管整篇**才核验；一旦发现限定在子部件上，就记「未核验」
并计入 `n_unverifiable`，由数据卡如实披露。**子部件切不干净，宁可标未核验也不误报。**

字数含不含标点在中文里有歧义，故两种都算：两种都超才算**确定违规**（剔除），
只有含标点超的记为**边界存疑**（保留并在日志标注）。

用法：
    python export.py                  # 汇总 → 去重 → 校验 → 写数据集与数据卡
    python export.py --dry-run        # 只统计与报告，不写文件
    python export.py --keep-invalid   # 不剔除规则校验不通过的条目
    python export.py --input-dir ... --output-dir ...
"""

import argparse
import glob
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# MinHash 参数
NUM_PERM = 128
_MERSENNE = (1 << 61) - 1
_SHINGLE_K = 3


def _ensure_utf8_stdout():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


# ============================== MinHash ==============================

def _permutation_coeffs(n):
    """由固定种子生成 n 组 (a, b)，保证每次运行结果一致（否则无法复现）。"""
    out = []
    for i in range(n):
        h = hashlib.blake2b(f"perm{i}".encode(), digest_size=16).digest()
        a = int.from_bytes(h[:8], "big") | 1        # 奇数，保证可逆
        b = int.from_bytes(h[8:], "big")
        out.append((a, b))
    return out


_PERMS = _permutation_coeffs(NUM_PERM)


def _shingles(text, k=_SHINGLE_K):
    """字符 k-gram 集合（去掉所有空白后切）。中文按字切，无需分词。"""
    t = "".join(text.split())
    if not t:
        return set()
    return {t[i:i + k] for i in range(max(len(t) - k + 1, 1))}


def minhash_signature(text):
    """标准 MinHash 签名：单个哈希 + 线性置换取最小。"""
    sh = _shingles(text) or {text}
    hashes = [int(hashlib.blake2b(s.encode("utf-8"), digest_size=8).hexdigest(), 16)
              for s in sh]
    sig = []
    for a, b in _PERMS:
        sig.append(min((a * x + b) % _MERSENNE for x in hashes))
    return sig


def jaccard_estimate(sig_a, sig_b):
    """由两个签名估计 Jaccard 相似度。"""
    return sum(x == y for x, y in zip(sig_a, sig_b)) / len(sig_a)


# ============================== 去重 ==============================

def exact_dedup(records):
    """整对 (instruction, response) 逐字相同 → 只留第一条（按输入顺序）。"""
    seen, kept, dropped = set(), [], []
    for r in records:
        key = (r["instruction"], r["response"])
        if key in seen:
            dropped.append((r, "整对逐字重复"))
        else:
            seen.add(key)
            kept.append(r)
    return kept, dropped


def minhash_dedup(records, threshold=0.8):
    """按 **instruction** 的 MinHash 近重复去重；保留首次出现的那条。

    贪心：新条目一旦与已保留的任一条相似度 >= threshold，即判为重复丢弃。

    ⚠ 丢弃原因里必须报**最相似**的那一条，不能报第一条命中的。第一版用 break，
    结果「用不超过 5 行的 Python 代码判断一个整数是否为素数」被报成
    「近重复 1.00 于『怎么用 Python 判断一个整数是不是素数？』」—— 这两句字面
    并不相同，真正 1.00 的那条在列表更后面。审计日志报错对照物 = 审计不可信。

    ⚠ 复杂度 O(n²·置换数)。500 条量级约 1 秒，够用；上万条时需换 LSH 分桶。
    """
    kept, dropped, sigs = [], [], []
    for r in records:
        sig = minhash_signature(r["instruction"])
        best, best_i = 0.0, -1
        for i, ksig in enumerate(sigs):
            s = jaccard_estimate(sig, ksig)
            if s > best:
                best, best_i = s, i
        if best >= threshold:
            dropped.append((r, f"指令近重复 {best:.2f} 于「{kept[best_i]['instruction'][:24]}…」"
                               f"（{kept[best_i]['_file'][15:23]}）"))
        else:
            sigs.append(sig)
            kept.append(r)
    return kept, dropped


# ============================== 规则校验 ==============================

def _count_cjk(out, with_punct):
    """字数。中文语境下「不超过 N 字」含不含标点本身有歧义，故两种都算。"""
    t = out.strip()
    if with_punct:
        return len("".join(t.split()))
    return len(re.sub(r"[\s\W_]", "", t, flags=re.UNICODE))


def _near(ins, m, span=14):
    """取约束出现位置前后 span 个字符，用来判断它限定的是哪个部件。"""
    return ins[max(0, m.start() - span):m.end() + span]


# 这几个词说明约束管的是**整篇**
_WHOLE = ("整体", "全文", "总字数", "总共", "全篇", "完整")
# 这几个词说明约束只管**某个部件** —— 部件要切出来才能数，切不干净就不该报
_PART = ("要点", "摘要", "分析", "正文", "注释", "标题", "备注", "说明", "总结",
         "描述", "解释", "理由", "结论", "建议", "需求", "字段", "段落",
         "每", "各", "分为", "分别")


def check_constraints(ins, out):
    """机械核验可核验的约束。

    返回 (violations, borderline, n_unverifiable)：
        violations      —— 确定违规（两种字数口径都超，或格式没遵守）
        borderline      —— 边界存疑（含标点超、不含标点不超；口径本身有歧义）
        n_unverifiable  —— 约束限定在某个部件上、无法机械切出来核验的条数

    ⚠ **核验必须按作用域解析。** 第一版用字面子串匹配，45 条里报了 9 条违规、
    7 条是误报；扩到 512 条后又报了 10 条、6 条是误报，误报全部同因——约束管的
    是子部件而非整篇：
        「principle字段（原理说明，不超过100字）」→ 限的是那个字段
        「每个要点不超过20字」                    → 限的是每个要点
        「1) 需求摘要：不超过50字」               → 限的是摘要那句
        「时间复杂度分析不超过100字」             → 限的是分析那节
        「生成一段不超过100字的新闻摘要」         → 限的是摘要那段
        「4. 用不超过100字总结」                  → 限的是总结那句
    子部件切不干净，**宁可记「未核验」也不误报**。数据卡里会如实写未核验条数。
    """
    violation, borderline = [], []
    n_unverifiable = 0

    def scope_of(m):
        # ⚠ 先判「部件」再判「整篇」。反过来的话，
        # 「将新闻压缩为3个要点，每个要点不超过20字，整体作为摘要输出」会因窗口里
        # 有"整体"二字被判成整篇约束 —— 而它管的是每个要点。部件词只要在窗口里，
        # 就无法确定约束边界，一律按部件处理（宁可标未核验）。
        w = _near(ins, m)
        if any(k in w for k in _PART):
            return "part"
        if any(k in w for k in _WHOLE):
            return "whole"
        return "whole"

    for m in re.finditer(r"不超过\s*(\d+)\s*行", ins):
        if scope_of(m) == "part":
            n_unverifiable += 1
            continue
        n = len([l for l in out.split("\n") if l.strip()])
        if n > int(m.group(1)):
            violation.append(f"限{m.group(1)}行，实际{n}行")

    for m in re.finditer(r"不超过\s*(\d+)\s*字", ins):
        if scope_of(m) == "part":
            n_unverifiable += 1
            continue
        lim = int(m.group(1))
        hard, soft = _count_cjk(out, True), _count_cjk(out, False)
        if soft > lim:
            violation.append(f"限{lim}字，实际{hard}字（不含标点{soft}字）")
        elif hard > lim:
            borderline.append(f"限{lim}字：不含标点{soft}字达标，含标点{hard}字超出")

    # 格式：指令明确要求 JSON 输出，就必须真能解析
    if re.search(r"JSON|json", ins) and re.search(r"输出|呈现|格式", ins):
        try:
            json.loads(out.strip())
        except Exception:
            violation.append("要求 JSON 输出，但回答里没有可解析的 JSON")

    return violation, borderline, n_unverifiable


# ============================== 数据卡 ==============================

def build_data_card(stats):
    """据实生成数据卡（Markdown）。「适用场景」为草稿，需人工确认。"""
    s = stats
    L = []
    L.append("# 数据卡（Data Card）— 合成指令数据集\n")
    L.append(f"> 生成时间：{s['generated_at']}　|　版本：`{s['version']}`\n")
    L.append("本数据卡按选题 41 需求「核心功能模块 5」要求，说明数据的"
             "**来源 / 构成 / 已知偏差 / 适用场景**。\n")

    L.append("## 一、来源\n")
    L.append(f"- **生成方式**：大模型合成（非人工标注），由 `synthesis/generate.py` 调用云端模型产生")
    L.append(f"- **来源文件**：`data/syn/` 下 {s['n_files']} 个 `syn_data_*.jsonl`")
    L.append(f"- **生成模型**：" + "、".join(f"`{m}`（{n} 条）" for m, n in s["by_model"].most_common()))
    L.append(f"- **合成范式**：" + "、".join(f"`{m}`（{n} 条）" for m, n in s["by_paradigm"].most_common()))
    L.append(f"- **提示词模板**：`synthesis/prompts/*.md`（随模块入库，可复现）")
    L.append(f"- **生成时间跨度**：" + "、".join(sorted(s["file_dates"])))
    L.append("")

    L.append("## 二、构成\n")
    L.append(f"- **精选后条数**：**{s['n_final']}** 条（每行一个 JSON 对象，UTF-8）")
    L.append(f"- **原始汇总**：{s['n_raw']} 条 → 逐字去重 −{s['n_exact_dup']} → "
             f"指令近重复去重 −{s['n_near_dup']} → 规则校验剔除 −{s['n_invalid']}"
             + ("（本次未剔除，见 `--keep-invalid`）" if s["keep_invalid"] else ""))
    L.append(f"- **字段**：`paradigm` / `category` / `source_model` / `instruction` / `response`")
    L.append("")
    L.append("**范式 × 模型分布：**\n")
    models = [m for m, _ in s["by_model"].most_common()]
    L.append("| 范式 | " + " | ".join(models) + " | 合计 |")
    L.append("|---" * (len(models) + 2) + "|")
    for p, _ in s["by_paradigm"].most_common():
        row = [str(s["cross"].get((p, m), 0)) for m in models]
        L.append(f"| `{p}` | " + " | ".join(row) + f" | {sum(int(x) for x in row)} |")
    L.append("")
    L.append(f"- **回答长度**：中位 {s['len_med']} 字符，均值 {s['len_avg']:.0f}，"
             f"最短 {s['len_min']}，最长 {s['len_max']}")
    L.append(f"- **指令长度**：中位 {s['ins_med']} 字符")
    L.append(f"- **重复率**：MinHash（{NUM_PERM} 置换，{_SHINGLE_K}-gram）阈值 0.8 下，"
             f"去重前 {s['raw_dup_rate']:.1f}%（{s['n_raw']} 条）→ 去重后 "
             f"**{s['dup_rate']:.1f}%**（{s['n_final']} 条，需求验证方案要求 < 5%）")
    L.append("  去重前的重复主要来自「同模板同种子重复运行」，不是模型输出缺陷；"
             "明细见 `export_*.log`。")
    L.append("")

    L.append("## 三、已知偏差\n")
    L.append("**以下均为实测，不是估计值。答辩时应主动说明。**\n")
    L.append(f"1. **约束满足率不是 100%**：{s['n_rule_checked']} 条里 "
             f"{s['n_invalid']} 条确定违反自身指令的字数/行数/格式约束"
             f"（占 {s['invalid_rate']:.1f}%），另有 {s['n_borderline']} 条边界存疑"
             f"（含标点超、不含标点不超——中文「不超过 N 字」含不含标点本身有歧义，"
             f"这类已保留并在 `export_*.log` 里标注）。")
    L.append("   **根因已实证**：给提示词模板加上「写完回头数字数」这类自检规则后重跑，")
    L.append("   违规反而从 1 条变成 2 条。**字数、行数、字段齐全这类可机械核验的约束，")
    L.append("   不能指望模型自觉** —— 必须在管道里用代码卡，这正是本脚本做规则校验的原因。")
    L.append(f"   ⚠ **核验有边界**：另有 **{s['n_unverifiable']} 处**约束限定在子部件上"
             f"（如「principle 字段不超过 100 字」「每个要点不超过 20 字」"
             f"「需求摘要不超过 50 字」），子部件无法机械切分，**这 {s['n_unverifiable']} 处未经核验**，"
             f"不计入上面的通过率。这是能力边界，不是「全部通过」。")
    L.append(f"2. **样本量小**：{s['n_final']} 条，且集中在 {len(s['by_paradigm'])} 种范式、"
             f"{len(models)} 个模型。配套的模型对比实验已实证样本量不足会让结论翻转。")
    L.append("3. **种子改写增强的变体回答会重合，这是设计使然，不是缺陷**：")
    L.append("   该范式的目标是「保义扩量」——同一任务换多种问法，"
             "纯换表达的变体（同义改写/句式变换/角色设定）得到相同回答是**正确且预期**的。")
    L.append("4. **领域覆盖窄**：模板内置种子集中在代码、公文邮件、文本摘要三类，"
             "领域多样性依赖 `--seeds` 传入更多种子。")
    L.append("5. **同模板同种子重复运行会产生重复数据**：已由本脚本的去重步骤消除，")
    L.append("   但说明生成过程本身不具有跨批次去重能力。")
    L.append("6. **数据未经下游微调验证**：需求验证方案中的「精选 20% ≥ 全量 95%」"
             "尚未执行，故本数据集的**有效性没有实验支撑**。")
    L.append("")

    L.append("## 四、适用场景\n")
    L.append("> ⚠ **本节为草稿，交付前请人工确认或改写。**\n")
    L.append("- **适合**：指令微调（SFT）冷启动阶段的格式与风格对齐；")
    L.append("  指令遵循能力的初步训练；合成数据管道的教学演示与流程验证。")
    L.append("- **不适合**：需要事实准确性的知识注入（数据由模型生成，未做事实核查）；")
    L.append("  高风险领域（医疗、法律、金融）的模型训练；")
    L.append("  对领域多样性要求高的场景。")
    L.append("- **使用前建议**：过一遍本卡「已知偏差」；按需用 `LLM_judge/` 的四维评估"
             "再做一轮筛选；若用于对比实验，注意每条数据都带有 `source_model` 标记。")
    L.append("")

    L.append("## 五、复现方式\n")
    L.append("```bash")
    L.append("cd synthesis")
    L.append("python generate.py                              # 生成（四范式 × 已配 Key 的模型）")
    L.append("python export.py                                # 去重 + 校验 + 出这份数据卡")
    L.append("python test_jsonx.py                            # 解析层回归测试")
    L.append("python test_export.py                           # 去重与规则校验回归测试")
    L.append("```")
    L.append("")
    L.append("> 数据集的 Alpaca / ShareGPT 格式转换由 `LLM_judge/convert.py` 负责，本脚本不重复实现。")
    return "\n".join(L)


# ============================== 主流程 ==============================

def load_records(input_dir):
    files = sorted(glob.glob(str(Path(input_dir) / "syn_data_*.jsonl")))
    records = []
    for fn in files:
        for i, line in enumerate(open(fn, encoding="utf-8"), 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                print(f"  ⚠ 跳过无法解析的行 {Path(fn).name}:{i}")
                continue
            if not r.get("instruction") or not r.get("response"):
                print(f"  ⚠ 跳过缺字段的行 {Path(fn).name}:{i}")
                continue
            r["_file"] = Path(fn).name
            records.append(r)
    return files, records


def main():
    _ensure_utf8_stdout()
    ap = argparse.ArgumentParser(
        description="精选数据集导出 + 数据卡生成（去重 + 规则校验）")
    ap.add_argument("--input-dir", default=str(_HERE / "data" / "syn"),
                    help="生成结果目录（默认 data/syn）")
    ap.add_argument("--output-dir", default=str(_HERE / "data" / "export"),
                    help="导出目录（默认 data/export）")
    ap.add_argument("--threshold", type=float, default=0.8,
                    help="MinHash 近重复阈值（默认 0.8，需求指定）")
    ap.add_argument("--keep-invalid", action="store_true",
                    help="保留规则校验不通过的条目（默认剔除）")
    ap.add_argument("--dry-run", action="store_true", help="只统计与报告，不写文件")
    args = ap.parse_args()

    files, raw = load_records(args.input_dir)
    if not raw:
        sys.exit(f"错误：{args.input_dir} 下没有可用的 syn_data_*.jsonl。")
    print(f"读入 {len(files)} 个文件，共 {len(raw)} 条")

    # 先去重前测一次，否则数据卡只写"0%"会让人以为原始数据本就没重复
    raw_sigs = [minhash_signature(r["instruction"]) for r in raw]
    raw_dup = sum(1 for i in range(len(raw_sigs)) for j in range(i + 1, len(raw_sigs))
                  if jaccard_estimate(raw_sigs[i], raw_sigs[j]) >= args.threshold)
    raw_dup_rate = raw_dup / max(len(raw) - 1, 1) * 100
    print(f"去重前重复率：{raw_dup} 对 ≥{args.threshold} / {len(raw)} 条 = {raw_dup_rate:.1f}%")

    kept, exact_dropped = exact_dedup(raw)
    print(f"逐字去重：−{len(exact_dropped)} → {len(kept)} 条")
    for r, why in exact_dropped:
        print(f"    - [{r['_file']}] {why}：{r['instruction'][:44]}")

    kept2, near_dropped = minhash_dedup(kept, args.threshold)
    print(f"指令近重复去重（MinHash {args.threshold}）：−{len(near_dropped)} → {len(kept2)} 条")
    for r, why in near_dropped:
        print(f"    - [{r['_file']}] {why}")

    invalid, valid, borderline, n_unver = [], [], [], 0
    for r in kept2:
        v, b, u = check_constraints(r["instruction"], r["response"])
        n_unver += u
        if v:
            invalid.append({**r, "_why": "; ".join(v)})
        else:
            valid.append(r)
        if b:
            borderline.append({**r, "_why": "; ".join(b)})
    print(f"规则校验：{len(invalid)} 条确定违规 / {len(borderline)} 条边界存疑 / "
          f"{len(kept2)} 条；另有 {n_unver} 处约束限定在子部件上、无法机械核验（未计入）")
    for r in invalid:
        print(f"    ✗ [{r['_file']}] {r['_why']}：{r['instruction'][:44]}")
    for r in borderline:
        print(f"    ? [{r['_file']}] {r['_why']}：{r['instruction'][:44]}")

    final = kept2 if args.keep_invalid else valid
    print(f"精选后：{len(final)} 条")

    # 重复率复检（对最终数据集，按需求口径：MinHash 0.8 下还剩多少相似对）
    sigs = [minhash_signature(r["instruction"]) for r in final]
    dup_pairs = sum(1 for i in range(len(sigs)) for j in range(i + 1, len(sigs))
                    if jaccard_estimate(sigs[i], sigs[j]) >= args.threshold)
    dup_rate = dup_pairs / max(len(final) - 1, 1) * 100
    print(f"重复率复检：{dup_pairs} 对相似（≥{args.threshold}），"
          f"{dup_rate:.1f}%（需求要求 < 5%）")

    if args.dry_run:
        print("\n--dry-run：未写任何文件。")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    ds_path = out_dir / f"syn_curated_{ts}.jsonl"
    with open(ds_path, "w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps({k: r[k] for k in
                                ("paradigm", "category", "source_model",
                                 "instruction", "response")},
                               ensure_ascii=False) + "\n")

    lens = sorted(len(r["response"]) for r in final)
    ins_lens = sorted(len(r["instruction"]) for r in final)
    cross = Counter((r["paradigm"], r["source_model"]) for r in final)
    stats = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": f"syn_curated_{ts}",
        "n_files": len(files), "n_raw": len(raw),
        "n_exact_dup": len(exact_dropped), "n_near_dup": len(near_dropped),
        "n_rule_checked": len(kept2), "n_invalid": len(invalid),
        "n_borderline": len(borderline), "n_unverifiable": n_unver,
        "invalid_rate": len(invalid) / max(len(kept2), 1) * 100,
        "keep_invalid": args.keep_invalid,
        "n_final": len(final),
        "by_model": Counter(r["source_model"] for r in final),
        "by_paradigm": Counter(r["paradigm"] for r in final),
        "cross": cross,
        "len_med": lens[len(lens) // 2], "len_avg": sum(lens) / len(lens),
        "len_min": lens[0], "len_max": lens[-1],
        "ins_med": ins_lens[len(ins_lens) // 2],
        "dup_rate": dup_rate, "raw_dup_rate": raw_dup_rate,
        "file_dates": {Path(f).stem.replace("syn_data_", "")[:8] for f in files},
    }
    card_path = out_dir / f"DATA_CARD_{ts}.md"
    card_path.write_text(build_data_card(stats), encoding="utf-8")

    # 剔除明细落盘，保证"剔了什么"可审计、不是无声删除
    log_path = out_dir / f"export_{ts}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        for r, why in exact_dropped + near_dropped:
            f.write(json.dumps({"reason": why, "file": r["_file"],
                                "instruction": r["instruction"],
                                "response": r["response"]}, ensure_ascii=False) + "\n")
        for r in invalid:
            f.write(json.dumps({"reason": "规则校验-确定违规：" + r["_why"], "file": r["_file"],
                                "instruction": r["instruction"],
                                "response": r["response"]}, ensure_ascii=False) + "\n")
        for r in borderline:
            f.write(json.dumps({"reason": "规则校验-边界存疑（保留）：" + r["_why"],
                                "file": r["_file"], "instruction": r["instruction"],
                                "response": r["response"]}, ensure_ascii=False) + "\n")

    print(f"\n精选数据集：{ds_path}")
    print(f"数据卡：    {card_path}")
    print(f"剔除明细：  {log_path}（剔除 "
          f"{len(exact_dropped) + len(near_dropped) + (0 if args.keep_invalid else len(invalid))} 条，"
          f"边界存疑 {len(borderline)} 条已保留）")


if __name__ == "__main__":
    main()
