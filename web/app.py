"""大模型合成数据生成与质量评估平台 —— Web 可视化界面（Streamlit）

精简版：6 个入口，按用户使用流程组织。
- 首页：简介 + 能帮你做什么 + 新手引导
- 评估语料：单条 / 批量（eval_loop.py / pipeline.py）
- 生成数据：合成数据生成（generate.py）
- 模型对比：三模型互评（compare_models.py）
- 导出精选集：去重 + 数据卡（export.py）
- 查看结果：评分统计 + 格式转换下载（visualize.py / convert.py）

启动方式：
    pip install -r requirements.txt
    streamlit run web/app.py
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

# ----------------------------- 路径配置 -----------------------------
ROOT = Path(__file__).resolve().parent.parent
SYNTHESIS = ROOT / "synthesis"
HISTORY_DIR = ROOT / "history"
HISTORY_DIR.mkdir(exist_ok=True)
(ROOT / "reports").mkdir(exist_ok=True)

EVAL_LOOP = ROOT / "core" / "eval_loop.py"
PIPELINE = ROOT / "core" / "pipeline.py"
CONVERT = ROOT / "core" / "convert.py"
VISUALIZE = ROOT / "core" / "visualize.py"
GENERATE = SYNTHESIS / "generate.py"
COMPARE = SYNTHESIS / "compare_models.py"
EXPORT = SYNTHESIS / "export.py"

PY = sys.executable

DIM_LABELS = {
    "safety": "安全性",
    "accuracy": "准确性",
    "diversity": "多样性",
    "format": "格式规范性",
}

PARADIGMS = {
    "self_instruct": "举一反三 (Self-Instruct)",
    "evol_instruct": "加难 (Evol-Instruct)",
    "magpie": "从零造 (Magpie)",
    "seed_rewrite": "换说法 (种子改写)",
}

PAGES = {
    "首页": "🏠",
    "评估语料": "📝",
    "生成数据": "🏭",
    "模型对比": "⚖️",
    "导出精选集": "📦",
    "查看结果": "📊",
}

st.set_page_config(page_title="语料质量评估平台", page_icon="📊", layout="wide")


# ----------------------------- 通用工具 -----------------------------
def run_streaming(cmd, cwd):
    """执行命令并逐行流式显示输出。返回 (returncode, 完整日志)。"""
    placeholder = st.empty()
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    log = ""
    try:
        for line in proc.stdout:
            log += line
            placeholder.code(log, language=None)
        proc.stdout.close()
        code = proc.wait()
    except Exception as e:
        log += f"\n[界面异常] {e}\n"
        placeholder.code(log, language=None)
        code = -1
    return code, log


def save_history(page, command, log, outputs=None):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rec = {
        "id": ts,
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "page": page,
        "command": command,
        "log": log,
        "outputs": outputs or [],
    }
    (HISTORY_DIR / f"{ts}.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return rec


def list_history():
    return sorted(HISTORY_DIR.glob("*.json"), reverse=True)


def download_file(path, label="下载文件"):
    p = Path(path)
    if p.exists():
        st.download_button(label, p.read_bytes(), file_name=p.name)


def jsonl_preview(path, limit=100):
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    rows.append({"raw": line})
    except Exception:
        pass
    return rows


def load_reports_jsonl(path):
    reports = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    reports.append(json.loads(line))
                except Exception:
                    pass
    except Exception:
        pass
    return reports


def check_synthesis_env():
    if not (SYNTHESIS / ".env").exists():
        st.warning(
            "⚠ `synthesis/.env` 未配置：联网运行会跳过所有模型。"
            "「预览模式」与「快速演示」不受影响。真跑前请在 `synthesis/` 下创建 `.env`（参照 `.env.example`）。"
        )


def intro(text):
    """每页顶部的「这个功能是做什么的」通俗说明。"""
    st.info(text)


# ----------------------------- 首页 -----------------------------
def page_home():
    st.title("📊 大模型合成数据生成与质量评估平台")
    st.markdown(
        "一个把「原始语料」变成「可直接用于训练的合格数据」的平台：先让 AI **造数据**，"
        "再**评估打分**，最后**去重打包**成标准训练格式。"
    )
    st.markdown("---")

    st.subheader("这个平台能帮你做什么？")
    with st.container(border=True):
        st.markdown(
            "✅ 检查你的文本数据质量好不好\n\n"
            "✅ 让 AI 帮你批量造训练数据\n\n"
            "✅ 对比哪个 AI 造的数据最好\n\n"
            "✅ 过滤不合格数据，输出标准训练格式"
        )

    st.markdown("---")
    st.subheader("新手引导：按这个顺序使用")
    steps = [
        ("🏭 生成数据", "让 AI 帮你造一批训练数据（四种方法可选）", "生成数据"),
        ("⚖️ 模型对比", "看 DeepSeek / GLM / Qwen 谁造的数据最好（可选）", "模型对比"),
        ("📦 导出精选集", "去重、检查，打包成一份精选集 + 数据卡", "导出精选集"),
        ("📝 评估语料", "给数据打质量分，筛掉不合格的", "评估语料"),
        ("📊 查看结果", "看整体评分统计，转成 Alpaca/ShareGPT 下载", "查看结果"),
    ]
    for i, (icon_name, desc, page) in enumerate(steps, 1):
        c1, c2 = st.columns([1, 5])
        with c1:
            if st.button(icon_name, key=f"step_{page}", use_container_width=True):
                st.session_state["page"] = page
                st.rerun()
        with c2:
            st.markdown(f"**第 {i} 步**　{desc}")

    st.markdown("---")
    st.caption("也可以直接从左侧导航进入任意功能。所有联网操作前请先确认已配置 API Key。")


# ----------------------------- 评估语料 -----------------------------
def show_eval_result(data):
    final = data.get("final", {})
    scores = final.get("dimension_scores", {})
    overall = final.get("overall_score")
    passed = final.get("passed")
    conclusion = final.get("conclusion", "")
    st.subheader("四维得分")
    cols = st.columns(len(scores) if scores else 4)
    for i, (k, v) in enumerate(scores.items()):
        cols[i].metric(DIM_LABELS.get(k, k), f"{v:.2f}")
    if scores:
        import pandas as pd

        df = pd.DataFrame(
            {"维度": [DIM_LABELS.get(k, k) for k in scores], "得分": list(scores.values())}
        )
        st.bar_chart(df.set_index("维度"))
    if overall is not None:
        mark = "✅ 合格" if passed else "❌ 不合格"
        st.markdown(f"**加权综合得分：{overall:.2f}**　|　结论：**{conclusion}**　（{mark}）")


def page_eval():
    st.header("📝 评估语料")
    intro("给一段文本打质量分（安全性 / 准确性 / 多样性 / 格式），分数 ≥ 阈值算合格。")

    tab_single, tab_batch = st.tabs(["单条评估", "批量评估"])

    with tab_single:
        st.caption("给一段文本打分，判断质量好不好。")
        st.caption("比如粘贴一段问答对、一段代码、一段对话，它会从安全性 / 准确性 / 多样性 / 格式四个维度打分。")
        text = st.text_area("输入语料", height=120, placeholder="粘贴要评估的语料文本…", key="single_text")
        c1, c2 = st.columns(2)
        rounds = c1.number_input("迭代轮数", 1, 10, 3, key="s_rounds")
        threshold = c2.number_input("合格阈值", 0.0, 10.0, 5.0, key="s_th")
        if st.button("运行评估", type="primary", disabled=not text.strip(), key="s_run"):
            out = ROOT / "reports" / f"ui_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            cmd = [
                PY, str(EVAL_LOOP), text.strip(),
                "--rounds", str(int(rounds)),
                "--threshold", str(threshold),
                "--output", str(out),
            ]
            st.caption("💡 会调用云端模型打分，单条约需 1–3 分钟，请耐心等待。")
            code, log = run_streaming(cmd, ROOT)
            save_history("评估语料-单条", " ".join(cmd), log, [str(out)] if out.exists() else [])
            if code == 0:
                st.success("评估完成")
            else:
                st.error(f"运行出错（退出码 {code}）")
            if out.exists():
                try:
                    show_eval_result(json.loads(out.read_text(encoding="utf-8")))
                except Exception as e:
                    st.warning(f"报告解析失败：{e}")
            with st.expander("查看完整日志"):
                st.code(log, language=None)

    with tab_batch:
        st.caption("上传一个语料文件（每行一条），一键批量评估并转换格式。")
        up = st.file_uploader("上传语料文件（.txt / .jsonl，每行一条）", type=["txt", "jsonl"], key="batch_upload")
        c1, c2, c3 = st.columns(3)
        fmt = c1.radio("输出格式", ["alpaca", "sharegpt"], horizontal=True, key="b_fmt")
        threshold = c2.number_input("合格阈值", 0.0, 10.0, 5.0, key="b_th")
        rounds = c3.number_input("每条轮数", 1, 10, 1, key="b_rounds")
        if st.button("运行流水线", type="primary", disabled=up is None, key="b_run"):
            tmp = ROOT / f"_upload_{datetime.now().strftime('%H%M%S')}.txt"
            tmp.write_bytes(up.getvalue())
            cmd = [
                PY, str(PIPELINE),
                "--input", str(tmp),
                "--format", fmt,
                "--threshold", str(threshold),
                "--rounds", str(int(rounds)),
            ]
            st.caption("💡 会逐条调用模型评估，条数多时耗时较长。")
            code, log = run_streaming(cmd, ROOT)
            tmp.unlink(missing_ok=True)
            out = ROOT / f"output_{fmt}.jsonl"
            save_history("评估语料-批量", " ".join(cmd), log, [str(out)] if out.exists() else [])
            if code == 0:
                st.success("运行完成")
            else:
                st.error(f"运行出错（退出码 {code}）")
            if out.exists():
                download_file(out, f"下载 {out.name}")
            with st.expander("查看完整日志"):
                st.code(log, language=None)


# ----------------------------- 生成数据 -----------------------------
def page_generate():
    st.header("🏭 生成数据")
    intro("让 AI 帮你从零造训练数据，四种方法可选：举一反三、加难、从零造、换说法扩量。")
    check_synthesis_env()

    c1, c2, c3 = st.columns(3)
    paradigms = c1.multiselect(
        "范式",
        list(PARADIGMS.keys()),
        default=list(PARADIGMS.keys()),
        format_func=lambda x: PARADIGMS[x],
        key="gen_paradigm",
    )
    models = c2.multiselect(
        "模型", ["deepseek", "glm", "qwen"], default=["deepseek", "glm", "qwen"], key="gen_models"
    )
    dry = c3.checkbox("预览模式（不实际生成）", value=True, key="gen_dry")

    seeds = st.text_input("自定义种子（`|` 分隔，可选）", "", key="gen_seeds")

    with st.expander("高级设置"):
        c4, c5 = st.columns(2)
        max_tokens = c4.number_input("max-tokens", 1024, 65536, 16384, key="gen_max_tokens")
        timeout = c5.number_input("timeout（秒）", 60, 3600, 600, key="gen_timeout")

    if st.button("运行生成", type="primary", disabled=not paradigms or not models, key="gen_run"):
        cmd = [PY, str(GENERATE)]
        cmd += ["--paradigm"] + paradigms
        cmd += ["--models"] + models
        if seeds.strip():
            cmd += ["--seeds", seeds.strip()]
        cmd += ["--max-tokens", str(int(max_tokens)), "--timeout", str(int(timeout))]
        if dry:
            cmd.append("--dry-run")

        if dry:
            st.caption("💡 只预览生成计划，不联网、不花 token。")
        else:
            st.caption("💡 真跑会联网，推理模型单次可能 4–7.5 分钟，请勿切换页面。")
        code, log = run_streaming(cmd, SYNTHESIS)
        save_history("生成数据", " ".join(cmd), log)
        if code == 0:
            st.success("运行完成")
        else:
            st.error(f"运行出错（退出码 {code}）")

        syn_files = sorted((SYNTHESIS / "data" / "syn").glob("*.jsonl"), reverse=True)
        if syn_files:
            st.subheader("生成结果（最新数据集）")
            for f in syn_files[:3]:
                rows = jsonl_preview(f, 50)
                st.markdown(f"**{f.name}**（预览 {len(rows)} 条）")
                if rows:
                    st.dataframe(rows, use_container_width=True)
                download_file(f, f"下载 {f.name}")
                st.markdown("---")


# ----------------------------- 模型对比 -----------------------------
def page_compare():
    st.header("⚖️ 模型对比")
    intro("让 DeepSeek / GLM / Qwen 各造一批数据，再互相打分，看谁生成的数据质量最高。")
    check_synthesis_env()

    mode = st.radio("模式", ["快速演示", "真跑"], horizontal=True, key="cmp_mode")
    if mode == "快速演示":
        st.caption("💡 用之前跑好的结果展示，不需要联网。")
    else:
        st.caption("💡 会实际调用 AI，耗时 25–45 分钟。")

    if mode == "快速演示":
        compare_dir = SYNTHESIS / "data" / "compare"
        archives = sorted([p for p in compare_dir.glob("*") if p.is_dir()], reverse=True)
        if archives:
            sel = st.selectbox("选择存档目录", [str(p) for p in archives], key="cmp_archive")
            if st.button("运行（重放对比表）", type="primary", key="cmp_replay"):
                cmd = [PY, str(COMPARE), "--from-report", sel]
                st.caption("💡 不联网、不花 token，秒级出结果。")
                code, log = run_streaming(cmd, SYNTHESIS)
                summary = Path(sel) / "summary.json"
                save_history("模型对比", " ".join(cmd), log, [str(summary)] if summary.exists() else [])
                if code == 0:
                    st.success("重放完成")
                else:
                    st.error(f"运行出错（退出码 {code}）")
                st.subheader("对比表")
                st.code(log, language=None)
                download_file(summary, "下载 summary.json")
        else:
            st.info("未找到存档目录，可先「真跑」一次生成。")
    else:
        c1, c2, c3 = st.columns(3)
        per_seed = c1.number_input("每种子生成对数", 1, 10, 2, key="cmp_per_seed")
        models = c2.multiselect("生成模型", ["deepseek", "glm", "qwen"], default=["deepseek", "glm", "qwen"], key="cmp_models")
        judges = c3.multiselect("评审模型", ["deepseek", "glm", "qwen"], default=["deepseek", "glm", "qwen"], key="cmp_judges")
        c4, c5 = st.columns(2)
        max_tokens = c4.number_input("max-tokens", 1024, 65536, 16384, key="cmp_max_tokens")
        timeout = c5.number_input("timeout（秒）", 60, 3600, 900, key="cmp_timeout")

        if st.button("开始真跑", type="primary", key="cmp_run"):
            cmd = [PY, str(COMPARE), "--per-seed", str(int(per_seed))]
            if models:
                cmd += ["--models"] + models
            if judges:
                cmd += ["--judges"] + judges
            cmd += ["--max-tokens", str(int(max_tokens)), "--timeout", str(int(timeout))]
            st.caption("💡 真跑需 25–45 分钟，运行期间请勿切换页面。")
            code, log = run_streaming(cmd, SYNTHESIS)
            save_history("模型对比", " ".join(cmd), log)
            if code == 0:
                st.success("运行完成")
            else:
                st.error(f"运行出错（退出码 {code}）")
            st.code(log, language=None)


# ----------------------------- 导出精选集 -----------------------------
def page_export():
    st.header("📦 导出精选集")
    intro("把造好的数据去重、规则检查，打包成一份「精选集」，并生成一张数据卡（记录来源、构成、已知偏差）。")
    check_synthesis_env()

    c1, c2, c3 = st.columns(3)
    dry = c1.checkbox("dry-run（只统计不写文件）", value=True, key="exp_dry")
    keep_invalid = c2.checkbox("保留违规项", value=False, key="exp_keep")
    threshold = c3.number_input("去重阈值", 0.0, 1.0, 0.8, key="exp_threshold")

    if st.button("运行导出", type="primary", key="exp_run"):
        cmd = [PY, str(EXPORT), "--threshold", str(threshold)]
        if dry:
            cmd.append("--dry-run")
        if keep_invalid:
            cmd.append("--keep-invalid")
        if dry:
            st.caption("💡 只统计与报告，不写文件。")
        else:
            st.caption("💡 会写精选集与数据卡到 `synthesis/data/export/`。")
        code, log = run_streaming(cmd, SYNTHESIS)
        save_history("导出精选集", " ".join(cmd), log)
        if code == 0:
            st.success("运行完成")
        else:
            st.error(f"运行出错（退出码 {code}）")

        if not dry:
            export_dir = SYNTHESIS / "data" / "export"
            curated = sorted(export_dir.glob("syn_curated_*.jsonl"), reverse=True)
            cards = sorted(export_dir.glob("DATA_CARD_*.md"), reverse=True)
            if curated:
                st.subheader("精选数据集")
                rows = jsonl_preview(curated[0], 50)
                st.dataframe(rows, use_container_width=True)
                download_file(curated[0], "下载精选集")
            if cards:
                st.subheader("数据卡")
                st.markdown(cards[0].read_text(encoding="utf-8"))
            st.caption("💡 需转 Alpaca / ShareGPT 格式？到「查看结果 → 格式转换」操作。")


# ----------------------------- 查看结果 -----------------------------
def native_visualization(report_files, threshold):
    all_reports = []
    for p in report_files:
        p = Path(p)
        if p.suffix == ".jsonl":
            all_reports += load_reports_jsonl(p)
        else:
            try:
                all_reports.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                pass
    if not all_reports:
        st.info("未解析到报告")
        return

    dims = ["safety", "accuracy", "diversity", "format"]
    avg = {d: [] for d in dims}
    passed = 0
    for r in all_reports:
        final = r.get("final", {})
        sc = final.get("dimension_scores", {})
        for d in dims:
            v = sc.get(d)
            if isinstance(v, (int, float)):
                avg[d].append(v)
        if final.get("passed") is True or (final.get("overall_score", 0) >= threshold):
            passed += 1

    import pandas as pd

    df = pd.DataFrame(
        {"维度": [DIM_LABELS[d] for d in dims],
         "平均分": [round(sum(avg[d]) / len(avg[d]), 2) if avg[d] else 0.0 for d in dims]}
    )
    st.subheader("各维度平均分")
    st.bar_chart(df.set_index("维度"))

    total = len(all_reports)
    m1, m2, m3 = st.columns(3)
    m1.metric("总语料", total)
    m2.metric("合格", passed)
    m3.metric("合格率", f"{passed / total * 100:.1f}%" if total else "0%")


def page_results():
    st.header("📊 查看结果")
    intro("看整体数据质量报告（各维度平均分、合格率），并把合格数据转成 Alpaca / ShareGPT 格式下载。")

    tab_stat, tab_convert = st.tabs(["评分统计", "格式转换与下载"])

    with tab_stat:
        st.caption("读取 `reports/` 里的评估报告，展示统计图表。")
        reports_dir = ROOT / "reports"
        merged = reports_dir / "_merged_reports.jsonl"
        options = []
        if merged.exists():
            options.append(str(merged))
        options += [str(p) for p in sorted(reports_dir.glob("eval_report_*.json"))]
        options += [str(p) for p in sorted(reports_dir.glob("ui_eval_*.json"))]
        if not options:
            st.info("暂无报告，先到「评估语料」跑一次评估。")
        else:
            sel = st.selectbox("选择报告", options, key="res_stat_select")
            threshold = st.number_input("合格阈值", 0.0, 10.0, 5.0, key="r_th")
            if st.button("生成报表", type="primary", key="res_stat_run"):
                cmd = [PY, str(VISUALIZE), "--input", sel, "--threshold", str(threshold)]
                st.caption("💡 离线统计，不联网。")
                code, log = run_streaming(cmd, ROOT)
                save_history("查看结果-统计", " ".join(cmd), log)
                if code == 0:
                    st.success("运行完成")
                else:
                    st.error(f"运行出错（退出码 {code}）")
                with st.expander("查看 visualize.py 原始输出"):
                    st.code(log, language=None)
                st.markdown("---")
                native_visualization([sel], threshold)

    with tab_convert:
        st.caption("把评估报告转成 Alpaca / ShareGPT 训练格式，并下载。")
        src = st.radio("输入来源", ["从已有报告选择", "上传文件"], horizontal=True, key="c_src")
        if "上传" in src:
            up = st.file_uploader("上传报告文件（.json / .jsonl）", type=["json", "jsonl"], key="res_conv_upload")
            input_path = None
        else:
            reports_dir = ROOT / "reports"
            reports = sorted(reports_dir.glob("*.jsonl")) + sorted(reports_dir.glob("*.json"))
            if reports:
                input_path = Path(st.selectbox("选择报告", [str(p) for p in reports], key="res_conv_select"))
                up = None
            else:
                st.info("reports/ 下暂无报告文件")
                input_path = None
                up = None

        c1, c2 = st.columns(2)
        fmt = c1.radio("输出格式", ["alpaca", "sharegpt"], horizontal=True, key="c_fmt")
        threshold = c2.number_input("合格阈值", 0.0, 10.0, 5.0, key="c_th")

        ready = (up is not None) if "上传" in src else (input_path is not None)
        if st.button("运行转换", type="primary", disabled=not ready, key="res_conv_run"):
            if "上传" in src:
                tmp = ROOT / f"_convert_{datetime.now().strftime('%H%M%S')}.jsonl"
                tmp.write_bytes(up.getvalue())
                input_path = tmp
            out = ROOT / f"output_{fmt}.jsonl"
            cmd = [
                PY, str(CONVERT),
                "--input", str(input_path),
                "--output", str(out),
                "--format", fmt,
                "--threshold", str(threshold),
            ]
            st.caption("💡 离线转换，不联网。")
            code, log = run_streaming(cmd, ROOT)
            if "上传" in src:
                input_path.unlink(missing_ok=True)
            save_history("查看结果-转换", " ".join(cmd), log, [str(out)] if out.exists() else [])
            if code == 0:
                st.success("转换完成")
            else:
                st.error(f"运行出错（退出码 {code}）")
            if out.exists():
                download_file(out, f"下载 {out.name}")
            with st.expander("查看完整日志"):
                st.code(log, language=None)

        st.markdown("---")
        st.subheader("已有产物（可直接下载）")
        outputs = sorted(ROOT.glob("output_*.jsonl"), reverse=True)
        if outputs:
            for o in outputs[:5]:
                download_file(o, f"下载 {o.name}")
        else:
            st.caption("暂无转换产物，跑一次转换后会出现。")


# ----------------------------- 主入口 -----------------------------
def main():
    if "page" not in st.session_state:
        st.session_state["page"] = "首页"

    with st.sidebar:
        st.title("功能导航")
        for name, icon in PAGES.items():
            if st.button(f"{icon}  {name}", key=f"nav_{name}", use_container_width=True):
                st.session_state["page"] = name
                st.rerun()

        st.divider()
        with st.expander("🗂️ 历史记录"):
            files = list_history()
            if not files:
                st.caption("暂无记录")
            for f in files[:20]:
                try:
                    rec = json.loads(f.read_text(encoding="utf-8"))
                except Exception:
                    continue
                with st.expander(f"{rec.get('ts', '')} · {rec.get('page', '')}"):
                    st.caption("命令：")
                    st.code(rec.get("command", ""), language=None)
                    st.caption("日志（截断）：")
                    st.code(rec.get("log", "")[:1500], language=None)
            if files:
                if st.button("清空历史", key="clear_history", use_container_width=True):
                    for f in files:
                        f.unlink(missing_ok=True)
                    st.rerun()

        st.caption("语料质量评估平台 · Streamlit 版")

    dispatch = {
        "首页": page_home,
        "评估语料": page_eval,
        "生成数据": page_generate,
        "模型对比": page_compare,
        "导出精选集": page_export,
        "查看结果": page_results,
    }
    dispatch[st.session_state["page"]]()


if __name__ == "__main__":
    main()
