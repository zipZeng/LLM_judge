# 语料质量评估平台

一个**语料（训练数据）质量评估平台**：对语料从「安全性、准确性、多样性、格式规范性」四个维度进行多模型迭代打分，过滤低质量语料，并转换为标准训练格式。

> **设计参考**：本项目参考了 FastChat 的 LLM-as-a-judge 评判格式设计，代码完全独立实现。

---

## 1. 项目概述

一句话说明：**一个把「原始语料」变成「可直接用于训练的合格数据」的评估与转换流水线**——用三个云端模型（DeepSeek + 硅基流动 GLM-5.3 / Qwen3.6-35B-A3B）对每条语料在 4 个维度上迭代打分，按阈值过滤，最终输出 Alpaca / ShareGPT 标准训练格式。

支持三种使用方式：

- **Web 界面（推荐）**：双击 `start.bat`，在浏览器里点按钮即可运行各功能，免记命令行
- **单条评估**：`core/eval_loop.py` 评估单条语料，输出 JSON 报告
- **一键批量**：`core/pipeline.py` 批量处理语料文件，自动完成「评估 → 过滤转换 → 统计」全流程

---

## 2. 目录结构

```
LLM_judge/
├── core/                     # 核心脚本
│   ├── eval_loop.py          # 单条语料多模型迭代打分
│   ├── pipeline.py           # 一键批量流水线
│   ├── convert.py            # 数据格式转换（Alpaca / ShareGPT）
│   └── visualize.py          # 评估结果统计可视化
├── web/
│   └── app.py                # Streamlit 可视化界面（浏览器点击运行各功能）
├── synthesis/                # 合成数据生成与模型对比模块（见 synthesis/README.md）
├── data/
│   └── judge_prompts.jsonl   # 4 条评判提示词（safety/accuracy/diversity/format）
├── reports/                  # 自动生成：所有评估报告
├── history/                  # 自动生成：Web 界面操作历史
├── start.bat                 # Windows 一键启动 Web 界面
├── .env                      # DeepSeek + 硅基流动 API Key（不提交到 Git）
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 3. 系统架构

**单条评估流程（`eval_loop.py`）：**

```
语料文本
   │
   ▼
┌────────────────────────────────────────────────┐
│  eval_loop.py —— 多模型迭代打分                 │
│  · 加载 data/judge_prompts.jsonl 评判提示词     │
│  · 从 .env 读取 DeepSeek / 硅基流动 API Key     │
│  · DeepSeek + GLM-5.3 + Qwen3.6-35B-A3B         │
│  · 4 维度 × 3 模型 × N 轮（默认 3 轮）           │
└────────────────────────────────────────────────┘
   │
   ▼
评估报告 JSON ──► reports/eval_report_*.json
```

**一键批量流程（`pipeline.py`）：**

```
corpus.txt（每行一条语料）
   │
   ▼
pipeline.py ──逐条调用 eval_loop.py（--rounds 1）──► reports/eval_report_XXXX.json
   │                                                          │
   │                                                          ▼（合并）
   │                                              reports/_merged_reports.jsonl
   │                                                          │
   │                                      ┌───────────────────┴───────────────────┐
   │                                      ▼                                       ▼
   │                                convert.py                            visualize.py
   │                                      │                                       │
   │                                      ▼                                       ▼
   │                    output_alpaca.jsonl /                     ASCII 统计报表
   │                    output_sharegpt.jsonl
   ▼
汇总信息（共处理 / 合格 / 不合格 / 输出路径）
```

---

## 4. 评估指标体系说明

每条语料由三个模型分别对 4 个维度打分，分数范围 **0–10 分**，再按各维度权重加权得到**加权综合得分**，**≥ 5 分判为合格**（阈值与权重均可配置）。

| 维度 | name | 评估要点 |
|------|------|----------|
| 安全性 | `safety` | PII 泄露（身份证/手机号/邮箱/住址/银行卡等）、敏感政治内容、暴力色情赌博毒品、越狱攻击（jailbreak）、提示词注入（prompt injection） |
| 准确性 | `accuracy` | 事实性错误、逻辑矛盾（前后矛盾/因果倒置）、指令遵循是否完整 |
| 多样性 | `diversity` | n-gram 重复率（机械重复/啰嗦冗余）、语义覆盖度（信息量与主题广度）、与已有语料的相似度（模板化/套话化） |
| 格式规范性 | `format` | 结构完整性（是否截断/残缺）、字段齐全性（结构化字段是否缺失错位）、长度合理性（过短/过长） |

**维度权重（默认）：**

| 维度 | name | 权重 |
|------|------|------|
| 安全性 | `safety` | 35% |
| 准确性 | `accuracy` | 30% |
| 多样性 | `diversity` | 20% |
| 格式规范性 | `format` | 15% |

综合得分按上述权重对各维度均分加权求和得到（若有维度未评分，会按剩余权重自动归一化）。权重可在 `core/eval_loop.py` 顶部配置区的 `DIMENSION_WEIGHTS` 字典中修改。

**评分标准（通用，各维度在提示词中略有细化）：**

| 分数段 | 含义 |
|--------|------|
| 9–10 分 | 该维度表现优秀，无问题 |
| 7–8 分 | 整体良好，仅极轻微瑕疵 |
| 4–6 分 | 存在明显问题，需修正 |
| 1–3 分 | 存在严重问题 |
| 0 分 | 完全不合规 |

每个维度的详细评分细则见 `data/judge_prompts.jsonl` 中各条 `prompt_template`。

---

## 5. 快速开始

### 5.1 环境要求

- **Python 3.12**（建议，脚本兼容 3.8+）
- **DeepSeek API Key**（在 DeepSeek 开放平台获取）
- **硅基流动 API Key**（在 siliconflow.cn 获取）
- Python 依赖：`requests`、`python-dotenv`

### 5.2 安装依赖

```bash
pip install requests python-dotenv
```

### 5.3 配置 API Key（.env）

1. 在项目根目录创建 `.env` 文件（若不存在），写入：

   ```
   DEEPSEEK_API_KEY=你的真实Key
   SILICONFLOW_API_KEY=你的硅基流动Key
   ```

2. `.env` 已在 `.gitignore` 中忽略，不会被提交到 Git。
3. 若未设置 Key，脚本会使用占位符 `YOUR_DEEPSEEK_KEY` / `YOUR_SILICONFLOW_KEY` 并给出友好提示，对应平台的请求会失败。

其他可改配置（`core/eval_loop.py` 顶部）：`DEEPSEEK_URL`、`DEEPSEEK_MODEL`、`SILICONFLOW_URL`、`SILICONFLOW_MODELS`、`DIMENSION_WEIGHTS`、`TIMEOUT`、`MAX_RETRY` 等。

---

## 6. 脚本使用说明

### 6.1 eval_loop.py —— 单条语料多模型迭代打分

**功能**：加载 4 条评判提示词，三个模型对 4 个维度迭代打分（默认 3 轮），输出完整评估报告。

**命令行参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `text`（位置参数，可选） | 命令行直接传入的语料文本 | — |
| `-f, --file` | 从文件读取语料 | — |
| `-o, --output` | 报告输出路径 | `reports/eval_report_<时间戳>.json` |
| `--rounds` | 迭代轮数 | 3 |
| `--threshold` | 合格阈值 | 5.0 |
| `--progress` | 用进度条替代逐条详情打印（批量评估时建议开启） | 关闭 |

**示例：**

```bash
# 命令行直接输入文本
python core/eval_loop.py "深度学习是机器学习的一个分支，通过多层神经网络自动学习数据特征。"

# 从文件读取
python core/eval_loop.py --file corpus.txt

# 指定轮数、阈值和输出文件
python core/eval_loop.py --file corpus.txt --rounds 3 --threshold 5 --output reports/my_report.json
```

**输出格式**（单个 JSON，含每轮详情与最终结论）：

```json
{
  "meta": {"input_source": "file:corpus.txt", "text": "原始语料", "text_length": 42,
           "models": ["DeepSeek(deepseek-chat)", "GLM-5.3（硅基流动）", "Qwen3.6-35B-A3B（硅基流动）"],
           "rounds": 3, "threshold": 5.0, "generated_at": "..."},
  "rounds": [
    {"round": 1, "dimensions": {"safety": {"DeepSeek(deepseek-chat)": {"score": 8, "reason": "..."},
                                            "GLM-5.3（硅基流动）": {"score": 7, "reason": "..."},
                                            "Qwen3.6-35B-A3B（硅基流动）": {"score": 7, "reason": "..."}},
                                "accuracy": {"..."}, "diversity": {"..."}, "format": {"..."}}},
    {"round": 2, "dimensions": {"..."}},
    {"round": 3, "dimensions": {"..."}}
  ],
  "final": {"dimension_scores": {"safety": 7.5, "accuracy": 8.0, "diversity": 6.5, "format": 7.0},
            "weights": {"safety": 0.35, "accuracy": 0.30, "diversity": 0.20, "format": 0.15},
            "overall_score": 7.38, "passed": true, "conclusion": "合格"}
}
```

`reason` 字段保存模型的完整分析内容；控制台运行时仅显示一句「精炼理由」摘要（提示词已要求模型先输出 20 字以内的精炼理由，再输出完整分析）。

### 6.2 pipeline.py —— 一键批量处理

**功能**：读取语料文件（每行一条），逐条调用 `eval_loop.py` 评估，自动合并报告，再调用 `convert.py` 转换格式、`visualize.py` 输出统计。

**命令行参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input` | 输入语料文件（每行一条） | 必填 |
| `--format` | 输出格式：`alpaca` 或 `sharegpt` | 必填 |
| `--threshold` | 合格分数阈值 | 5.0 |
| `--rounds` | 每条语料评估轮数 | 1 |

**示例：**

```bash
python core/pipeline.py --input corpus.txt --format alpaca --threshold 5
python core/pipeline.py --input corpus.txt --format sharegpt --rounds 2 --threshold 6
```

**执行流程：**

1. 逐条评估，报告写入 `reports/eval_report_XXXX.json`
2. 合并所有报告为 `reports/_merged_reports.jsonl`
3. 调用 `convert.py` 输出 `output_alpaca.jsonl` 或 `output_sharegpt.jsonl`
4. 调用 `visualize.py` 输出统计报告
5. 打印汇总信息（共处理 / 合格 / 不合格 / 输出文件路径）

### 6.3 convert.py —— 数据格式转换

**功能**：读取评估报告，过滤掉 `overall_score < 阈值` 的语料，把通过的语料转成 Alpaca / ShareGPT 训练格式。

**命令行参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-i, --input` | 输入文件（报告 JSON / JSON 数组 / JSONL） | 必填 |
| `-o, --output` | 输出文件（JSONL，每行一条） | 必填 |
| `-f, --format` | 输出格式：`alpaca` 或 `sharegpt` | 必填 |
| `-t, --threshold` | 合格分数阈值 | 5.0 |

**示例：**

```bash
python core/convert.py --input reports/_merged_reports.jsonl --output output.jsonl --format alpaca --threshold 5
python core/convert.py --input reports/eval_report_0001.json --output out.jsonl --format sharegpt
```

**输出格式：**

- Alpaca：`{"instruction": "...", "input": "...", "output": "..."}`（普通无结构文本全放入 `instruction`）
- ShareGPT：`{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}`

**语料结构自动识别规则**：对 `meta.text` 逐行解析，自动识别对话结构（`human:`/`assistant:`/`用户:`/`助手:` 等）、Alpaca 结构（`instruction:`/`input:`/`output:`/`指令:`/`输入:`/`输出:` 等）或普通文本。详细映射规则见脚本 docstring。

**输出统计**：读取总数、保留（得分达标）、过滤（得分不足）、跳过（字段缺失/异常）。

### 6.4 visualize.py —— 结果可视化

**功能**：聚合一份或多份评估报告，输出统计信息（纯 `print`，无 GUI 依赖），并用 ASCII 柱状图展示各维度得分。

**命令行参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-i, --input` | 一个或多个报告文件（支持通配符，如 `*.jsonl`） | 必填 |
| `-t, --threshold` | 合格分数阈值 | 5.0 |

**示例：**

```bash
python core/visualize.py --input reports/eval_report_0001.json
python core/visualize.py --input reports/_merged_reports.jsonl --threshold 5
python core/visualize.py --input "reports/eval_report_*.json" --threshold 6
```

**输出内容：**

- 总语料数量、合格/不合格数量及占比
- 各维度平均分（ASCII 柱状图）
- 各维度分数分布（0-3分 / 4-6分 / 7-10分 占比）
- 各模型打分一致性（样本数、平均分、标准差、与其他模型的平均绝对差异）

---

## 7. 配置文件说明

### 7.1 judge_prompts.jsonl —— 评判提示词配置

路径：`data/judge_prompts.jsonl`

每行一个 JSON 对象，字段如下（4 条，对应 4 个维度）：

```json
{
  "name": "safety",
  "type": "single",
  "system_prompt": "你是一名专业的内容安全与合规审核专家……",
  "prompt_template": "[Instruction]……评分标准……\n\n[待评估语料]\n{text}",
  "description": "评估语料安全性（PII、敏感政治、暴力色情、越狱攻击、提示词注入）",
  "category": "corpus",
  "output_format": "[[rating]]"
}
```

- `name`：维度标识（`safety` / `accuracy` / `diversity` / `format`）
- `prompt_template`：完整评判提示词，`{text}` 为待评估语料占位符，要求模型依次输出「精炼理由（20字以内）/ 完整分析 / 分数 `[[评分]]`」三部分
- `output_format`：`"[[rating]]"`，脚本用正则 `\[\[(\d+)\]\]` 提取评分

### 7.2 .env —— API Key 配置

路径：`.env`（项目根目录，已在 `.gitignore` 中忽略）

```
DEEPSEEK_API_KEY=你的真实Key
SILICONFLOW_API_KEY=你的硅基流动Key
```

- `core/eval_loop.py` 启动时通过 `load_dotenv()` 自动读取项目根目录的 `.env`
- 未配置时回退到占位符 `YOUR_DEEPSEEK_KEY` / `YOUR_SILICONFLOW_KEY`，并在控制台给出友好提示

### 7.3 如何修改评估维度

1. 在 `data/judge_prompts.jsonl` 中新增/修改一条提示词（设置新的 `name`）。
2. 同步修改 `core/eval_loop.py` 顶部的 `DIMENSIONS` 列表，加入新的维度名：

   ```python
   DIMENSIONS = ["safety", "accuracy", "diversity", "format", "你新增的维度"]
   ```

3. 若需在 `core/visualize.py` 中显示中文名，同步修改其 `DIM_LABELS` 字典。

### 7.4 如何修改维度权重

`core/eval_loop.py` 顶部的 `DIMENSION_WEIGHTS` 字典定义各维度权重（总和建议为 1.0）：

```python
DIMENSION_WEIGHTS = {
    "safety": 0.35,
    "accuracy": 0.30,
    "diversity": 0.20,
    "format": 0.15,
}
```

修改对应数值即可调整各维度在综合得分中的占比（若有维度未评分，脚本会按剩余权重自动归一化）。

### 7.5 如何添加新模型

`core/eval_loop.py` 中模型以统一接口 `call(system_prompt, user_prompt) -> str` 注册在 `MODELS` 列表：

1. 写一个调用函数（参照 `call_deepseek` / `call_siliconflow`），返回模型原始输出文本：

   ```python
   def call_my_model(system_prompt, user_prompt):
       # 调用你的模型 API，返回文本
       ...
   ```

2. 在 `MODELS` 列表中注册：

   ```python
   from functools import partial

   MODELS = [
       {"key": "deepseek", "label": f"DeepSeek({DEEPSEEK_MODEL})", "call": call_deepseek},
       {"key": "glm53", "label": "GLM-5.3（硅基流动）", "call": partial(call_siliconflow, model=SILICONFLOW_MODELS[0])},
       {"key": "mymodel", "label": "MyModel", "call": call_my_model},
   ]
   ```

评估循环会自动遍历 `MODELS` 中所有模型，无需改动其他逻辑。

---

## 8. 项目成果清单

| 文件 | 说明 |
|------|------|
| `data/judge_prompts.jsonl` | 4 条语料质量评判提示词（safety/accuracy/diversity/format） |
| `core/eval_loop.py` | 单条语料多模型迭代打分脚本 |
| `core/pipeline.py` | 一键批量流水线脚本 |
| `core/convert.py` | 数据格式转换脚本（Alpaca/ShareGPT） |
| `core/visualize.py` | 评估结果可视化统计脚本 |
| `web/app.py` | Streamlit 可视化界面 |
| `synthesis/` | 合成数据生成与模型对比模块（见 `synthesis/README.md`） |
| `.env` | DeepSeek + 硅基流动 API Key（不提交到 Git） |
| `README.md` | 本文档 |

---

## 9. 常见问题

**Q1：硅基流动请求返回 401 或报「API Key 仍是占位符」？**
A：请在 `.env` 中正确设置 `SILICONFLOW_API_KEY`（在 siliconflow.cn 获取），并确认模型 ID 正确（见 `core/eval_loop.py` 顶部 `SILICONFLOW_MODELS`）。若 Key 未填写，脚本会使用占位符并给出提醒。

**Q2：DeepSeek 返回 401 或报「API Key 仍是占位符」？**
A：请在项目根目录的 `.env` 文件中正确设置 `DEEPSEEK_API_KEY`（在 DeepSeek 开放平台获取），并确认账户有可用额度。若 `.env` 不存在或 Key 未填写，脚本会使用占位符并给出提醒。

**Q3：模型输出了理由但脚本提示「未提取到分数」？**
A：评判提示词要求模型以 `[[评分]]` 格式输出，脚本用正则 `\[\[(\d+)\]\]` 提取。若模型未按格式输出，脚本会尝试兜底提取数字，仍失败则记 `score=None`。可适当调低温度、在提示词中更强调输出格式，或换用更强模型。

**Q4：Windows 控制台输出中文乱码？**
A：脚本已内置 `_ensure_utf8_stdout()` 尽量将 stdout 设为 UTF-8。若仍乱码，可在运行前执行 `chcp 65001`，或使用支持 UTF-8 的终端（如 Windows Terminal）。

**Q5：提示 `ModuleNotFoundError: No module named 'requests'` 或 `'dotenv'`？**
A：安装依赖：`pip install requests python-dotenv`。

**Q6：`convert.py` 里普通文本语料转 ShareGPT 后内容进了 `gpt` 而不是 `human`？**
A：这是无结构纯文本的默认兜底行为（视作一条模型回复）。若你的纯文本大多是问题/指令，可修改 `convert.py` 中 `to_sharegpt` 的 plain 分支，把 `"from": "gpt"` 改为 `"human"`。

---

## 10. 参考资料

- [DeepSeek API 文档](https://api-docs.deepseek.com)
- [硅基流动 API 文档](https://docs.siliconflow.cn)
