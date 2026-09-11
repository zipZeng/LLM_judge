# 合成数据生成与模型对比（synthesis/）

基于 `prompts/` 目录下的提示词模板，为选题 41「大模型合成数据生成与质量评估平台」
补齐两块核心能力：

1. **合成数据生成**（`generate.py`）—— 按 Self-Instruct / Evol-Instruct / Magpie /
   种子改写增强 四种范式生成指令-响应对，输出统一 JSONL 数据集。
2. **评价生成语料的大模型**（`compare_models.py`）—— 同一批种子让
   DeepSeek / GLM-5.3 / Qwen3.6 各生成一批数据，再由三者互评（LLM-as-Judge），
   对比“谁生成的数据质量更高”，并给出排除自评的公平口径。
3. **精选数据集导出与数据卡**（`export.py`）—— 汇总全部分批生成结果，去重、
   规则校验、输出统一精选集，并据实生成数据卡（来源/构成/已知偏差/适用场景）。
   对应需求「核心功能模块 5」与验证方案「重复率 < 5%」。

## 目录结构

```
synthesis/
├── llm.py             # 三家云端模型统一调用（DeepSeek + 硅基流动，OpenAI 兼容）
├── jsonx.py           # 模型 JSON 输出鲁棒解析（围栏剥离/括号配平/递归展平）
├── promptio.py        # 读取 prompts/*.md 提示词模板（路径自动推导）
├── generate.py        # 合成数据生成器（四范式）
├── compare_models.py  # 生成模型横向对比（生成 → 互评 → 汇总排名）
├── export.py          # 精选数据集导出 + 数据卡（去重 / 规则校验 / 统计）
├── test_jsonx.py      # jsonx 回归测试（python test_jsonx.py，不联网）
├── test_export.py     # export 回归测试（python test_export.py，不联网）
├── prompts/           # 提示词模板（5 个 .md，随本模块一起入库）
│   ├── self_instruct.md / evol_instruct.md / magpie.md
│   ├── seed_rewrite.md       # 种子改写增强（第四种范式）
│   └── model_comparison.md   # 模型对比用评审模板
├── .env.example       # API Key 配置示例（复制为 .env 填写）
├── data/syn/          # 自动生成：数据集 JSONL / 错误日志 / raw_<ts>/ 模型原始输出
├── data/compare/<ts>/ # 自动生成：每家原始数据 + 每位评审报告 + summary.json
├── data/export/       # 自动生成：精选数据集 + DATA_CARD_<ts>.md + 剔除明细
└── README.md
```

> `prompts/` 是**必须随模块交付**的：缺了它 `generate.py` / `compare_models.py`
> 会直接报「找不到提示词模板」退出。查找顺序为
> `synthesis/prompts/` → `<上级目录>/prompts/`（兼容模板放在项目根目录的旧布局）。

## 环境准备

```bash
pip install requests python-dotenv
```

**API Key（`.env`）**：在 `synthesis/` 目录创建 `.env`（参照 `.env.example`）：

```
DEEPSEEK_API_KEY=sk-xxx          # DeepSeek 开放平台
SILICONFLOW_API_KEY=sk-xxx       # 硅基流动（GLM-5.3 / Qwen3.6 共用）
```

Key 读取优先级：`synthesis/.env` → `git_hub/LLM_judge/.env`（若存在，与评估项目共用）
→ 系统环境变量。缺 Key 的模型会被自动跳过并提示，不影响其他模型执行。
`.env` 含密钥，请勿提交到 Git（已建议加入 .gitignore）。

模型名/接口地址集中配置在 `llm.py` 的 `PROVIDERS` 表，若硅基流动模型 tag 变动，
只改这一处即可。

## 1. 合成数据生成 generate.py

把范式模板全文作为提示词发给模型（附补充种子），解析并展平返回的 JSON，
输出统一 JSONL：`{"paradigm", "category", "source_model", "instruction", "response"}`，
全程按 (instruction, response) 去重。

```bash
# 四范式 × 所有已配 Key 的模型（缺 Key 自动跳过）
python generate.py

# 指定范式与模型
python generate.py --paradigm self_instruct magpie --models deepseek glm

# 自定义种子指令（| 分隔；不给则用模板内置默认种子）
python generate.py --paradigm magpie --seeds "帮我写一封请假邮件|总结一篇论文的核心观点"

# 预览不调用 API
python generate.py --dry-run
```

参数：`--paradigm`（self_instruct/evol_instruct/magpie/**seed_rewrite**）、
`--models`（deepseek/glm/qwen）、`--seeds`、`--max-tokens`（默认 16384）、
`--timeout`（默认 600，单次调用读超时秒数）、`--output-dir`。

**四种范式的分工：**

| 范式 | 输入 | 目标 | 模板 |
|---|---|---|---|
| Self-Instruct | 少量种子指令 | 自举扩展出大量新任务 | `self_instruct.md` |
| Evol-Instruct | 种子指令 | **加难 / 加宽**，提升复杂度 | `evol_instruct.md` |
| Magpie | 无需种子 | 利用对齐特性从零自合成 | `magpie.md` |
| **种子改写增强** | 已有**数据对** | **保义扩量**，难度持平、换表达 | `seed_rewrite.md` |

> ⚠ 种子改写增强与 `evol_instruct.md` 里的「指令改写」分支**不是一回事**：
> 后者是进化的一个分支，目标是把指令改得更复杂；前者保持任务本质不变、
> 难度与种子持平，定位是"扩量"。两者在模板里都做了明确说明。

> 关于 `--seeds`：`seed_rewrite` 的模板内部已自带 3 条**数据对**形式的种子
> （含 instruction 与 response），默认直接用它们即可。`--seeds` 传入的是
> **纯指令**（无配套回答），适用于只想按指令做改写扩量的场景。

**容错设计**：模型输出带 ```json 围栏、前后废话、键名漂移（如"指令/响应"）都能解析；
某次调用失败记入 `data/syn/errors_<ts>.log` 并继续，不中断整批。

**原始输出落盘**：每次调用的模型原始返回同时存到 `data/syn/raw_<ts>/<范式>_<模型>.txt`，
**解析失败的那份存成 `.failed.txt`** 并保留。这是排查问题的唯一依据 —— 数据对进了
JSONL 之后就看不出模型到底返回了什么（被 `max_tokens` 截断？键名漂移？结构不对？）。
`--no-save-raw` 可关闭。演示时若不需要留档，加这个开关即可。

**推理模型适配（两个脚本均适用）**：GLM-5.3 / Qwen3.6 的思维链写在 `reasoning_content`，
**同样消耗 `max_tokens` 预算**，且单次大 JSON 生成实测约 250 秒。故 `--max-tokens` 默认
16384、`--timeout` 默认 600 秒（勿低于 300）。预算不足时正文会被截断成半截 JSON，
`llm.py` 会检测 `finish_reason=length` 与空响应并直接报错提示调大预算 —— 同预算下重试
必然复现，故不再浪费 3 次重试；网络慢时可 `--timeout 900`。

## 2. 模型对比 compare_models.py

四步流水线，全部按 `prompts/model_comparison.md` 设计：

1. **生成**：同种子（默认模板附的 5 条）→ 三家各生成 `种子数 × --per-seed` 对；
2. **组装**：按模板输入格式生成 `{"datasets": {deepseek:[...], glm:[...], qwen:[...]}}`；
3. **互评**：deepseek / glm / qwen 各自按模板当评审，对四维打分、排名；
4. **汇总**：打印对比表 + 两种口径排名（全量均分 / **排除自评均分**）。

> ⚠ 评审团恰是三家被评模型自身，“自己给自己打分”天然有偏。
> 故汇总时除全量口径外，另算**非自评均分**（该家数据只由另外两家评审打分），
> 报告建议以该口径排名为准 —— 这也是本脚本相比直接调用模板更严谨的地方。

```bash
python compare_models.py                      # 默认：5 种子 × 2 对 × 3 家 = 各 10 对
python compare_models.py --per-seed 3         # 加大样本量
python compare_models.py --models glm qwen    # 只对比部分生成模型
python compare_models.py --judges deepseek glm
python compare_models.py --dry-run
python compare_models.py --timeout 900        # 网络慢时加大读超时（--max-tokens 见上节）
```

输出到 `data/compare/<时间戳>/`：
`gen_*.jsonl`（每家原始生成数据，可直接二次加工）、`judge_*.json`（每位评审完整打分与
理由）、`summary.json`（投票汇总与排名）；控制台打印 ASCII 对比表与结论。

## 3. 精选数据集导出 export.py

```bash
python export.py                  # 汇总 → 去重 → 校验 → 写数据集与数据卡
python export.py --dry-run        # 只统计与报告，不写文件
python export.py --keep-invalid   # 不剔除规则校验不通过的条目
python export.py --threshold 0.8  # MinHash 近重复阈值（默认即 0.8，需求指定）
```

输出到 `data/export/`：`syn_curated_<ts>.jsonl`（精选集，字段与 `generate.py` 一致）、
`DATA_CARD_<ts>.md`（数据卡）、`export_<ts>.log`（**被剔除的每一条 + 原因**，
保证「剔了什么」可审计、不是无声删除）。

**去重口径（关键）。** 分两道，判据不同：

| 判据 | 处理 | 为什么 |
|---|---|---|
| `(instruction, response)` 逐字相同 | 剔除 | 毫无争议的重复 |
| **`instruction`** 的 MinHash ≥ 0.8 | 剔除 | 同一条指令配多个不同回答，对 SFT 是有害数据 |

MinHash **只算 instruction、不把 response 拼进去**：种子改写增强的变体回答常常
逐字相同，拼进去会让这类**正常变体**拿到 0.74 的高相似度而被误删。
实测数据里 512 条去重前 4.5%、去重后 0%（需求要求 < 5%）；重复主要来自
**同模板同种子重复运行**，不是模型输出缺陷。

**规则校验按作用域解析，且承认能力边界。** 只核验**明确管整篇**的字数/行数约束
与格式约束（要求 JSON 就必须真能解析）。一旦发现约束限定在子部件上——
「principle 字段不超过 100 字」「每个要点不超过 20 字」「需求摘要不超过 50 字」
「时间复杂度分析不超过 100 字」——子部件无法机械切分，就记**未核验**并在数据卡里
如实写条数，**宁可标未核验也不误报**。中文「不超过 N 字」含不含标点本身有歧义，
故两种都算：两种都超才算确定违规（剔除），只有含标点超的记为**边界存疑**（保留）。

> 这条边界是踩出来的：第一版用字面子串匹配，45 条报 9 条违规、**7 条误报**；
> 扩到 512 条报 10 条、**6 条误报**，误报全部同因——约束管的是子部件不是整篇。
> `test_export.py` 把这两类误报场景逐条钉死。

**数据卡的「适用场景」是草稿，交付前需人工确认**；「来源/构成/已知偏差」全部由
数据统计生成，不做手工修饰。

> Alpaca / ShareGPT 格式转换由 `LLM_judge/convert.py` 负责，本脚本不重复实现。

## 与 LLM_judge 的关系

- `LLM_judge/`：语料（含生成后数据）的四维质量评估流水线 —— 已单独成仓库；
- `synthesis/`：**数据的产生（生成）与生成方对比**，二者结合即完整的
  “生成 → 评估 → 筛选 → 导出”闭环。
- 生成结果可直接喂给 `LLM_judge` 的 `pipeline.py`/`eval_loop.py` 做逐条筛选。

## 待办 / 已知限制

- `self_instruct.md` 模板未内嵌明确默认种子，脚本为它补了 5 条内置通用种子；
  建议用 `--seeds` 按自己领域指定以获得更好效果。
- 单次生成数量受模型单轮输出上限（`--max-tokens`）限制，需要大批量时建议
  多跑几轮（当前脚本一轮一文件/一调用），或后续加批次参数自动续生成。
- **提示词管不住"可数约束"（实测）**：模板里写了「写完回头数字数」这类自检规则后
  重跑，字数/行数超限反而从 1 条变成 2 条（限 5 行写了 7 行、限 30 字写了 33 字）。
  字数、行数、字段是否齐全这类**可机械核验**的约束，不能指望模型自觉 ——
  必须用代码卡。`export.py` 已承担这步规则校验；仍有 **24 处**约束因限定在子部件上
  而无法机械核验（数据卡里如实披露），这部分要靠人工抽查或 LLM-as-Judge。
- 同一模板、同一批种子跑多次会产生重复数据（实测同模板跑 3 次，45 条里 11 对
  被 MinHash≥0.8 判为重复）。`export.py` 的去重可消除，属事后补救；
  根治要靠生成侧跨批次去重，目前没有。
- **数据集的「有效性」尚无实验支撑**：需求验证方案里的「精选 20% ≥ 全量 95%」
  还没跑。数据卡第 6 条已如实标注。
