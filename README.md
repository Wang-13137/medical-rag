# 中文医疗知识库 RAG 系统（W1 阶段）

面向 2027 届校招算法岗的转码作品集项目。整个项目的差异化不在「能问答」，而在配套一套可量化的评测体系（准确率、幻觉率、拒答正确率）。本仓库截至当前只实现 W1：文档解析、切片、向量化、FAISS 检索基线。

---

## 一、本阶段已完成什么

| 模块 | 文件 | 职责 |
| --- | --- | --- |
| 配置 | `configs/config.yaml` | 全部可调参数外置，代码内无魔法数字 |
| 解析 | `src/medical_rag/ingest.py` | PDF（pdfplumber，带页码）、docx（python-docx，含表格）、txt、md，统一为带页码的行序列 |
| 切片 | `src/medical_rag/chunking.py` | 标题层级优先，超长段落再走字符滑窗；保留文件名、路径、页码、标题链、字符区间 |
| 向量化 | `src/medical_rag/embedding.py` | bge-small-zh-v1.5，批量推理，query 侧自动加指令前缀，输出 L2 归一化 |
| 索引 | `src/medical_rag/index.py` | FAISS IndexFlatIP，落盘与重载，元数据与向量行号严格对齐 |
| 检索 | `src/medical_rag/retrieve.py` | 单条与批量检索，输出带分数和溯源串的结果 |
| 混合检索 | `src/medical_rag/hybrid.py` | BM25（rank_bm25 + jieba，保留数字与英文缩写）+ 向量，用 RRF 按排名融合；分词结果带指纹缓存 |
| 精排 | `src/medical_rag/rerank.py` | `bge-reranker-base` cross-encoder，对「向量宽召回 → 交互式打分 → 取 Top-K」做二阶段检索 |
| 脚本 | `scripts/build_index.py`、`scripts/query.py`、`scripts/clean_corpus.py`、`scripts/answer.py` | 建索引、命令行查询、语料清洗、带引用的问答 |
| 评测 | `scripts/eval_chunk_params.py`、`scripts/eval_answer_recall.py`、`scripts/compare_retrievers.py`、`scripts/compare_generation.py` | 切片参数对比、答案关键串召回、四种检索策略横向对比、生成策略配对显著性检验 |
| 评测集 | `scripts/build_benchmark.py`、`scripts/eval_benchmark.py`、`data/eval/benchmark_v1.json` | 100 条带标准答案的评测集：构建 + 原文校验，以及在上面跑 baseline（详见第十三节） |
| 生成 | `src/medical_rag/generate.py` | 受限提示词的答案合成，强制引用编号与无依据时拒答；内置离线替身 `EchoGenerator` |
| 生成评测 | `scripts/run_generation.py`、`scripts/eval_generation.py` | 在 benchmark 上批量生成并评测正确性、引用可溯源性、语境支持度、拒答权衡 |

**未完成（后续阶段）**：Gradio 演示界面、FastAPI 服务化、把「Constraint Decoding」式的 hallucination 拦截做成硬闸门。

---

## 二、环境准备

### 1. Python 环境

统一使用隔离 venv，不要安装到系统 Python：

```powershell
python -m pip install -r requirements.txt
```

本机配置：**12 核 CPU、Windows、无 GPU**。

### 2. 依赖安装结果（实测）

| 包 | 版本 | 说明 |
| --- | --- | --- |
| torch | 2.14.1+cpu | **已存在，未被升级或覆盖**，与新依赖无冲突（实测 `torch.cuda.is_available()` 为 False，符合预期） |
| numpy | 2.5.3 | 已存在，未被降级 |
| faiss-cpu | 1.15.1 | 新增 |
| sentence-transformers | 6.1.0 | 新增，连带安装 transformers 5.19.0、tokenizers 0.23.3、huggingface-hub 1.33.0 |
| modelscope | 1.40.2 | 新增，用于下载 bge 模型 |
| pdfplumber | 0.11.10 | 新增 |
| python-docx | 1.2.0 | 新增 |
| rank_bm25 | 0.2.2 | 新增，W2 才用到，本阶段只装 |
| jieba | 0.42.1 | 新增，W2 才用到，本阶段只装 |
| pytest | 9.1.1 | 已存在 |

### 3. Embedding 模型下载与缓存位置

需要的模型：**BAAI/bge-small-zh-v1.5**（约 130 MB，实际下载 191 MB，快照内含 `pytorch_model.bin` 与 `model.safetensors` 两份权重）。

下载过程实测：

1. 先尝试 `HF_ENDPOINT=https://hf-mirror.com`，失败，报 `httpx.ProxyError: 502 Bad Gateway`；
2. 再尝试直连 `huggingface.co`，同样不可达；
3. 最后改用 ModelScope 成功：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"          # 可选，本机已验证无效
python -m pip install modelscope
python -c "from modelscope import snapshot_download; print(snapshot_download('BAAI/bge-small-zh-v1.5'))"
```

**缓存路径（README 记录项）**：

```
%USERPROFILE%\.cache\modelscope\models\BAAI--bge-small-zh-v1.5\snapshots\master
```

目录内含 `config.json`、`modules.json`、`1_Pooling/config.json`、`tokenizer.json`、`vocab.txt`、`sentence_bert_config.json` 与模型权重，`SentenceTransformer` 可直接加载。

代码内的查找顺序（`embedding.py` 的 `resolve_model_dir`）：

1. 配置里的 `embedding.local_model_dir`（显式指定的本地目录）；
2. ModelScope 本地缓存（离线命中，不发网络请求）；
3. HuggingFace Hub 本地缓存；
4. 联网从 HF 下载；
5. 联网从 ModelScope 下载。

因此本机在**断网状态下也能直接加载**：实测设置 `HF_HUB_OFFLINE=1` 后 `SentenceTransformer` 仍能正常出 512 维向量。若要把模型挪位置（例如放到项目里），只需把一个同结构目录的路径填进 `configs/config.yaml` 的 `embedding.local_model_dir`。

---

## 三、准备数据

支持的格式：`.pdf`、`.docx`、`.txt`、`.md`、`.markdown`，会递归扫描子目录。

```
data/raw/       ← 真实知识库，由你自己放入
data/smoke/     ← 20 条样例条目（见下节说明）
```

如果原始数据是扫描版 PDF（没有文本层），pdfplumber 取不到文字，需要先做 OCR，本阶段未实现。

### 当前 `data/raw` 里放的是什么

16 份真实的中文临床指南／专家共识全文（纯文本，1,693,363 字符），出处与来源等级见 **第十节**，
采集来源清单机器可读版本在 `data/raw/sources_manifest.json`。原始采集物（PDF/HTML）归档在
`data/source_original/`，不在版本库内。

---

## 四、冒烟测试数据说明

**`data/smoke/` 下的内容仅为样例数据，用于验证链路是否跑通，不是真实医疗知识库，不可作为任何医学参考或临床依据。**

共 20 条条目，覆盖四类题材：慢病诊断标准、控制目标、用药注意事项、检验参考区间。

| 文件 | 条目数 | 题材 |
| --- | --- | --- |
| `内分泌代谢.md` | 5 | 2 型糖尿病诊断标准、血糖控制目标、HbA1c、二甲双胍、糖尿病足筛查 |
| `心血管疾病.md` | 5 | 高血压诊断分级、血压测量方法、降压目标、血脂分层、他汀注意事项 |
| `用药安全.md` | 4 | 华法林 INR 监测、非甾体抗炎药禁忌、抗菌药物原则、老年多重用药 |
| `检验参考区间.md` | 5 | 血常规、肝功能、肾功能 eGFR、电解质、甲状腺功能 |
| `呼吸与感染.md` | 1 | 社区获得性肺炎诊断要点 |

条目中的数值为常识性量级，未逐条核对最新指南。`data/smoke/README.md` 也会被一并入库，因此索引里会有 1 条说明性切片，这是故意保留的无关内容，方便后面调检索阈值时当干扰项。

---

## 五、建索引

```powershell
cd <repo-root>

# 用 smoke 样例数据跑通全流程
python scripts/build_index.py --smoke

# 用 data/raw 的真实语料建索引（默认行为）
python scripts/build_index.py

# 指定其它目录与输出位置
python scripts/build_index.py --data-dir D:/医疗文献 --index-path data/index/main.index --meta-path data/index/chunks_main.json
```

实测输出：

```
[build] 数据目录：...\medical-rag\data\smoke
[build] 解析完成：6 份文档，共 5053 字符
[build] 切片完成：21 片，平均 243 字符（chunk_size=400, overlap=80）
[embedding] 命中 ModelScope 缓存：...\BAAI--bge-small-zh-v1.5\snapshots\master
[build] 向量化完成：shape=(21, 512), dtype=float32
[build] 向量 L2 范数检查：min=1.000000, max=1.000000
[build] 索引已写入：data\index\faiss_smoke.index（0.04 MB，21 条）
[build] 元数据已写入：data\index\chunks_smoke.json
```

产物是两个文件：FAISS 二进制索引，以及配套的切片元数据（`chunks_smoke.json`）。元数据里保存了每个切片的文件名、路径、页码、标题链、字符区间，以及建索引时用的模型与切片参数，用于引用溯源和排查「索引与模型不匹配」。

---

## 六、查询

```powershell
python scripts/query.py --q "2型糖尿病血糖控制目标"
python scripts/query.py --q "他汀类药物的禁忌" --top-k 3
python scripts/query.py --q "..." --full            # 打印完整文本而不是截断预览
```

实测输出（`score` 是余弦相似度）：

```
[query] query：2型糖尿病血糖控制目标
[query] 命中 3 条（索引共 21 条，score 为余弦相似度）
[1] score=0.7326  内分泌代谢.md md · 内分泌代谢 > 2型糖尿病的血糖控制目标
    内分泌代谢 > 2型糖尿病的血糖控制目标 对大多数成年患者，推荐的控制目标为：空腹血糖 4.4 至 7.0 mmol/L，
    非空腹血糖小于 10.0 mmol/L，糖化血红蛋白 HbA1c 小于 7.0%。病程短、无并发症、预期寿命长的中青年患者
    可进一步收紧到 HbA1c 小于 6.5%。...
[2] score=0.5887  内分泌代谢.md md · 内分泌代谢 > 2型糖尿病的诊断标准
[3] score=0.5522  内分泌代谢.md md · 内分泌代谢 > 糖化血红蛋白的临床意义
```

Top-1 命中预期条目，且明显高于第二名的 0.5887。

---

## 七、配置项说明

全部参数集中在 `configs/config.yaml`，命令行 `--data-dir`、`--index-path`、`--top-k` 等可覆盖同名项。

| 配置路径 | 默认值 | 含义 |
| --- | --- | --- |
| `paths.data_dir` | `data/raw` | 真实语料目录 |
| `paths.smoke_data_dir` | `data/smoke` | 样例数据目录 |
| `paths.index_path` | `data/index/faiss_main.index` | FAISS 索引文件（正式索引，见第十节） |
| `paths.meta_path` | `data/index/chunks_main.json` | 切片元数据 |
| `chunking.chunk_size` | 300 | 切片最大字符数（实验选出，见第十一节） |
| `chunking.chunk_overlap` | 60 | 相邻切片重叠字符数 |
| `chunking.heading_max_chars` | 60 | 超过此长度的行不算标题 |
| `chunking.prepend_heading_path` | true | 是否把标题链写进切片正文 |
| `embedding.model_name` | `BAAI/bge-small-zh-v1.5` | 向量模型 |
| `embedding.device` | cpu | 推理设备 |
| `embedding.batch_size` | 16 | 批大小 |
| `embedding.normalize` | true | L2 归一化，配合 IndexFlatIP 得到余弦相似度 |
| `embedding.query_instruction` | `为这个句子生成表示以用于检索相关文章：` | bge 官方查询前缀 |
| `embedding.local_model_dir` | 空 | 显式指定本地模型目录 |
| `retrieval.top_k` | 5 | 默认返回条数 |
| `retrieval.use_hybrid` | false | BM25+向量 RRF 混合。实测低于纯向量，默认关闭（见第十二节） |
| `retrieval.hybrid.rrf_k` | 60 | RRF 常数，越大越弱化头部名次差距 |
| `retrieval.hybrid.candidate_multiplier` | 4 | 每路候选数 = max(top_k, 20) × 该系数 |
| `rerank.enabled` | true | 是否启用 cross-encoder 精排 |
| `rerank.model_name` | `BAAI/bge-reranker-base` | 重排模型 |
| `rerank.candidate_k` | 20 | 第一阶段宽召回条数，精排的候选池 |
| `generation.enabled` | true | 是否调用真实生成接口；false 时用离线替身 `EchoGenerator` |
| `generation.model` | `deepseek-chat` | 生成模型 |
| `generation.temperature` | 0.0 | 固定为 0，保证评测可复现 |
| `generation.budget_chars` | 2400 | 送进提示词的上下文字符预算 |
| `generation.top_n` | 8 | 送进提示词的片段数上限 |
| `runtime.num_threads` | 12 | CPU 推理线程数 |

生成接口只从环境变量 `DEEPSEEK_API_KEY` 读取密钥，配置文件里不落盘密钥：

```powershell
$env:DEEPSEEK_API_KEY = "sk-xxxx"      # PowerShell
export DEEPSEEK_API_KEY="sk-xxxx"      # bash
```

---

## 八、测试

```powershell
python -m pytest tests -q
```

实测：**33 项全部通过**（约 17 秒，模型加载占大头）。

| 测试文件 | 覆盖内容 |
| --- | --- |
| `tests/test_chunking.py` | 长文本切成多片、相邻切片逐字符重叠正确、窗口步长等于 `chunk_size - overlap`、标题链与页码保留、短小节独立成片、六类标题识别规则、非法 overlap 报错 |
| `tests/test_embedding.py` | 输出 512 维、query 侧自动拼指令前缀（与手动拼前缀结果一致、与不拼明显不同、重复前缀不叠加）、相同文本向量完全一致、批量输出顺序正确、语义相近文本相似度更高、空批次不崩溃 |
| `tests/test_faiss_index.py` | 保存后重载的命中顺序与分数逐位一致、溯源字段往返不丢失、缺失文件与维度/数量不一致时报错、自身向量检索必命中自己、结果降序、`top_k` 超过语料规模自动裁剪 |
| `tests/test_smoke_retrieval.py` | 用 smoke 数据走完整条链路，5 组 query 断言 Top-1 就是预期条目；溯源串、批量与单条结果一致、空 query 报错 |

---

## 九、目录结构

```
medical-rag/
├── README.md
├── requirements.txt
├── configs/config.yaml
├── src/medical_rag/
│   ├── __init__.py
│   ├── config.py          # 配置读取（"a.b.c" 路径取值、命令行覆盖）
│   ├── ingest.py
│   ├── chunking.py
│   ├── embedding.py
│   ├── index.py
│   ├── retrieve.py
│   ├── hybrid.py          # BM25 + 向量 RRF 混合
│   ├── rerank.py          # cross-encoder 二阶段精排
│   └── generate.py        # 答案合成（受限提示词 + 引用编号 + 拒答）
├── data/
│   ├── raw/               # 原始语料（含网页样板），不进版本库
│   ├── clean/             # 清洗后的语料（scripts/clean_corpus.py 产出，实际入库）
│   ├── source_original/   # 原始采集物（PDF/HTML）归档，不进版本库
│   ├── smoke/             # 20 条样例条目（仅用于冒烟测试）
│   ├── index/             # 构建产物，不进版本库
│   └── eval/              # 评测集：标注源文件 + 构建产物
│       ├── queries_v1.yaml      # 人工标注源文件（100 题）
│       └── benchmark_v1.json    # 构建产物：关键串经原文校验
├── reports/               # 实验结果 JSON（切片参数对比、答案关键串召回、策略横向对比、benchmark 跑分）
├── scripts/
│   ├── build_index.py
│   ├── query.py
│   ├── clean_corpus.py           # 网页样板清洗
│   ├── eval_chunk_params.py      # 三档切片参数对比（文档级指标）
│   ├── eval_answer_recall.py     # 等上下文预算下的答案关键串召回
│   ├── compare_retrievers.py     # 四种检索策略横向对比
│   ├── convert_gout_sources.py   # HTML/PDF 原始指南转 txt
│   ├── mine_eval_material.py     # 从语料抽取命题素材（供人工出题核对）
│   ├── build_benchmark.py        # 构建评测集并强制校验答案出处
│   └── eval_benchmark.py         # 在评测集上跑 baseline 指标
└── tests/
    ├── conftest.py
    ├── test_chunking.py
    ├── test_embedding.py
    ├── test_faiss_index.py
    ├── test_smoke_retrieval.py
    ├── test_hybrid.py
    ├── test_rerank.py
    └── test_benchmark.py         # 题库自身的校验（关键串可溯、拒答词零出现）
```

---

## 十、真实语料：数据来源、等级与合规

### 来源清单

| 文件 | 指南名称 | 发布机构 | 出处 | 来源等级 |
| --- | --- | --- | --- | --- |
| `stroke_ais_2023.txt` | 中国急性缺血性卒中诊治指南2023 | 中华医学会神经病学分会、脑血管病学组 | 中华神经科杂志 2024;57(6) | official_publisher |
| `cap_2016.txt` | 中国成人社区获得性肺炎诊断和治疗指南(2016年版) | 中华医学会呼吸病学分会 | 中华结核和呼吸杂志 2016;39(4) | official_publisher |
| `asthma_2024.txt` | 支气管哮喘防治指南（2024年版） | 中华医学会呼吸病学分会哮喘学组 | 中华结核和呼吸杂志 2025;48(3):208-248 | official_publisher |
| `hp_2022.txt` | 2022中国幽门螺杆菌感染治疗指南 | 中华医学会消化病学分会幽门螺杆菌学组 | 中华消化杂志 2022;42(11) | official_publisher |
| `hp_consensus6_2022.txt` | 第六次全国幽门螺杆菌感染处理共识报告(非根除治疗部分) | 中华医学会消化病学分会幽门螺杆菌学组 | 中华消化杂志 2022;42(5) | official_publisher |
| `pediatric_fever_2026.txt` | 儿童发热全程管理专家共识（2026版） | 中华医学会感染病学分会等 | 中华医学杂志 2026;106(12) | official_publisher |
| `appendicitis_pregnancy_2026.txt` | 妊娠合并急性阑尾炎诊治专家共识（2026） | 中华医学会妇产科学分会产科学组等 | 中华围产医学杂志 2026;29(4) | official_publisher |
| `appendicitis_ultrasound_2025.txt` | 急性阑尾炎超声检查及报告规范专家共识（2025版） | 中华超声医学培训工程等 | 中华医学超声杂志(电子版) 2025;22(6) | official_publisher |
| `appendicitis_wses_2020.txt` | Diagnosis and treatment of acute appendicitis: 2020 update of the WSES Jerusalem guidelines | World Society of Emergency Surgery | World Journal of Emergency Surgery 2020;15:27 | official_publisher |
| `hyperuricemia_gout_2019.txt` | 中国高尿酸血症与痛风诊疗指南(2019) | 中华医学会内分泌学分会 | 中华内分泌代谢杂志 2020;36(1):1-13 | official_publisher |
| `hyperuricemia_gout_2024.txt` | 中国高尿酸血症与痛风诊疗指南(2024) 第一部分+第二部分 | 中华医学会内分泌学分会 | 中华内分泌代谢杂志 2025;41(10)、41(11) | third_party_mirror |
| `primary_diabetes_2025.txt` | 国家基层糖尿病防治管理指南（2025） | 国家卫生健康委基层卫生健康司指导、中华医学会 | 中华内科杂志 2025;64(12) | third_party_mirror |
| `heart_failure_2024_official.txt` | 中国心力衰竭诊断和治疗指南2024 | 中华医学会心血管病学分会等 | 中华心血管病杂志 2024;52(3) | official_publisher |
| `hypertension_clinical_practice_2022.txt` | 中国高血压临床实践指南 | 国家心血管病中心等 | 中华心血管病杂志 2022;50(11) | official_typeset_pdf |
| `primary_hypertension_2025.txt` | 国家基层高血压防治管理指南2025版 | 国家心血管病中心等 | 中国循环杂志 2025;40(9) | official_typeset_pdf |

合计 16 份：14 份官方全文，2 份第三方转载的完整版（痛风 2024 官方原文在付费墙后，只能用转载版；基层糖尿病 2025 官方页同样只放出摘要，实际入库的是内容一致的第三方文件）。

### 合规与用途说明

1. 这些指南均为**公开发表的学术文献**，本项目只在本地做技术验证，**不重新分发原文**：`data/raw` 与 `data/source_original` 已写入 `.gitignore`，不进版本库。
2. 检索结果的医学内容**不能作为临床决策依据**。本项目到 W2 之前没有生成模块，输出的是原始片段；即使后面接上生成模型，也只用于演示检索增强流程。
3. 语料存在**年份分布不均**（覆盖 2016 至 2026，部分领域只有旧版），因此「检索不到」不等于「指南没有写」，只等于「本库没收录」。

### 正式索引（真实语料）

```powershell
python scripts/build_index.py     # 读 data/raw，写 data/index/faiss_main.index
```

实测输出：

```
[build] 解析完成：16 份文档，共 1693363 字符
[build] 切片完成：7147 片，平均 301 字符（chunk_size=300, overlap=60）
[build] 向量化完成：shape=(7147, 512), dtype=float32
[build] 向量 L2 范数检查：min=1.000000, max=1.000000
[build] 索引已写入：data/index/faiss_main.index（13.96 MB，7147 条）
```

### 语料清洗：PubReader 网页样板

`scripts/clean_corpus.py` 按两条可解释的规则清洗（详见第十二节），产物落在 `data/clean/`：

```
[clean] 语料 16 份；跨文档重复行（df>=3）黑名单 436 条
[clean] 合计：删除 6105 行，字符 1699646 -> 1572542（砍掉 7.48%）
[build] 解析完成：16 份文档，共 1569107 字符
[build] 切片完成：6625 片，平均 301 字符（chunk_size=300, overlap=60）
[build] 索引已写入：data/index/faiss_clean_base.index（12.94 MB，6625 条）
```

### 检索实测

默认策略为「向量召回 Top-20 + `bge-reranker-base` 精排」，`scripts/query.py` 实测：

| query | Top-1 来源 | 精排得分 | 精排前名次 |
| --- | --- | --- | --- |
| 支气管哮喘的长期控制药物如何选择 | `asthma_2024.txt` · 哮喘慢性持续期的治疗 > 制订治疗方案 | 4.753 | 第 7 名，精排后升到第 1 |
| 慢性心力衰竭的药物治疗方案包括哪些药物 | `heart_failure_2024_official.txt` · 急性心衰的治疗 > 药物治疗 | 5.678 | —— |

第二条顺带說明一个已知问题：query 问「慢性」心衰，Top-1 落在「急性心衰」小节——这套检索还**分不清
同一份指南内部的临床情境差异**，是当前最明显的失败模式。

---

## 十一、切片参数是怎么定下来的

没有拍脑袋选参数，跑了两轮实验，脚本和结果都在仓库里：`scripts/eval_chunk_params.py`、
`scripts/eval_answer_recall.py`，原始结果在 `reports/*.json`。

### 第一轮：文档级指标（失效）

11 条探针 query，每条标注「答案应当出自哪份指南」，用 `bge-small-zh-v1.5` + FAISS 检索 Top-1/Top-3：

| 切片配置 | 切片数 | 均长 | 超过 512 字符占比 | 文档级 Recall@1 | Recall@3 | MRR@5 | 编码耗时 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 300 / 60 | 7147 | 301 | 0.0% | 1.000 | 1.000 | 1.000 | 105.3s |
| 400 / 80 | 5420 | 394 | 0.0% | 1.000 | 1.000 | 1.000 | 112.1s |
| 600 / 120 | 3700 | 570 | **89.2%** | 1.000 | 1.000 | 1.000 | 122.2s |

三档全是满分，说明这个指标已经**饱和**——每份指南主题单一，query 里带病名就能认出文档，**失去了区分能力**。这一轮的结论不是「随便选」，而是「必须换更细的指标」。

顺带得到一个硬约束：600 字符档有 89.2% 的切片超过模型 `max_seq_length=512`，尾部会被截断，信息静默丢失。

### 第二轮：答案关键串召回（有区分度）

把每条 query 的答案拆成关键实体（数值、药名、检查项，例如溶栓题是「4.5 / 静脉溶栓 / 阿替普酶」），
看**定长上下文预算**内能否把这些实体全召回来。两个公平性处理：

- 关键串必须真实出现在语料正文里，脚本先核验，查不到的自动剔除；
- 三档的检索条数按等上下文预算换算（1200 字符 → 300 档取 4 片、400 档取 3 片、600 档取 2 片），
  避免长切片因为塞进更多文本而天然占便宜。

| 切片配置 | 预算内条数 | 关键串全中率 AnswerRecall | 平均覆盖率 Coverage |
| --- | --- | --- | --- |
| **300 / 60** | Top-4 | **0.500** | **0.633** |
| 400 / 80 | Top-3 | 0.400 | 0.600 |
| 600 / 120 | Top-2 | 0.300 | 0.500 |

差距不大但方向一致且单调，**300/60 胜出**，因此写进 `configs/config.yaml`。

### 这轮实验暴露的真问题

探针里 3 条 query 在三档配置下**关键串覆盖率都是 0**（社区获得性肺炎的经验性抗感染治疗、
慢性心力衰竭的药物治疗方案、基层糖尿病筛查与转诊；对应的具体药名检查项是喹诺酮／阿莫西林／克拉维酸、
沙库巴曲／达格列净、空腹血糖／OGTT，这些串都真实存在于对应指南里）：

- 文档找对了，但召回的是概述或背景段，**真正的推荐条目没排进预算内**；
- `prepend_heading_path` 把标题链拼进了正文，对「找对文档」帮助大，对「找对具体条款」帮助有限。

这是 W2 要解决的问题：BM25 + 向量混合检索（依赖已装）、rerank，以及按「推荐意见」结构重新组织切片。
本轮的 10 条 query 只作诊断用，样本量太小，不能当 benchmark 对外汇报。

---

## 十二、检索：为什么最终选了「向量召回 + cross-encoder 精排」

这一节记录的是**包括负结果在内的完整过程**。三件事：先清洗语料，再横向对比四种检索策略，
最后按实测结果定默认方案。

### 1. 语料清洗（PubReader 网页样板）

上一节说的 UI 噪声怎么来的：一部分指南是从出版商网页（中华医学期刊网等）抓下来的，转 txt 时
导航栏、备案号、版权声明、甚至「AI 问答」挂件的话术都被当成正文收了进来。混合检索上线后，
几条 query 的 Top-3 直接被「高级检索 / 上一篇 下一篇 / 暂无评论」占掉，才暴露出来。

清洗规则（两条，都可解释，不用黑箱模型）：

1. **跨文档重复行**：同一行在 ≥ 3 份不同文档里都出现 → 判为样板。依据：正版医学内容不会在
   十几份不同指南里逐字重复，只有网站框架会。
2. **备案/版权/挂件正则**：ICP 备案、网信算备、版权所有、隐私政策、SparkDesk 用户协议等。

删掉了什么（抽样核对过，不误伤正文）：参考编号 `[123]`、`返回引文位置Google Scholar`、
`帮助中心`、`历史对话`、`Copyright by Chinese Medical Association`、相关阅读标题列表等。
**误伤检查**：长度 > 40 字且被删的行一共只有 34 条，逐条看全是版权声明和「相关阅读」外链标题，
没有一条是正文。合计 6,105 行、字符 1,699,646 → 1,572,542（7.48%），索引 7,147 片 → 6,625 片。

### 2. 四种检索策略横向对比

脚本 `scripts/compare_retrievers.py`，10 条探针 query + 答案关键串，同一索引同一预算：

| 策略 | 做法 | 预算 1200（Top-4）AnswerRecall | Coverage | 预算 2400（Top-8）AnswerRecall | Coverage |
| --- | --- | --- | --- | --- | --- |
| `vector` | 纯向量 | **0.500** | 0.633 | 0.500 | 0.667 |
| `hybrid_rrf` | BM25 与向量各自召回后 RRF 融合 | 0.400 | 0.600 | 0.400 | 0.633 |
| `bm25_rerank` | 向量宽召回，BM25 在候选集内重排 | 0.400 | 0.500 | 0.400 | 0.567 |
| `cross_encoder` | 向量宽召回 Top-N，`bge-reranker-base` 精排 | **0.500** | **0.667** | **0.600** | **0.700** |

**负结果要写清楚：两种 BM25 方案都是负收益**，而且换预算、换索引都一样方向的下降。诊断结论：

- 中文临床指南里「基层医疗卫生机构」「推荐使用」这类短语在**单份文档内部**反复出现，
  BM25 会被这种局部高频短语主导；
- 于是「适用范围」「服务要求」这类章节被顶到前排，真正写着推荐条款的片段被挤出上下文预算；
- 也就是说，**瓶颈是「同一文档内部的段落判别」，不是「找对文档」** —— 文档级指标早就满分了。
  稀疏的字面匹配解决不了这件事。

`cross_encoder` 是全项目唯一有正向收益的改动：把 query 和候选段拼在一起做交互式打分，
对「这段到底答没答这个问题」敏感得多；单次 20 个候选对的精排在本机 CPU 上约 0.26 秒，可用。

因此默认配置是 `retrieval.use_hybrid: false` + `rerank.enabled: true`；
混合检索的开关和代码保留，就是为了把上面这组对照重现出来，不会被悄悄删掉。

### 3. 剩下的失败模式（下一步要解决的）

2 条 query 在四种策略下关键串覆盖率恒为 0：

| query | 缺的关键串 | 原因分析 |
| --- | --- | --- |
| 慢性心力衰竭的药物治疗方案包括哪些药物 | 沙库巴曲 / 达格列净 / ARNI | 答案横跨多个小节；实测 Top-1 落在「急性心衰」而非「慢性心衰」，模型没学会区分同一指南内的临床情境 |
| 基层医疗卫生机构2型糖尿病的筛查和转诊建议 | 空腹血糖 / OGTT / 糖化血红蛋白 | query 里的「基层医疗卫生机构」在该指南里是高频套话，字面匹配直接被带偏 |

10 条探针样本量太小，只够做诊断，不够做结论。因此有了**第十三节的 100 条 benchmark**。

---

## 十三、评测集 benchmark v1（100 条带标准答案）

探针能指出方向，但撑不起结论。这一节把它扩成了可对外汇报的评测集：
`data/eval/benchmark_v1.json`（85 条可回答 + 15 条应拒答），配套脚本
`scripts/build_benchmark.py`（构建 + 校验）与 `scripts/eval_benchmark.py`（跑分）。

### 十三.1 铁律：答案必须能在原文里搜到

医疗问答评测最容易糊弄的地方是「答案看起来对」——写题的人凭医学常识写个答案，
但语料里根本没这句话，评测就成了自我实现的预言。这里的做法是：

- 每题标注 `gold_doc`（答案应当出自哪份指南）和 `answer_keys`（答案关键串，药名、剂量、阈值、时间窗）。
- **构建时逐条用 `answer_keys` 去 `gold_doc` 原文里做子串搜索，搜不到就整条剔除**，
  145 个关键串零例外（`scripts/build_benchmark.py`）。
- 匹配前做 `NFKC` 折叠 + 去空白。这不省事但必须做：有些 PDF 抽出的文档会在每个字符间塞空格
  （`１ ４ ０ ｍｍＨｇ`），有些网页版用全角字符（`７~１０ ｄ`、`ｍｇ/ｄ`），不做归一化的话两侧永远对不上。

这条铁律当场生效：首轮校验就打回 1 条（QA066 痛风一线用药）。原因不是答案写错，而是那页是
**双栏 PDF**，提取时左右栏文字穿插，`...无症状高尿｜苯溴马隆的降尿酸疗效...｜酸血症患者...`
被别的栏拦腰截断，整句在原文里根本不存在。最终改用推荐意见汇总表里的连续写法。

### 十三.2 拒答题不能想当然

另一类更容易造假的是「应拒答」题。直觉上写个冷门病就行，但核验后发现
**多数冷门病其实在语料里零星出现过**：

| 我以为没有 | 实际出现 | 出现在哪 |
| --- | --- | --- |
| 靶向药 | 49 次 | 哮喘指南讲生物靶向药物 |
| 癫痫 | 22 次 | 卒中、儿科指南 |
| 前列腺 | 7 次 | 其实是「前列腺素」「前列腺肥大」 |
| 青光眼 | 5 次 | 作为激素禁忌症被列举 |
| 质谱 | 5 次 | 高血压指南讲串联质谱法测醛固酮 |

所以每条拒答题都挂了 `probe_terms`，**构建时强制要求在全语料零出现**，零出现才算数；
「前列腺」这种情况就换成零出现的「前列腺增生 / 坦索罗辛 / 非那雄胺」。最终 15 条拒答题分两类：
11 条主题缺失（银屑病、帕金森病、孤独症、多动症等），4 条越界（个体化处方、剂型换算、
无法推断的未来事项、跨领域提问）。

留了一条刻意的陷阱 QU015「MALDI-TOF 质谱的峰对齐 ppm 容差该设多少」：语料里确实有 5 处「质谱」
字样，但都是背景引用，答不了这个问题。词面命中而实质无关，是最难的一种幻觉来源。

### 十三.3 构成

| 维度 | 分布 |
| --- | --- |
| 可回答题 | 85 条（numeric 64 / fact 12 / disambiguation 9） |
| 应拒答题 | 15 条 |
| 覆盖文档 | 16 份全部有题（最少 2 条，最多 10 条） |
| 关键串 | 145 个，全部通过原文校验 |

### 十三.4 Baseline（等上下文预算 2400 字符，最多 8 片）

| 策略 | doc_hit@1 | MRR | AnswerRecall | 全中率 |
| --- | --- | --- | --- | --- |
| 纯向量 | 0.918 | 0.951 | 0.794 | 0.765 |
| 向量 + cross-encoder 精排 | **0.965** | **0.971** | **0.912** | **0.906** |

分题型看，最难的是情境消歧题，而精排对它的提升也最大：

| 题型 | 纯向量 Recall | 精排 Recall |
| --- | --- | --- |
| numeric (64) | 0.844 | 0.922 |
| fact (12) | 0.792 | 0.917 |
| disambiguation (9) | 0.444 | **0.778** |

### 十三.5 精排的真正价值：省钱

把上下文预算放开到 4 倍（9600 字符 / 32 片），纯向量的表现：

| 预算 | 纯向量 Recall | 全中率 |
| --- | --- | --- |
| 2400 字符 | 0.794 | 0.765 |
| 4800 字符 | 0.859 | 0.847 |
| 9600 字符 | 0.894 | 0.882 |

也就是说，**精排用 2400 字符预算（0.912）跑赢了纯向量 9600 字符预算（0.894）**——
上下文开销只有四分之一，效果反而更好。这直接决定了后续接生成模块时的成本结构。

### 十三.6 剩下的 8 条硬骨头，以及真正的根因

精排后仍有 8 条未全中，其中 **6 条 `rank=1`，即文档完全找对了**。逐一核对原文后定位到一个具体问题：

> 部分推荐意见是**超过 300 字符的长句**，正好跨过切片边界。例如「建议在 H.pylori 感染初次和再次
> 根除治疗中使用铋剂四联方案，**疗程为 14 d**」——前半句进了切片 A，`疗程为 14 d` 落到切片 B；
> 切片 B 只剩长句尾巴，与问题语义几乎无关，排序自然进不了 Top-8。

典型例子：`QA045` 的「铋剂四联方案」命中、「疗程为14 d」没命中；`QA018` 泼尼松剂量、
`QA037` 地高辛血药浓度范围、QA042 液体出入量同理。

先前猜测的「PDF 表格串行」不是主因——核验了这 8 条的原文位置，只有 1 条实际落在制表符表格里。
**真正的根因是切片不做句边界对齐**（也正好印证了已知限制第 1 条）。

### 十三.7 拒答阈值：能用，但别指望它

生成模块还没接，这里先用检索置信度做「系统是否会硬答」的代理，扫描阈值看两类题能否分开：

| 策略 | 阈值 | 拒答正确率 | 可答题误拒率 |
| --- | --- | --- | --- |
| 纯向量（余弦） | 0.6548 | 0.867 | 0.118 |
| 精排（logit） | 5.3896 | 1.000 | **0.224** |

结论是**负面的**：精排分数能做到 15 条拒答题全对，代价是误拒 22% 的可答题；余弦次之。
15 条拒答题的 Top-1 分数分布在 0.527~0.748，和真实可答题大量重叠——
最刺眼的是 QU014「2028 年会不会发布新版卒中指南」，Top-1 余弦 0.748 直接命中卒中指南，
看起来很像能答。

**单靠检索分数做拒答闸门不可靠**，这个问题要在生成阶段结合「答案里有没有可引用的原文」一起解决。

---

## 十四、已知限制

1. **切片只用字符数，不做句边界对齐**：滑窗按字符切，长句会被拦腰截断。这不是理论担忧——benchmark v1 里剩余 8 条未全中的题有 6 条文档找对了却答不全，根因就是这个（见第十三.6）。下一步按句队列打包，让每个切片自然收尾在句末。
2. **跨页 Section 只记起始页码**：一个标题下的正文跨多页时，页码取第一页。后续可在 Section 内存 `page_range`。
3. **PDF 表格提取未处理**：pdfplumber 的表格数值目前混在正文里，排版较乱的表格会串行；没有 OCR，扫描件不可用。
4. **参考文献区块没单独处理**：清洗脚本删掉了参考编号和「返回引文位置」这类样板，但参考文献的正文（英文期刊名、作者列表）仍在库里，会稀释密度。下一步可以按 `[参考文献]` 标题切断，单独存放或不入库。
5. **检索分不清同一指南内部的临床情境**：实测「慢性心力衰竭的药物治疗」Top-1 落在「急性心衰」小节。切片已经把标题链写进正文，说明光靠标题文本不够，需要按情境（急性/慢性、成人/儿童）做元数据过滤。
6. **检索置信度不适合单独当拒答闸门**：15 条拒答题的 Top-1 分数（0.527~0.748）与可答题大量重叠，
   精排分数做到全对要付出 22% 误拒率的代价（见第十三.7）。拒答判定要等生成阶段结合引用证据一起做。
7. **benchmark 的答案粒度是「关键串」不是「完整答案」**：自动评分只看关键实体是否出现在上下文里，
   不等价于生成答案正确。真正的端到端质量（尤其是幻觉率）要等生成模块接上后再补。
8. **测试集与调优样本没有彻底隔离**：切片参数（第十三节的 300/60）是先用 10 条探针调出来的，
   这 10 条的主题与 benchmark 有少量重合（心衰、基层糖尿病）。数字整体可信，
   但严格来说应补一组留出验证：只用 benchmark 的子集调参，在另一个从未参与调参的子集上确认。
9. **生成模块还没有接入**：目前只有检索结果，没有答案合成、没有引用标注、没有幻觉率统计。
10. **Windows 终端编码**：`scripts/` 下脚本已把 stdout 重设为 UTF-8，若仍出现乱码，执行前加 `$env:PYTHONIOENCODING = "utf-8"`。
11. **重排模型是 2.2GB 的外部资源**：`bge-reranker-base` 走 ModelScope 下载到本地缓存，脚本会从 `rerank.local_model_dir` 离线加载；没下载时 `--no-rerank` 依然可用，相关测试会自动跳过。

---

## 十五、下一步（W2 后半 ~ W3）

1. **按句边界重做切片**：直接针对第十三.6 的根因——长推荐句跨切片导致答案落在相邻切片。做法是把行按句末标点打包后再滑窗，让切片自然收尾在句末。预期能补回剩余 8 条硬骨头中的一部分。
2. **接 DeepSeek 生成**：带引用标注的答案合成，每条答案必须能指出来源切片。
3. **幻觉率与拒答率评测**：在 benchmark v1 上统计「答案是否有原文支撑」「应拒答题是否真的拒答」，补齐第十三.7 里单靠检索分数解决不了的那部分。
