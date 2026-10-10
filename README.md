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
| 脚本 | `scripts/build_index.py`、`scripts/query.py` | 一键建索引、命令行查询 |

**未完成（后续阶段）**：生成式答案合成、RAGAS/自建评测集、幻觉率与拒答正确率统计、BM25 混合检索（依赖已装，逻辑未写）、bge-reranker 重排、Gradio/FastAPI 服务、效果对比实验。

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
| `paths.index_path` | `data/index/faiss_smoke.index` | FAISS 索引文件 |
| `paths.meta_path` | `data/index/chunks_smoke.json` | 切片元数据 |
| `chunking.chunk_size` | 400 | 切片最大字符数 |
| `chunking.chunk_overlap` | 80 | 相邻切片重叠字符数 |
| `chunking.heading_max_chars` | 60 | 超过此长度的行不算标题 |
| `chunking.prepend_heading_path` | true | 是否把标题链写进切片正文 |
| `embedding.model_name` | `BAAI/bge-small-zh-v1.5` | 向量模型 |
| `embedding.device` | cpu | 推理设备 |
| `embedding.batch_size` | 16 | 批大小 |
| `embedding.normalize` | true | L2 归一化，配合 IndexFlatIP 得到余弦相似度 |
| `embedding.query_instruction` | `为这个句子生成表示以用于检索相关文章：` | bge 官方查询前缀 |
| `embedding.local_model_dir` | 空 | 显式指定本地模型目录 |
| `retrieval.top_k` | 5 | 默认返回条数 |
| `runtime.num_threads` | 12 | CPU 推理线程数 |

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
│   └── retrieve.py
├── data/
│   ├── raw/               # 真实文档，自行放入
│   ├── smoke/             # 20 条样例条目（仅用于冒烟测试）
│   └── index/             # 构建产物，不进版本库
├── scripts/
│   ├── build_index.py
│   └── query.py
└── tests/
    ├── conftest.py
    ├── test_chunking.py
    ├── test_embedding.py
    ├── test_faiss_index.py
    └── test_smoke_retrieval.py
```

---

## 十、已知限制

1. **切片只用字符数，不做句边界对齐**：滑窗按字符切，长句可能被拦腰截断。下一步可以按句队列打包，让每个切片自然收尾在句末。
2. **跨页 Section 只记起始页码**：一个标题下的正文跨多页时，页码取第一页。后续可在 Section 内存 `page_range`。
3. **PDF 表格提取未处理**：pdfplumber 的表格数值目前混在正文里，排版较乱的表格会串行；没有 OCR，扫描件不可用。
4. **现有检索只有向量一路**：BM25（rank_bm25 + jieba）依赖已装，混合检索放在 W2。
5. **Windows 终端编码**：`scripts/` 下两个脚本已把 stdout 重设为 UTF-8，若仍出现乱码，执行前加 `$env:PYTHONIOENCODING = "utf-8"`。
