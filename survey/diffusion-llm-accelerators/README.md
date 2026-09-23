# Diffusion LLM 加速器論文調查（軟體 + 硬體）

> 整理日期：2026-09。範圍：以 **diffusion / masked-diffusion 語言模型 (dLLM)** 為對象的推論加速研究，涵蓋硬體加速器、serving 系統、KV cache 與稀疏化、平行 / 投機解碼、量化，以及作為評估對象的基礎模型。每篇論文各有一個 md（見下方目錄），內容固定為：一句話總結、要解決的問題、核心方法與特色、使用的模型、PPA / 效能數據、限制、與其他論文的關係。

## 0. 背景：dLLM 推論為什麼慢、加速器要解什麼

dLLM（LLaDA、Dream、Mercury、Gemini Diffusion、LLaDA 2.0…）用「整段序列從全 [MASK] 開始、反覆去噪」的方式生成文字，理論上一次 forward 可以解出多個 token，但實務上有四個瓶頸：

| 瓶頸 | 原因 | 對應的加速方向 |
|---|---|---|
| **沒有 KV cache** | 注意力是雙向的，任何 token 改變都會讓所有 K/V 失效，所以每一步都要重算整段 | 近似 / 延遲 / 區塊式 KV cache、特徵快取（caching/） |
| **步數太多** | 預設「一步解一個 token」才能保品質，步數 ≈ 生成長度，等於退化成 AR 但每步更貴 | 信心閾值平行解碼、早停 (early commit)、投機驗證、蒸餾少步模型（decoding/） |
| **每步算整段** | prefix + 全部 response（含仍是 [MASK] 的後綴）都參與 attention / FFN | 後綴剪枝、稀疏注意力、token 選擇、MoE 專家重用（caching/） |
| **sampling 與記憶體流量** | 每步要對全詞彙 logits 做 softmax / top-k / 信心排序，再回寫遮罩狀態；activation footprint 遠大於 AR | 專用 NPU / 向量單元、混合精度、LPDDR 頻寬設計、KV/權重量化（hardware/, quantization/） |
| **serving 批次困難** | 不同請求收斂步數差異大、記憶體隨 block 長度線性成長 | block-grained 批次、reuse/refresh 交錯排程、CPU-GPU 協同 cache（systems/） |

三個常用的評估模型：**LLaDA-8B**（ML-GSAI，全注意力 masked diffusion）、**Dream-7B**（HKU，由 Qwen2.5 初始化）、**LLaDA-MoE / LLaDA 2.0**（Ant Group，block diffusion + MoE）。近期硬體與系統論文也開始用 block-diffusion 架構（BD3-LM、SDAR、Fast-dLLM v2）作為目標，因為 block 內雙向、block 間因果，可以直接沿用 AR 的 KV cache 與 serving 堆疊。

## 1. 閱讀建議

- 想做 **硬體加速器**：先讀 `hardware/npu-dllm-sampling.md`（第一個 dLLM 專用 NPU，7nm RTL）→ `hardware/masq.md`（mask-aware 混合精度 PE）→ `hardware/block-diffusion-edge-hw.md`（LPDDR + KV 壓縮 + FFN 替換的 edge 全端）→ `hardware/serving-mdlm-characterization.md`（真實 GPU profiling 找瓶頸）。
- 想做 **軟體推論框架**：`caching/fast-dllm.md` 是所有後續工作的共同基線；`systems/dinfer.md` 與 `systems/blockserve.md` 是目前吞吐最高的框架 / serving 設計。
- 想知道 **模型端可以配合什麼**：`models/fast-dllm-v2.md`、`models/llada2.md`、`models/sdar.md` 都在講「AR → block diffusion 轉換」，這是讓硬體與 serving 可以重用 AR 基礎設施的關鍵趨勢。

## 2. 論文目錄（每篇一行；點檔名看詳細筆記）

每個檔案固定包含：一句話總結 / 要解決的問題 / 核心方法與特色 / 使用的模型 / PPA 效能數據 / 限制 / 與其他論文的關係。「亮點數字」欄是該篇最具代表性的一個數字，條件細節請看檔內表格。

### 2.1 硬體加速器與硬體特性分析（`hardware/`，7 篇）

| 檔案 | 論文 | arXiv | 一行摘要 | 亮點數字 |
|---|---|---|---|---|
| [npu-dllm-sampling.md](hardware/npu-dllm-sampling.md) | Beyond GEMM-Centric NPUs: Enabling Efficient Diffusion LLM Sampling（舊名 NPU Design for Diffusion Language Model Inference；v2 代號 DART） | 2601.20706 | 第一顆 dLLM 專用 NPU：發現 sampling（logits→top-k→mask 更新）佔 GPU 端 70% 延遲，加 Vector-Scalar Sampling Engine、dLLM ISA、BAOS 4-bit KV 量化，7nm RTL | v1 對 A6000 2.53×；v2 對 A6000 4.91× TPS / 23.3× tok/J，對 H100 2.06× |
| [block-diffusion-edge-hw.md](hardware/block-diffusion-edge-hw.md) | Hardware Acceleration of Block-Diffusion LLM for Edge Devices | 2609.01084 | 邊緣 batch-one block-diffusion 全端：WIFiV-LPDDR 寬 I/O 精度標記讀取 + BRQ-KV 低秩+INT8 殘差 KV + DAT-FFN 權重替換，映射到混合精度 systolic array | 1.5B/7B 能耗降 3.79×/3.96×、延遲 2.88×/4.44×（模擬 Jetson-class，掉分 <1pp） |
| [mobile-npu-llada-cpp.md](hardware/mobile-npu-llada-cpp.md) | Efficient On-Device Diffusion LLM Inference with Mobile NPU（llada.cpp） | 2606.13740 | 首個手機 NPU-aware dLLM 框架：多 block 投機解碼 + 雙路徑漸進修訂 + swap 最佳化 runtime | LLaDA-8B 延遲較 CPU 降 17×–42×（Snapdragon 8 Elite Hexagon NPU） |
| [masq.md](hardware/masq.md) | MASQ: Accelerating Masked Diffusion via Stage-Wise Multi-Precision Quantization | 2605.23226 | **影像 masked diffusion，非 dLLM**。mask-aware MXINT8/4/2 多精度 MP-MPU（32 BMPE）、timestep-aware 排程；設計思路可直接移植到 dLLM 的 mask 稀疏 | 對 A100 16.06× / Orin NX 5.39×；能效 4.18×/4.93×；6.15 mm² @ 800 MHz |
| [serving-mdlm-characterization.md](hardware/serving-mdlm-characterization.md) | Serving Masked Diffusion LLMs: Characterization and Design Principles from Real Hardware | 2608.23807 | H200 實測 LLaDA-8B serving：瓶頸在 CPU 控制流而非 GPU kernel，步數只有 11 個離散層級，應同步批次共用 forward | 單請求僅 24.4% 時間是 GPU kernel；同步批次 batch 16 得 16.0× 吞吐 |
| [dipe-edge.md](hardware/dipe-edge.md) | DiPe: Real-Time Diffusion LLM Inference on Edge Devices for Planning Tasks（SenSys 2026 demo） | 無 | Jetson Orin NX 上量化 + 跨步資訊聚合的即時 dLLM 規劃示範 | 公開摘要無具體數字 |
| [activation-concentration.md](hardware/activation-concentration.md) | Activation Concentration: Characterizing Column-Level Output Sparsity Across Diffusion Model Architectures | 2606.00567 | **影像/音訊/影片 diffusion，不含 dLLM**。column-level 稀疏才是硬體真正可用的稀疏；附對 dLLM 加速器的啟示 | element-level 稀疏高估可用稀疏最多 78 pp；cycle 減少最多 30.6%（UNet） |

> 額外搜尋（FPGA / ASIC / PIM / CIM + diffusion LLM）未找到其他專門針對 dLLM 的硬體加速器論文；DiffAxE、Diff-Acc、SD-Acc 等是影像 diffusion，HPIM / Pimba 等 PIM 是 AR LLM。截至 2026-09，dLLM 專用硬體仍是很新的題目。

### 2.2 推論框架與 serving 系統（`systems/`，9 篇）

| 檔案 | 論文 | arXiv | 一行摘要 | 亮點數字 |
|---|---|---|---|---|
| [dinfer.md](systems/dinfer.md) | dInfer: An Efficient Inference Framework for Diffusion Language Models（Ant Group） | 2510.08666 | model / iteration manager / decoding strategy / KV-cache manager 四模組解耦，加 TP/EP、CUDA Graph、loop unrolling | LLaDA-MoE HumanEval >1,100 tok/s（8×H800, bs=1），約 10× Fast-dLLM |
| [dllm-serve.md](systems/dllm-serve.md) | Taming the Memory Footprint Crisis: System Design for Production Diffusion LLM Serving（dLLM-Serve） | 2512.17077 | logit 記憶體預算、Refresh/Reuse phase 交錯排程、head-centric 稀疏 KV，解決 dLLM serving 記憶體爆炸 | 吞吐 1.61–1.81×（RTX 4090），尾延遲降近 4× |
| [sangam.md](systems/sangam.md) | Sangam: Efficiently Serving Diffusion LLMs with the AR Stack | 2607.04206 | 重用 AR serving 機制（FlashInfer、CUDA Graph）+ deficit token-budget 排程處理不可分割的 dLLM prefill，支援 P/D 分離 | 同延遲下承受約 2.5–3× Fast-dLLM 的負載（LLaDA-8B / Dream-7B, H100） |
| [herald.md](systems/herald.md) | HERALD: High-Throughput Block Diffusion LLM Serving via CPU-GPU Cooperative KV Cache Retrieval | 2606.21633 | 基於 SGLang，block 內 KV 只挑選一次、CPU 選 GPU 算、跨 PCIe 一次，長上下文 KV offloading | 5–10% KV 預算近乎無損；32K 上下文 decode 吞吐最高 2.47× |
| [blockserve.md](systems/blockserve.md) | BlockServe: Block-Grained Continuous Batching for High-Throughput Diffusion LLM Serving | 2607.08930 | 以 block 為排程量子、block 邊界驅逐 + gather-scatter 混合狀態執行，解決請求收斂步數異質性 | 1.9–10.6× 吞吐 vs Fast-dLLM；batch 容量 2–4× |
| [dual-boundaries.md](systems/dual-boundaries.md) | Orchestrating Dual-Boundaries: An Arithmetic Intensity Inspired Acceleration Framework for Diffusion Language Models（ODB-dLLM, DAC 2026） | 2511.21759 | Roofline 雙邊界分析：EOS 自適應長度預測砍掉 compute-bound 的 prefill，jump-share 投機解碼填滿 memory-bound 的迭代 | 46–182× vs vanilla LLaDA；2.6–7.2× vs Fast-dLLM（A100） |
| [flash-dllm.md](systems/flash-dllm.md) | Flash-dLLM: IO-Aware KV Caching and Parallel Decoding for Fast, Memory-Efficient Diffusion LLMs | 2609.26796 | Triton 融合 IO-aware KV kernel + 自我 draft-and-verify，把瓶頸從 FLOPs 移到記憶體 I/O | 11.0×（HumanEval）、5.1×（GSM8K）vs Elastic-Cache |
| [sglang-dllm.md](systems/sglang-dllm.md) | SGLang「Power Up Diffusion LLMs: Day-0 Support for LLaDA 2.0」（部落格 + RFC） | 非論文 | 重用 chunked-prefill 管線 + diffusion algorithm 抽象層，LLaDA 2.0 直接擁有 batching / TP / EP / CUDA Graph | LLaDA2.0-flash-CAP 約 500 TPS，最多 1.9× AR |
| [other-frameworks.md](systems/other-frameworks.md) | vLLM / llama.cpp / Ollama / TensorRT-LLM 對 dLLM 的支援彙整 | 非論文 | vLLM 以 DiffusionGemma 走 spec-decode 路徑；llama.cpp 有 CLI 但無 KV cache；Ollama 仍為 draft；TRT-LLM 只有視覺 diffusion | vLLM DiffusionGemma 1,288 tok/s（H200 FP8 bs=1，約 6× AR） |

### 2.3 KV cache、特徵快取、稀疏注意力與 token 剪枝（`caching/`，15 篇）

| 檔案 | 論文 | arXiv | 一行摘要 | 亮點數字 |
|---|---|---|---|---|
| [fast-dllm.md](caching/fast-dllm.md) | Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding（NVIDIA, ICLR 2026） | 2505.22618 | block-wise 近似 KV cache（DualCache 連 suffix 也快取）+ 信心閾值平行解碼；幾乎所有後續工作的 baseline | 27.6× 吞吐（LLaDA GSM8K 8-shot, 1024 tokens, A100；準確度 76.0 vs 77.3） |
| [dkv-cache.md](caching/dkv-cache.md) | dKV-Cache: The Cache for Diffusion Language Models | 2505.15781 | 已解碼 token 延遲一步再快取 KV（delayed caching），Decode / Greedy / PD 三種變體 | 2–10× 加速；GSM8K 6.6× 且 Pass@1 上升 |
| [dllm-cache.md](caching/dllm-cache.md) | dLLM-Cache: Accelerating Diffusion LLMs with Adaptive Caching | 2506.06295 | prompt 特徵長間隔快取、response 特徵短間隔 + V-verify 只更新變化大的 token | 9.1×（LLaDA-8B, LongBench-HotpotQA） |
| [d2cache.md](caching/d2cache.md) | d²Cache: Accelerating Diffusion-Based LLMs via Dual Adaptive Caching | 2509.23094 | 兩階段（確定性先驗 + 注意力分數）挑出要更新 KV 的 token | 平均 3.5×；GSM8K 2.62→12.25 tok/s |
| [sparse-dllm.md](caching/sparse-dllm.md) | Sparse-dLLM: Accelerating Diffusion LLMs with Dynamic Cache Eviction | 2508.02558 | 延遲雙向稀疏快取 + 注意力導向的 prefix / suffix KV 淘汰 | 最高 10× 吞吐、記憶體與原模型相當 |
| [sparsed.md](caching/sparsed.md) | SparseD: Sparse Attention for Diffusion Language Models | 2509.24014 | 早期步驟用全注意力算出 head-specific 稀疏 pattern，之後各步重用 | 1.50× vs FlashAttention（64K 上下文，無損） |
| [elastic-cache.md](caching/elastic-cache.md) | Attention Is All You Need for KV Cache in Diffusion LLMs（Elastic-Cache, ICLR 2026） | 2510.14973 | 用「最被注意 token 的漂移」決定何時刷新、深層優先決定從哪層刷新 | 45.1×（LLaDA-1.5 GSM8K 512；README 修正後最高 16×） |
| [dpad.md](caching/dpad.md) | DPad: Efficient Diffusion Language Models with Suffix Dropout | 2508.14148 | 滑動視窗 + 距離衰減，丟棄遠端 suffix mask token 不參與計算 | 61.39×（LLaDA-1.5 GSM8K 1024 tokens，疊加 Fast-dLLM） |
| [focus-dllm.md](caching/focus-dllm.md) | Focus-dLLM: Accelerating Long-Context Diffusion LLM Inference via Confidence-Guided Context Focusing | 2602.02159 | 前一步信心預測下一步會 unmask 的區域，加 sink-aware 剪枝 prompt 注意力 | 29.6× 無損（32K 上下文） |
| [dyllm.md](caching/dyllm.md) | DyLLM: Efficient Diffusion LLM Inference via Saliency-based Token Selection and Partial Attention | 2603.08026 | 相鄰步 attention context 相似度挑 salient token，只重算它們的 attention + FFN | 7.6× / 9.6× 吞吐（LLaDA / Dream） |
| [r2-dllm.md](caching/r2-dllm.md) | R²-dLLM: Accelerating Diffusion LLMs via Spatio-Temporal Redundancy Reduction | 2604.18995 | 信心群聚合 + 定稿穩定預測 + 冗餘感知 SFT | 解碼步數最多 −88%、最高 10.1×（H200） |
| [es-dllm.md](caching/es-dllm.md) | ES-dLLM: Efficient Inference for Diffusion LLMs by Early-Skipping | 2603.10088 | 依張量變化與前輪信心，在淺層就跳過不重要 token 的計算 | 308.51 TPS（Dream-7B, H200）；5.6–16.8× |
| [prefilling-dllm.md](caching/prefilling-dllm.md) | Prefilling-dLLM: Predictive Prefilling for Long-Context Inference in Diffusion LMs | 2606.10537 | prefix 分 chunk 一次 prefill、只取 top-K chunk + 非連續 KV kernel | 28.0×（32K）；8K 9.1×、16K 16.1× |
| [team-moe.md](caching/team-moe.md) | TEAM: Temporal-Spatial Consistency Guided Expert Activation for MoE dLLM Acceleration | 2602.08404 | 利用專家路由的時空一致性少啟用專家，熱門 token 才投機探索 | 最高 2.2×（SDAR-30B-A3B HumanEval；專家 −35–39%） |
| [streaming-dllm.md](caching/streaming-dllm.md) | Streaming-dLLM: Accelerating Diffusion LLMs via Suffix Pruning and Dynamic Decoding | 2601.17917 | 衰減導向 suffix 剪枝 + 動態信心解碼 + block early exit | 68.2×（LLaDA-1.5 MBPP 512） |

### 2.4 平行解碼、投機解碼、早停與解碼策略（`decoding/`，20 篇）

| 檔案 | 論文 | arXiv | 一行摘要 | 亮點數字 |
|---|---|---|---|---|
| [prophet.md](decoding/prophet.md) | Diffusion Language Models Know the Answer Before Decoding（Prophet） | 2508.19982 | top-2 logit gap 判斷答案已定，一次 all-in 提早結束解碼 | 步數最多 −3.4×；GSM8K 97% 樣本半程即正確 |
| [apd.md](decoding/apd.md) | Accelerating Diffusion LLMs via Adaptive Parallel Decoding（APD, NeurIPS 2025） | 2506.00413 | 小 AR 模型驗證 dLLM 的平行 draft，加 KV cache 與限制 mask 長度 | Dream-7B 59 tok/s vs Qwen2.5-7B AR 37 tok/s（A5000） |
| [wino.md](decoding/wino.md) | Wide-In, Narrow-Out: Revokable Decoding（WINO, ICLR 2026） | 2507.18578 | 大膽 draft 再用雙向上下文撤銷可疑 token | GSM8K +2.58% 且 6.10× TPS；最高 10× |
| [learn2pd.md](decoding/learn2pd.md) | Learning to Parallel（Learn2PD, ICLR 2026） | 2509.25188 | 輕量 filter 模擬 oracle 決定 unmask，加 EoT 預測截斷 padding | GSM8K 22.58×，配 KV cache 57.51× |
| [dparallel.md](decoding/dparallel.md) | dParallel: Learnable Parallel Decoding for dLLMs（ICLR 2026） | 2509.26488 | certainty-forcing distillation 讓 mask token 信心同時快速收斂 | LLaDA-8B GSM8K 256→30 步，8.5× 不掉分 |
| [spiffy.md](decoding/spiffy.md) | Spiffy: Multiplying Diffusion LLM Acceleration via Lossless Speculative Decoding（後改題 Structuring The Future） | 2509.18085 | 自我投機 + 離線校準的有向 draft graph，分佈無損 | 單獨 2.8–3.1×，疊 KV cache 最高 7.9× |
| [freedave.md](decoding/freedave.md) | Free Draft-and-Verification（FreeDave） | 2510.00294 | 上一步的平行預測當 draft，下一步 forward 順便驗證，零額外 forward | 最高 2.83× lossless |
| [cdlm.md](decoding/cdlm.md) | CDLM: Consistency Diffusion Language Models For Faster Sampling | 2511.19269 | 從雙向 teacher 蒸餾 block-causal student，多 token 定案且可用標準 KV cache | 延遲降 3.6–14.5× |
| [dmax.md](decoding/dmax.md) | DMax: Aggressive Parallel Decoding for dLLMs | 2604.08302 | On-Policy Uniform Training 讓模型自我修正 + soft parallel decoding | LLaDA-2.0-mini 每 forward 2.8→6.2 token；2×H200 1,338 TPS |
| [psd.md](decoding/psd.md) | PSD: Pushing the Pareto Frontier via Parallel Speculative Decoding | 2605.15609 | 空間多 token unmask × 時間多深度投機驗證，同一 forward 相乘 | 最高 5.5× tokens/forward |
| [blockbatch.md](decoding/blockbatch.md) | BlockBatch: Multi-Scale Consensus Decoding | 2605.29233 | 同時跑多個 block size 分支，信心閘控合併 | 比 Fast-dLLM 少 26.6% 步數 |
| [starr.md](decoding/starr.md) | STaRR: Spatial-Temporal Token-Dynamics-Aware Responsive Remasking | 2601.04205 | 跨步信心變異數與空間偏離做動態 remask 門檻 | 平均 4.1×、最高 8.9× |
| [tacg.md](decoding/tacg.md) | TACG: Trajectory-Aware Commit Gating | 2607.03236 | EMA logits 對比 + history gate 決定「可否 commit」 | 零額外 forward；倍率未查到 |
| [dawn.md](decoding/dawn.md) | DAWN: Dependency-Aware Fast Inference for Diffusion LLMs | 2602.06953 | 由 attention 建依賴圖，避免同步解耦合的位置 | 1.80–8.06×（LLaDA-1.5 MBPP 8.06×） |
| [swordsman.md](decoding/swordsman.md) | Swordsman: Entropy-Driven Adaptive Block Partition | 2602.04399 | 依相鄰 token entropy 跳變切 block，對齊語意邊界 | GSM8K 81.50%（+6.29 vs Fast-dLLM）且 8.79× |
| [simsd.md](decoding/simsd.md) | SimSD: Simple Speculative Decoding in Diffusion Language Models | 2606.02544 | masking 策略給 dLLM 時間一致上下文，小 dLLM draft 大 dLLM 驗證 | SDAR-1.7B→8B 63–74 tok/s vs 9.6 tok/s |
| [dartree.md](decoding/dartree.md) | DARTree: Speculative Diffusion Decoding with Autoregressive Draft Trees | 2608.13524 | **加速 AR 模型**：diffusion drafter + AR correction 擴成樹再樹狀驗證（作對照） | 接受長度 12.97 tok/輪，最高 9.73× 無損 |
| [longest-stable-prefix.md](decoding/longest-stable-prefix.md) | Beyond Scattered Acceptance: Longest Stable Prefixes（LSP） | 2603.05454 | 每步只 commit 最長左對齊穩定前綴，KV cache 可連續 append | 最高 3.4×（LLaDA-8B / Dream-7B） |
| [block-verification.md](decoding/block-verification.md) | Accelerating Speculative Diffusions via Block Verification | 2606.13426 | **連續 diffusion** 的精確 speculative sampling + block verification（理論對照） | 比既有 speculative diffusion 再快最多 6.3% |
| [jacobi-forcing.md](decoding/jacobi-forcing.md) | Fast and Accurate Causal Parallel Decoding using Jacobi Forcing | 2512.14681 | **AR 方法**：沿 Jacobi 軌跡做 consistency 蒸餾，保持 causal 的平行解碼（與 dLLM 對照） | HumanEval 4.0×（163.9 TPS, 83.5%） |

### 2.5 量化與壓縮（`quantization/`，8 篇）

| 檔案 | 論文 | arXiv | 一行摘要 | 亮點數字 |
|---|---|---|---|---|
| [quantization-meets-dllms.md](quantization/quantization-meets-dllms.md) | Quantization Meets dLLMs: A Systematic Study of PTQ for Diffusion LLMs | 2508.14896 | 首個 dLLM PTQ benchmark：發現 massive outlier，GPTQ > AWQ、rotation 類 ≫ SmoothQuant | W4A4 下 SmoothQuant 在 code/math 幾近崩潰，QuaRot / DuQuant 全面勝出 |
| [dllmquant.md](quantization/dllmquant.md) | DLLMQuant: Quantizing Diffusion-based Large Language Models | 2508.14090 | TMAS 校準抽樣 + IA-AQ 激活量化 + CGQ 誤差補償 | LLaDA 4-bit GSM8K 回升 >10 分（AWQ W4A4 原本掉 16%） |
| [quant-dllm.md](quantization/quant-dllm.md) | Quant-dLLM: Post-Training Extreme Low-Bit Quantization for Diffusion LLMs | 2510.03274 | MCS 遮罩校準模擬 + DAQ 任意順序量化器 + ABMP 混合精度，達 2-bit weight-only | LLaDA-8B 16.09 GB → 3.69 GB；7 任務平均 51.3% vs GPTQ 36.5% |
| [star-quant.md](quantization/star-quant.md) | STaR-Quant: State-Time Consistent PTQ for Diffusion LLMs | 2606.04945 | SGAT 分離 masked / unmasked 激活空間 + TAC 補償跨步 attention 誤差 | LLaDA-8B W4A4 九任務平均 57.07 |
| [sink-aware-pruning.md](quantization/sink-aware-pruning.md) | Sink-Aware Pruning for Diffusion Language Models | 2602.17664 | dLLM attention sink 是暫態的，剪掉不穩定 sink 再套 Wanda / SparseGPT | LLaDA-8B 50% 稀疏平均 53.18 vs Wanda 52.70（dense 57.93） |
| [fair-calib.md](quantization/fair-calib.md) | FAIR-Calib: Frontier-Aware Instability-Reweighted Calibration for PTQ of Diffusion LLMs（ICML 2026） | 2606.06547 | 以 write-frontier 脆弱度先驗重加權 layer-wise 校準損失 | 校準序列僅 1024 即優於 W4A4 基線 |
| [cd4lm.md](quantization/cd4lm.md) | CD⁴LM: Consistency Distillation and aDaptive Decoding for Diffusion LMs | 2601.02236 | 離散空間 consistency 蒸餾 + 置信度自適應跳步解碼 | GSM8K 5.18× 且 77.4→77.6%；平均 3.62× |
| [didi-instruct.md](quantization/didi-instruct.md) | Ultra-Fast Language Generation via Discrete Diffusion Divergence Instruct（DiDi-Instruct） | 2509.25035 | 積分 KL 蒸餾把 masked dLLM 壓成 few-step 學生 | 最高 64× 加速（169M, OWT） |

### 2.6 基礎模型與架構（`models/`，15 篇）

這一類的「PPA」是參數量、訓練 token 數、推論 tok/s 與 benchmark 分數，以及模型本身內建的加速設計。

| 檔案 | 論文 | arXiv | 一行摘要 | 亮點數字 |
|---|---|---|---|---|
| [llada.md](models/llada.md) | Large Language Diffusion Models（LLaDA；含 LLaDA 1.5 備註） | 2502.09992 | 從零訓練 8B masked diffusion LM，追平 LLaMA3 8B；無 KV cache、慢於 AR，是所有加速研究的主要靶子 | 2.3T tokens / 0.13M H800 GPU-hours；MMLU 65.9 vs LLaMA3 65.4 |
| [llada-moe.md](models/llada-moe.md) | LLaDA-MoE: A Sparse MoE Diffusion Language Model | 2509.24389 | 首個從零訓練的 MoE dLLM，7B 總 / 1.4B 啟用 | Instruct 平均 53.12 vs Qwen2.5-3B-Instruct 53.51；dInfer 1,100+ tok/s |
| [llada2.md](models/llada2.md) | LLaDA2.0: Scaling Up Diffusion Language Models to 100B | 2512.15745 | AR MoE 經 block-size WSD 課程轉成 block diffusion，16B-A1B 與 100B-A6B，CAP 平行解碼 | 535 tok/s（約 2.1× 同級 AR，SGLang TP8 H20） |
| [dream.md](models/dream.md) | Dream 7B: Diffusion Large Language Models（含 Dream-Coder） | 2508.15487 | Qwen2.5-7B 初始化 + CART 噪聲排程，580B tokens 即超越 LLaDA，規劃任務大勝 AR | Sudoku 81.0 vs Qwen2.5-7B 21.0；訓練資料為 LLaDA 的 1/4 |
| [mercury.md](models/mercury.md) | Mercury: Ultra-Fast Language Models Based on Diffusion（Inception Labs） | 2506.17298 | 首個商用 dLLM，Coder Mini / Small 在 H100 上 5–10× 快於速度型 AR | 1,109 tok/s（Mini, H100, 第三方量測） |
| [gemini-diffusion.md](models/gemini-diffusion.md) | Gemini Diffusion（Google DeepMind，無論文；後續 DiffusionGemma） | 無 | 自報 1,479 tok/s、約 5× Flash-Lite，程式持平但推理落後 | 1,479 tok/s；HumanEval 89.6 vs 90.2 |
| [seed-diffusion.md](models/seed-diffusion.md) | Seed Diffusion: A Large-Scale Diffusion Language Model with High-Speed Inference（ByteDance） | 2508.02193 | 兩階段（遮罩→編輯）課程 + 限制順序蒸餾 + on-policy 步數優化 | 2,146 tok/s on H20（同規模 AR 的 5.4×） |
| [block-diffusion-bd3lm.md](models/block-diffusion-bd3lm.md) | Block Diffusion: Interpolating Between AR and Diffusion LMs（BD3-LM, ICLR 2025 Oral） | 2503.09573 | block 間 AR、block 內擴散，取回 KV cache 與變長生成；後續所有 block diffusion 模型的起點 | LM1B PPL 28.23（比 MDLM 好 13%） |
| [sdar.md](models/sdar.md) | SDAR: A Synergistic Diffusion–AutoRegression Paradigm（Shanghai AI Lab） | 2510.06303 | 50B tokens 把 Qwen3（1.7B–30B-A3B）轉為 block diffusion，通用持平、科學推理反超 | 僅原預訓練 0.14% 的資料；SDAR-4B 3,700+ tok/s（H200） |
| [fast-dllm-v2.md](models/fast-dllm-v2.md) | Fast-dLLM v2: Efficient Block-Diffusion LLM（NVIDIA） | 2509.26328 | ~1B tokens 微調 Qwen2.5 成 block 32 / sub-block 8 模型，互補遮罩 + 階層式 cache | 2.54× 吞吐 vs Qwen2.5-7B-Instruct；資料比 Dream 少 500× |
| [efficient-dlm.md](models/efficient-dlm.md) | Efficient-DLM: From AR to Diffusion LMs, and Beyond in Speed | 2512.14067 | 證明 block-wise attention 最能保留 AR 權重分布，加位置相依遮罩縮小 train-test gap | 8B：+5.4% 準確率且 4.5× 吞吐 vs Dream 7B |
| [nemotron-labs-diffusion.md](models/nemotron-labs-diffusion.md) | Nemotron-Labs-Diffusion: A Tri-Mode LM Unifying AR, Diffusion, and Self-Speculation（NVIDIA） | 2607.05722 | 一組權重三種解碼模式；擴散起草 + AR 驗證的無損加速，3B / 8B / 14B | 8B 每 forward 6× tokens vs Qwen3-8B；GB200 上約 4× |
| [dllm-simple.md](models/dllm-simple.md) | dLLM: Simple Diffusion Language Modeling | 2602.22661 | 統一 dLLM 訓練 / 推論 / 評測的開源框架，內建 MDLM / BD3LM 與 Fast-dLLM cache | 框架論文，無 PPA 數字 |
| [set-diffusion.md](models/set-diffusion.md) | Set Diffusion: Interpolating Token Orderings Between AR and Diffusion（ICML 2026） | 2607.01775 | 把 block 推廣為任意位置 / 大小的 token 集合，set-causal 架構每步都能更新 KV cache | 倍率未查到 |
| [survey-dlm.md](models/survey-dlm.md) | A Survey on Diffusion Language Models（MBZUAI） | 2508.10875 | 綜述；推論效率分為 unmasking / remasking / caching / guidance / sampling / quantization | 工業 dLLM 「最高 ~10×」、學術方法「2–4×」 |

## 3. 橫向比較：哪類方法給多少加速、代價是什麼

| 方法類別 | 代表 | 典型加速（對 vanilla LLaDA/Dream） | 是否需訓練 | 主要代價 |
|---|---|---|---|---|
| 近似 / 延遲 KV cache | Fast-dLLM、dKV-Cache、Elastic-Cache | 3–10×（單獨）、疊平行解碼可到 20–60× | 否 | block 邊界仍需全量重算；近似誤差 |
| 特徵快取 + 部分更新 | dLLM-Cache、d²Cache、DyLLM | 3–9× | 否 | 需挑 token 的額外指標計算；記憶體多存一份特徵 |
| suffix 剪枝 / 稀疏注意力 | DPad、Streaming-dLLM、SparseD、Focus-dLLM | 長序列 10–60× | 否 | 開放式生成或長 suffix 相依任務可能掉分 |
| 信心閾值平行解碼 | Fast-dLLM、WINO、DAWN、Swordsman | 2–10× | 否 | 閾值調參；低信心任務退化為逐 token |
| 可學習平行 / 蒸餾 | dParallel、Learn2PD、CDLM、DMax、R²-dLLM | 步數 −80~90%，8–20× | 是（小規模 SFT / 蒸餾） | 需要訓練資料與 GPU 時數；泛化到新任務要重訓 |
| 投機解碼（無損） | Spiffy、FreeDave、PSD、SimSD | 2–8× | 否（部分需離線校準） | draft graph / draft 模型的額外記憶體 |
| 早停 | Prophet、TACG、LSP | 2–3.4× | 否 | 需要 gap 閾值；對難題可能過早 commit |
| 量化 | Quant-dLLM、STaR-Quant、DLLMQuant | 記憶體 −75%（W2/W4） | 否（PTQ） | dLLM 的 masked/unmasked 激活分布不同，通用 PTQ 直接套會崩 |
| serving 系統 | dInfer、BlockServe、Sangam、dLLM-Serve | 對 Fast-dLLM 再 2–10× 吞吐 | 否 | 工程量大；block-diffusion 模型才能完整重用 AR 堆疊 |
| 專用硬體 | DART NPU、Block-Diffusion Edge HW、MASQ | 對 GPU 2–5× TPS、15–23× tok/J | 否 | 只有 RTL / 模擬結果，尚無流片；只評估 LLaDA 系列 |

三個值得注意的趨勢：

1. **瓶頸不在 GEMM**：硬體側（DART 的 sampling 佔 70%）與系統側（H200 characterization 的 CPU 端佔 75.6%）都指向非矩陣運算與控制流，這和 AR LLM 加速器「餵飽 systolic array」的思路不同。
2. **模型正在向 block diffusion 收斂**：LLaDA 2.0、SDAR、Fast-dLLM v2、Efficient-DLM、Nemotron-Labs-Diffusion 全都是「AR 權重 → block diffusion」。這讓 KV cache 變成精確而非近似、serving 可以直接用 SGLang / vLLM，硬體也能重用 AR 的 append-only cache 假設，只需在 block 內處理雙向注意力與 mask 更新。
3. **數字的可比性差**：各篇 baseline（vanilla LLaDA vs Fast-dLLM vs Elastic-Cache）、生成長度（256 vs 1024）、GPU（A100 / H100 / H200）都不同，看到 60× 這種數字要先看條件欄。

## 4. 資料來源與可信度說明

- 整理時 arXiv、Hugging Face、alphaXiv、Semantic Scholar 等站點都被本環境的網路 proxy 擋住，所有內容來自 WebSearch 摘要與 GitHub README / issue，並以撰寫者既有知識補充方法描述。
- 每篇檔案裡查不到的數字一律標「論文未提供 / 未查到」，僅憑記憶的標「(依記憶，待確認)」，不同來源衝突的會並列並註明。**引用前請對照原文。**
- 已知的版本差異：2601.20706 在 v1 / v2（DART）數字不同；Spiffy 後續改題；Learn2PD 正確 ID 為 2509.25188。
- 三篇非 dLLM 的對照論文（MASQ、Activation Concentration、DARTree / Block Verification / Jacobi Forcing）保留是因為它們的硬體或解碼思路可直接移植，檔內都有明確標註。

## 5. 檔案結構

```
survey/diffusion-llm-accelerators/
├── README.md            ← 本檔（目錄 + 橫向比較）
├── _TEMPLATE.md         ← 每篇筆記的固定格式，新增論文請照此格式
├── hardware/            ← 7 篇：NPU / 邊緣加速器 / 硬體特性分析
├── systems/             ← 9 篇：推論框架與 serving
├── caching/             ← 15 篇：KV cache、特徵快取、稀疏注意力、token 剪枝
├── decoding/            ← 20 篇：平行 / 投機 / 早停解碼
├── quantization/        ← 8 篇：PTQ、剪枝、少步蒸餾
└── models/              ← 15 篇：基礎模型、block diffusion 架構、綜述
```
