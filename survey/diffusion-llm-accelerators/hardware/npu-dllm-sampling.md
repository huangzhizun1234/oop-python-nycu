# Beyond GEMM-Centric NPUs: Enabling Efficient Diffusion LLM Sampling（舊題名：NPU Design for Diffusion Language Model Inference；架構代號 DART）

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2601.20706`（v1 2026-01；v2 2026-04 改題為 DART 全平台版本；最新版題名 "Beyond GEMM-Centric NPUs"） |
| 作者 / 單位 | Binglei Lou, Haoran Wu, Kevin Lau, Gregor MacDonald et al.（Imperial College London、University of Edinburgh、University of Cambridge） |
| 日期 | 2026-01（v1）/ 2026-04（v2） |
| 類別 | 硬體加速器 |
| 連結 | [arXiv](https://arxiv.org/abs/2601.20706) · GitHub：未查到公開 repo |

## 一句話總結

第一顆專為 dLLM 設計的 NPU：發現 dLLM 的「取樣 (sampling)」階段（logits → 信心分數 → top-k → 遮罩更新）是非 GEMM 的向量/歸約運算，在 GPU 上可佔 70~71% 端到端延遲；因此在傳統 Transformer Engine 之外加一個 Vector-Scalar Sampling Engine、dLLM 專用 ISA、原地記憶體重用與混合精度記憶體階層，7nm RTL 合成後對 RTX A6000 最高 2.53x speedup（v1）；v2 (DART, VLEN=2048) 進一步報告對 A6000 4.91x TPS / 23.3x tok/J、對 H100 2.06x TPS / 15.5x tok/J。

## 要解決的問題

- AR LLM 加速器的假設在 dLLM 上不成立：dLLM 用雙向注意力、每步都是「全序列 prefill-like pass」、block-wise KV cache 需要刷新（非 append-only）、且每一步之後都有一個 reduction-heavy、top-k 驅動的取樣階段。
- 取樣階段的工作負載：vocabulary 大小的 logits（LLaDA 約 126K 詞彙）要做 softmax/argmax/信心分數、streaming top-k、再對整段 token 序列做遮罩更新；每步產生三種資料型別（logit 向量、FP 信心純量、整數 token index），存取不規則、需要大 on-chip SRAM。GEMM-centric NPU（systolic array + 少量向量單元）處理這段非常沒效率。
- 論文在 GPU 上 profiling：取樣階段最高可佔 dLLM 端到端延遲的 70%（v1）/ 71%（v2）。
- AR 式 KV 量化（離線校準、靜態分佈）不適用於 dLLM：block-wise 迭代精煉會造成逐步 (step-wise) 的分佈飄移與 channel 級離群值變動。

## 核心方法與特色

- **dLLM 專用 ISA 與編譯器**：在 GEMM 指令之外加入輕量非 GEMM 向量原語：fused max-with-index、streaming top-k、整數 masked-token update 等，讓取樣階段可以用少量向量/純量指令流水化，而不是回落到主機或通用向量單元。編譯器把 transformer forward 與 diffusion sampling 都排程到同一套執行模型。
- **雙引擎架構 (v2, DART)**：Transformer Engine 負責 GEMM 密集的 forward pass；Vector-Scalar Sampling Engine 負責 softmax/信心/top-k/遮罩更新。取樣引擎的向量長度 VLEN 可配置（評估到 2048），代價是 SRAM 與面積隨 VLEN 成長。
- **原地記憶體重用 + 解耦混合精度記憶體階層**：logits、FP 信心純量、整數 token index 三種資料在 on-chip 記憶體中「物理隔離」成不同 domain，各自用適合的精度；取樣過程對 logits 做原地 (in-place) 覆寫，避免 vocab-wide 的大量寫回/重載。論文另證明取樣運算可從 FP64 降到 MXFP8 而不損品質，把取樣佔比從 ~70% 壓到 <10%。
- **Block-Adaptive Online Smoothing (BAOS) KV 量化**：利用 block-wise 解碼固有的「warm step」重算作為零額外成本的線上校準點；在 warm step 計算每 channel 的 scaling factor、把正規化後的 key 寫進 cache，精煉步驟的 attention 則把「反向 scaling」融合到 query 上而不是去解量化 cached key。使用 MX 格式，可穩定做 4-bit KV cache，且完全不需離線校準資料。
- **完整 RTL + 7nm 合成，三路交叉驗證**：記憶體子系統模型對 AMD HBM2e 實測校正、計算流水線對 Verilator RTL 模擬校正，再做端到端模擬；不是純分析模型。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| LLaDA-8B（Base / Instruct） | 8B | 主要工作負載；GSM8K 用於驗證 BAOS 4-bit KV 與 MXFP8 取樣的精度 |
| LLaDA 系列（其他規模） | 論文未在摘要列明 | v2 稱「across LLaDA-series models」做 FP64→MXFP8 精度掃描 |

## PPA / 效能數據

> 硬體論文：製程、面積、功耗、頻率、能效 (TOPS/W)、對比 GPU 的 speedup / energy。

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| 製程 | 7nm（RTL 合成） | — | 完整 RTL；Verilator 驗證 |
| 面積 / 功耗 / 頻率 | 論文未在公開摘要提供 / 未查到 | — | 需查全文表格 |
| 取樣階段延遲佔比 | 最高 70%（v1）/ 71%（v2） | GPU 上原生 dLLM 推論 | 代表性 dLLM 工作負載 |
| 取樣精度降階後的佔比 | <10% 端到端延遲 | 對比 FP64 取樣 | 取樣改用 MXFP8，品質不變 |
| 端到端 speedup（v1） | 最高 2.53x | 高階 NVIDIA GPU（RTX A6000） | 依使用者提供資訊為 A6000；v1 摘要只寫 "high-end NVIDIA GPU" |
| TPS speedup（v2, DART） | 最高 4.91x | NVIDIA A6000 | LLaDA-8B，VLEN=2048 |
| TPS speedup（v2, DART） | 最高 2.06x | NVIDIA H100 | LLaDA-8B |
| 能效 tok/J（v2） | 最高 23.3x | A6000 | LLaDA-8B |
| 能效 tok/J（v2） | 最高 15.5x | H100 | LLaDA-8B；「所有配置皆超越 H100」 |
| BAOS 4-bit KV 精度 | 與 BF16 baseline 持平或略高 | BF16 全精度 | GSM8K，全量化 DART 配置 |

## 限制 / 備註

- 版本差異大：v1 (2026-01) 的主張是 2.53x / 70%；v2 (2026-04) 改名 DART、加入 ISA/編譯器/雙引擎與 BAOS，數字變成 4.91x / 23.3x tok/J。撰寫時請標明引用哪一版。最新 arXiv 摘要（"Beyond GEMM-Centric NPUs" 題名）又回到 2.53x，代表兩個題名可能對應不同版本，待確認。
- 面積、功耗、頻率、SRAM 容量等 PPA 細節只在全文表格，本次無法取得（arXiv 被 proxy 擋住），標「未查到」。
- 對 GPU 的比較基準是 A6000（Ampere）與 H100；v1 只比 A6000，所以 2.53x 的說服力要看 GPU 端是否用了 Fast-dLLM 等優化 kernel（摘要未說明）。
- 只評估 LLaDA 系列（masked diffusion, 全注意力 + block-wise 解碼）；未涵蓋 Dream、block-diffusion 模型（Fast-dLLM v2 / SDAR）。

## 與其他論文的關係

- 與 **llada.cpp (mobile-npu-llada-cpp.md)** 互補：llada.cpp 是在既有商用手機 NPU 上用軟體對齊 dLLM；本文則是重新設計 NPU 微架構，直接在硬體層解掉取樣瓶頸。
- 與 **Hardware Acceleration of Block-Diffusion LLM for Edge Devices (block-diffusion-edge-hw.md)** 是「雲/桌面 vs. 邊緣」的兩條路線：本文對比 A6000/H100、以取樣引擎為核心；後者針對 batch-one 邊緣推論、以 LPDDR 頻寬與 KV/FFN 權重流量為核心。
- BAOS 與軟體側的 dLLM KV 量化 / cache 研究（dLLM-Cache、Fast-dLLM 的 block-wise KV cache）建立在同一個 block-wise 解碼觀察上；BAOS 的貢獻是把「warm step 重算」轉成免費的線上校準點。
- 「取樣佔 70% 延遲」的 profiling 結論與 **serving-mdlm-characterization.md**（CPU 端 dispatch 佔 75.6%）都指出 dLLM 的瓶頸不在 GEMM，兩者一硬一軟。
