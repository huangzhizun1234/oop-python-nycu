# HERALD: High-Throughput Block Diffusion LLM Serving via CPU-GPU Cooperative KV Cache Retrieval

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2606.21633` |
| 作者 / 單位 | 未查到作者列表 |
| 日期 | 2026-06 |
| 類別 | 系統與 serving（KV cache offloading） |
| 連結 | [arXiv](https://arxiv.org/abs/2606.21633) · GitHub：未查到 |

## 一句話總結

HERALD 是第一個針對 block diffusion LLM 的 KV cache offloading 系統：把 KV cache 放在 host DRAM，利用「block 內各 denoising step 需要的 KV 子集幾乎不變」這個結構性質，讓 CPU 用一個 center-[MASK] query 每個 block 只選一次 top-k KV、GPU 只跨 PCIe 抓一次，在 5–10% KV 預算下幾乎無損，長上下文 decode 吞吐最高提升 2.47×。

## 要解決的問題

- dLLM 靠一次 forward 產多個 token 提升 GPU 利用率，但 KV cache 仍隨 context 線性成長；長上下文（如 32K）時 KV cache 佔滿 HBM，限制 batch 與吞吐。
- 把 KV offload 到 host DRAM 是 AR LLM 的常見解法（InfiniGen 等），但 PCIe 頻寬有限，只能回傳稀疏子集；AR 模型每個 token 都要重新選 top-k，選擇本身（CPU 端 attention score 估計）與 PCIe 傳輸成為瓶頸。
- 既有 AR-style offloading 方法（InfiniGen、MAGE-Offload）直接套在 block dLLM 上時，即使給到 20% KV 預算精度仍明顯下降，因為它們的選擇是逐 token、逐 step 的，與 dLLM 的 block-wise 雙向解碼不匹配。

## 核心方法與特色

- **Block 內 KV 相關性一致 → 每 block 只選一次**：觀察到 block dLLM 在同一個 block 的所有 denoising step 中，attention 關注的 KV entry 高度一致；因此只需在 block 開始時識別 top-k，之後所有 step 重用，把 PCIe 傳輸從「每 step」降為「每 block」一次，解除互連瓶頸。
- **單一 center-[MASK] query 做 block 級選擇**：不用 block 內所有位置的 query 去打分，而是取 block 中央的 [MASK] token 的 query 作為整個 block 的代表來估計 attention 分數，使 CPU 端選擇成本平均降低 **24.2×**，讓 CPU 跟得上 GPU。
- **Step-0 logits 作為語義代理以支援 prefetch**：下一個 block 的 query 理論上要等當前 block 解完才知道；HERALD 用當前 block 第 0 步的 argmax（draft）作為前綴代理，提前估計下一 block 的 query 並預先選取／預取 KV，把下一 block 的 KV 檢索完全與當前 block 的 denoising 重疊。
- **CPU-GPU 雙串流管線與雙緩衝稀疏 KV 池**：GPU 串流負責 denoising，CPU 串流負責 top-k 選擇與 H2D 傳輸；GPU 上維護 double-buffered sparse KV pool，一個 buffer 供當前 block 計算，另一個接收下一 block 的 KV，避免同步等待。
- **建構在 SGLang 之上**：以 SGLang 官方 dLLM 框架為基底實作，繼承其 batching、CUDA Graph 與 block diffusion 支援。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| LLaDA2.0-mini | 16B MoE | 三個「production block dLLM」之一（搜尋結果提及） |
| SDAR | 未查到具體規格（SDAR 系列 1.7B–30B） | 三個 block dLLM 之一（搜尋結果提及） |
| 第三個 block dLLM | 未查到 | 論文稱「three production block dLLMs」 |

## PPA / 效能數據

> 軟體論文：speedup、throughput (tok/s)、latency、記憶體、benchmark 準確度變化。

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| 最大 decode throughput | 最高 2.47× | Dense（全 KV 在 GPU） | 32K context；speedup 隨 context 增長 |
| 精度（LongBench 五項長上下文任務） | 近乎無損，與 in-GPU MAGE 相差數個百分點內 | Dense / MAGE | KV 預算 5–10% |
| AR-style offloading 對照 | 即使 20% 預算仍明顯掉精度 | InfiniGen、MAGE-Offload | 同任務 |
| CPU 選擇成本 | 平均降低 24.2× | 全 query 打分 | center-[MASK] query |
| KV 預算 | 5–10% | 100%（Dense） | 近乎無損 |
| 硬體 | Intel Xeon Platinum 8481C、936 GB DDR5-4800、NVIDIA H100 SXM 80GB、PCIe 5.0 | — | — |
| 絕對 tok/s | 未查到 | — | — |

## 限制 / 備註

- 只適用於 **block diffusion** dLLM（LLaDA2.0、SDAR、Fast-dLLM v2 等）；full-attention dLLM（LLaDA-8B、Dream）沒有「block 內 KV 相關性一致」的性質。
- 依賴 host 端大量 DRAM（實驗機 936 GB）與 PCIe 5.0；在 PCIe 4.0 或 DRAM 較小的機器上收益可能縮小。
- step-0 argmax 作為 prefetch 代理是近似，若 block 早期預測與最終結果差異大，預取到的 KV 可能不準（論文宣稱近乎無損，但條件限定在 5–10% 預算與其測試任務）。
- 未查到 GitHub 連結與完整作者名單。

## 與其他論文的關係

- 建立在 **SGLang dLLM 框架**（`sglang-dllm.md`）之上，是官方框架被學術系統作為基底的例子。
- 與 AR 的 KV offloading 系統 **InfiniGen**、**MAGE** 直接比較，並指出 AR-style 逐 token 選擇不適合 dLLM。
- 與 dLLM KV 稀疏 / 淘汰工作（**Sparse-dLLM**、*Mask Tokens as Prophet*、**GroupKV**（arXiv 2609.17573，長上下文 dLLM 階層式 KV 管理））互補：那些工作在 GPU 內裁剪 KV，HERALD 把完整 KV 留在 DRAM 只回傳子集，不丟資訊。
- 與 **dLLM-Serve** 的 Head-Centric Sparse Attention 同樣關注稀疏 KV 的物理存取效率，但 HERALD 的重點在跨 PCIe 的檢索與重疊。
