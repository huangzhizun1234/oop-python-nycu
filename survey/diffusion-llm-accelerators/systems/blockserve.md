# BlockServe: Block-Grained Continuous Batching for High-Throughput Diffusion LLM Serving

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2607.08930` |
| 作者 / 單位 | Zihe Song et al.（University of Illinois Chicago，依搜尋結果推測，待確認） |
| 日期 | 2026-07 |
| 類別 | 系統與 serving |
| 連結 | [arXiv](https://arxiv.org/abs/2607.08930) · GitHub：未查到 |

## 一句話總結

BlockServe 把「每個執行 block」當成獨立的排程量子，讓已完成的請求在 block 邊界立即被驅逐、新請求以 token 預算補位，並用 gather-scatter 索引讓 dual cache 與平行解碼能在狀態不一的 batch 上運作，解決 dLLM batching 時的收斂異質性，在 Dream / LLaDA 上比 Fast-dLLM 高 1.9–10.6× 吞吐。

## 要解決的問題

- **收斂異質性（convergence heterogeneity）**：threshold-based 平行解碼下，不同請求每步 unmask 的 token 數差異很大，同一 batch 內有的請求幾步就收斂、有的要跑滿；靜態 batching（如 Fast-dLLM 的 batch 模式）必須等最慢的 straggler，早完成的請求持續佔用計算與 KV 記憶體，造成 compute bubble 與尾延遲。
- **AR 的 continuous batching 無法直接套用**：AR 以「token」為排程量子，每步都能換人；dLLM 一個 block 內多步 denoising 共用同一組 KV cache 與 mask 狀態，中途插拔請求會破壞 dual cache 的一致性。
- **batch 容量受限**：dLLM 的 activation / logit 峰值大，固定 batch size 會為最壞情況保留記憶體，實際利用率低。

## 核心方法與特色

- **Block-grained scheduling（block 粒度排程）**：把每個 block（例如 32 token 的解碼視窗）作為不可分割的排程單位，但在 block 邊界做決策：完成生成的請求立即驅逐（block-level eviction），釋放其 KV 與 slot；未完成的繼續下一個 block。這使 batch 組成每個 block 都能更新，而不必等整個 batch 全部結束。
- **Mixed-state execution + gather-scatter 索引**：batch 內各請求處於不同的 block 序號、不同的 cache 狀態（有的剛 refresh、有的在 reuse）；BlockServe 用 gather-scatter indexing 把異質請求的 KV 與 mask 位置收集成連續張量送入 kernel，再把結果散射回各自的 cache，讓 Fast-dLLM 的 dual cache（prefix + suffix cache）與 threshold parallel decoding 能在異質 batch 上正確運作。
- **Compute-aware admission controller（計算感知准入）**：不以固定 batch size 而以「每步 token 預算」決定可容納多少請求；驅逐釋放空間後用 token-budgeted refill 補進新請求，使有效 batch 容量在相同硬體下提升 2–4×。
- **不改模型架構、不需訓練**：完全在 serving 層實作，適用於現有 Dream / LLaDA 權重。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| Dream（Dream-v0-Instruct-7B，推測） | 7B | 五個 benchmark |
| LLaDA（LLaDA-8B-Instruct，推測） | 8B | 五個 benchmark |

## PPA / 效能數據

> 軟體論文：speedup、throughput (tok/s)、latency、記憶體、benchmark 準確度變化。

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| Throughput 提升 | 1.9×–10.6× | Fast-dLLM（static batching） | Dream 與 LLaDA，五個 benchmark（含 HumanEval；其餘未查到） |
| 有效 batch 容量 | 2–4× | 固定 batch size | 相同硬體，token-budgeted admission |
| 生成品質 | 「comparable」 | Fast-dLLM | 具體 accuracy 數字未查到 |
| 絕對 tok/s、GPU 型號、尾延遲數字 | 未查到 | — | — |

## 限制 / 備註

- 定位為 **offline / 高吞吐** dLLM 推論（論文自述「foundation for high-throughput offline dLLM inference」），對線上 SLO 與 TTFT 的討論較少。
- 10.6× 的上界出現在收斂異質性最嚴重的 workload；異質性低（例如所有請求長度相近且 threshold 低）時收益趨近 1.9×。
- 依賴 Fast-dLLM 式的 block-wise dual cache 與 threshold 解碼；若改用其他快取策略（如 dLLM-Cache 的特徵快取），gather-scatter 邏輯需重寫。
- 未查到開源程式碼與硬體細節。

## 與其他論文的關係

- 建立在 **Fast-dLLM**（dual cache + threshold parallel decoding）之上，並以其 batch 模式為 baseline。
- 與 **Sangam**（`sangam.md`）互補：Sangam 解決 prefill 不可分割造成的 decode stall，BlockServe 解決 decode 階段的 straggler；兩者都借用 AR continuous batching 的思想。
- 與 **dLLM-Serve**（`dllm-serve.md`）的 Phase-Multiplexed Scheduler 互補：dLLM-Serve 交錯 Refresh/Reuse phase，BlockServe 在 block 邊界換人並擴大 batch 容量。
- 與 **SGLang dLLM 框架** 的 dynamic batching / 「early block departure」路線圖項目概念相近（見 `sglang-dllm.md`）。
