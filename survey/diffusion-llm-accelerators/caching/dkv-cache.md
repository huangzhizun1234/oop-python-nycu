# dKV-Cache: The Cache for Diffusion Language Models

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2505.15781`（NeurIPS 2025） |
| 作者 / 單位 | Xinyin Ma, Runpeng Yu, Gongfan Fang, Xinchao Wang（xML Lab, National University of Singapore） |
| 日期 | 2025-05 |
| 類別 | KV cache 與稀疏 |
| 連結 | [arXiv](https://arxiv.org/abs/2505.15781) · [GitHub](https://github.com/horseee/dKV-Cache) |

## 一句話總結

觀察到 dLLM 中「已解碼 token」的 KV 在解碼後幾乎不再變動，提出「延遲一步再快取」的 delayed KV-Cache，使雙向注意力的 dLLM 也能重用 KV，得到 2–10× 加速且幾乎不損（長序列甚至提升）品質。

## 要解決的問題

- dLLM 是非自迴歸且使用雙向注意力，token 的 KV 理論上會隨每一步的 unmask 而變化，因此 AR 式「算完就快取」不成立。
- 現有 dLLM（LLaDA、Dream）每一 denoising step 都重算整段序列，推論速度遠慢於 AR 模型。
- 需要一個能在預訓練 dLLM 上直接套用、不需訓練的 cache 機制。

## 核心方法與特色

- **關鍵觀察：token 的表徵動態分為兩類**。mask token 的 KV 每步都變，但一個 token 一旦被解碼（unmask），其 KV 表徵在後續 step 幾乎固定。因此只對「已解碼 token」做 cache，mask token 仍每步重算。
- **Delayed caching（延遲一步）**：token 被解碼的那一步其 KV 仍在劇烈變化（因為 embedding 由 mask 換成真實 token），所以不在解碼當下就快取，而是延後一步（one-step delay）等表徵穩定後再存入 cache。這個小技巧是準確度幾乎無損的關鍵。
- **dKV-Cache-Decode（近乎無損版）**：對已解碼 token 使用 delayed cache，並每隔 `cache_steps`（2–16 步）刷新一次；在長生成（L=512）上甚至能提升準確度（推測是 cache 抑制了已解碼 token 的表徵漂移）。
- **dKV-Cache-Greedy（激進版）**：縮短 cache 生命週期、只重算一個小視窗（例如 window=4）內的 token，把每步計算從 O(L²) 降到接近視窗大小，換取更高加速但有可觀的準確度下降。
- **dKV-Cache-PD / Prefill**：針對長 prompt 的 prefill 階段快取 prompt KV，在 MMLU / GPQA 這類長輸入短輸出的任務上達到最高約 10× 加速。
- **代價**：加速幅度依 batch size、prefill 與生成長度而異；作者在 README 註明 batch_size=1 時加速不明顯，需較大 batch 才能看出效果。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| LLaDA-8B-Instruct | 8B | 主要實驗 |
| Dream-v0-Base-7B | 7B | 長生成（L=512）上準確度反而提升 |

## PPA / 效能數據

> 軟體論文：speedup、throughput (tok/s)、latency、記憶體、benchmark 準確度變化。GPU 型號論文摘要未明示（未查到）。

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| 整體加速範圍 | 2–10× | 原始 LLaDA / Dream 實作 | 依 prefill 長度與生成長度而異 |
| GSM8K（dKV-Cache） | 6.6× 加速，Pass@1 提升至 63.31 | 原始實作 | decoding length 256、64 steps |
| LLaDA-8B-Instruct GSM8K 0-shot | Decode 版：78.85%，2.35×；Greedy 版（window 4）：68.23%，1.63× | 原始 LLaDA | — |
| Dream-Base-7B, L=512 | GSM8K 80.97→83.13；HumanEval 39.63→46.34 | 原始 Dream | dKV-Cache-Decode |
| 長 prefill 任務 | 最高約 10× | 原始實作 | MMLU / GPQA，dKV-Cache-Prefill |
| cache 刷新間隔 | 2–16 steps | — | `cache_steps` 超參數 |

## 限制 / 備註

- Greedy 版的準確度下降明顯（GSM8K 78.85→68.23），實務上多用 Decode 版。
- 作者自述 batch_size=1 時加速不顯著（因為原始實作已為 memory-bound），需較大 batch。
- 只快取「已解碼 token」，mask token 仍需每步重算，因此對長生成、短 prompt 的情境加速有限，這正是後來 Fast-dLLM DualCache、Sparse-dLLM 針對 suffix 的改進點。
- 依賴 `transformers==4.46.3` 的自製 attention 實作，尚未整合 FlashAttention / paged cache。

## 與其他論文的關係

- 與 **Fast-dLLM**（block-wise approximate cache）和 **dLLM-Cache**（feature-level cache）幾乎同時發表，是 dLLM KV cache 三篇開創性工作之一；差異在於 dKV-Cache 以「token 是否已解碼」為快取條件，Fast-dLLM 以「block」為單位。
- 「延遲一步再快取」的思想被 **Sparse-dLLM** 的 delayed bidirectional sparse caching 直接沿用並擴充到 suffix 淘汰。
- 同實驗室（NUS xML Lab）後續提出 **SparseD**（稀疏注意力）與 dParallel，可與 dKV-Cache 疊加。
- **Elastic-Cache**、**d²Cache** 以 dKV-Cache 為 baseline，批評其固定週期刷新（cache_steps）不具適應性，改以注意力漂移 / 確定性先驗決定何時刷新。
