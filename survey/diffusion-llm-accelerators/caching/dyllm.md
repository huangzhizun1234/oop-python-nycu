# DyLLM: Efficient Diffusion LLM Inference via Saliency-based Token Selection and Partial Attention

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2603.08026`（ICML 2026） |
| 作者 / 單位 | Younjoo Lee, Junghoo Lee, Seungkyun Dan, Jaiyoung Park, Jung Ho Ahn（Seoul National University, SCALE Lab） |
| 日期 | 2026-03 |
| 類別 | KV cache 與稀疏（token 選擇 / 部分注意力） |
| 連結 | [arXiv](https://arxiv.org/abs/2603.08026) · GitHub：未查到 |

## 一句話總結

以「相鄰 denoising step 之間 attention context 的 cosine 相似度」找出真正在變的 salient token，只對它們重算 attention 與 FFN、其餘沿用快取，並用 salient-aware 近似注意力，在 LLaDA / Dream 上分別達 7.6× / 9.6× 吞吐提升且幾乎無損。

## 要解決的問題

- masked dLLM 每步對整段序列重算 attention 與 FFN，但大多數 token 的表徵在相鄰 step 間幾乎不變，只有一小部分 salient token 真正推動下一步更新。
- 既有 cache 方法（dLLM-Cache）每步固定重算一個比例的 token，且超參數需依模型 / 資料集微調；Fast-dLLM 以 block 為單位刷新，粒度太粗。
- 需要一個能動態決定「每步重算多少、重算哪些」的 training-free 機制。

## 核心方法與特色

- **Saliency 定義**：對每個 token 比較本步與上一步的 attention context（注意力輸出）cosine 相似度，相似度低者為 salient token；這比比較 KV 或 hidden state 更能反映「該 token 對其他 token 的影響是否改變」。
- **動態數量而非固定比例**：salient token 的數量由相似度閾值決定，隨 step 自然變化（早期多、收斂後少），不像 dLLM-Cache 固定 ρ。
- **只重算 salient token 的 attention + FFN**：非 salient token 的 attention 輸出與 FFN 輸出直接從快取讀回，因此省下的是整個 transformer block 的計算，不只 KV。
- **Salient-aware approximate attention**：salient token 作為 query 時仍要看全部 key/value，但 DyLLM 對 key/value 側也做近似（沿用快取的 KV），並以自訂 CUDA kernel 實作稀疏 attention 與 cache gather/scatter，使加速反映在真實 wall-clock。
- **跨 GPU 世代一致**：在 H100 與 B200 上皆觀察到一致的加速。
- **代價**：需要每步計算相似度與 gather/scatter，token 數少時 kernel 效率下降；閾值仍為超參數。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| LLaDA-8B | 8B | 7.6× |
| Dream-7B | 7B | 9.6× |

## PPA / 效能數據

> 軟體論文：speedup、throughput (tok/s)、latency、記憶體、benchmark 準確度變化。GPU：NVIDIA H100 與 B200。

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| 吞吐提升（LLaDA-8B） | 最高 7.6× | 原始 LLaDA 實作 | — |
| 吞吐提升（Dream-7B） | 最高 9.6× | 原始 Dream 實作 | — |
| 準確度 | 持平或略優於原模型，優於 Fast-dLLM 與 dLLM-Cache | 原模型 / Fast-dLLM / dLLM-Cache | GSM8K、MMLU-Pro、MBPP |
| 逐 benchmark tok/s | 論文未在可用來源提供 / 未查到 | — | — |

## 限制 / 備註

- 具體逐 benchmark 的吞吐與準確度數字未從可用來源取得；GitHub repo 未查到。
- 需要自訂 CUDA kernel，移植到非 NVIDIA 平台有門檻。
- 與 suffix pruning 類方法（DPad）是否可疊加未見報告。

## 與其他論文的關係

- 直接以 **dLLM-Cache**（固定比例 V-verify）與 **Fast-dLLM**（block cache）為 baseline，主張動態 saliency 優於固定比例。
- 與 **d²Cache**（certainty prior + attention-aware 選 token）、**Elastic-Cache**（most-attended token 漂移）、**ES-dLLM**（tensor variation + 信心估重要性）同屬「自適應 token 級重算」家族，四者於 2025-09 至 2026-03 間密集出現。
- 「快取 FFN 輸出而不只 KV」承襲 **dLLM-Cache**，並被 **ES-dLLM** 進一步推到「淺層直接跳過」。
