# ES-dLLM: Efficient Inference for Diffusion Large Language Models by Early-Skipping

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2603.10088`（ICLR 2026 poster） |
| 作者 / 單位 | Zijian Zhu, Fei Ren, Zhanhong Tan, Kaisheng Ma（清華大學） |
| 日期 | 2026-03 |
| 類別 | KV cache 與稀疏（token 剪枝 / 層級跳過） |
| 連結 | [arXiv](https://arxiv.org/abs/2603.10088) · [OpenReview](https://openreview.net/pdf?id=O2WvMkJbws) · GitHub：未查到 |

## 一句話總結

觀察到 dLLM 相鄰迭代間 K、V、hidden state 只有細微變化，因此在「早期層」直接跳過不重要 token 的計算、只對重要 token 做部分 cache 更新，在 H200 上把 LLaDA-8B / Dream-7B 推到 226.6 / 308.5 TPS，相對原始實作 5.6–16.8×、相對 SOTA cache 方法最高 1.85×。

## 要解決的問題

- 既有 cache 方法（Fast-dLLM、dLLM-Cache）以粗粒度 token 啟發式決定刷新，且只快取 KV，計算量最大的 FFN 仍每步全算。
- 每一層都對所有 token 做完整計算，但多數 token 的中間張量（K/V/hidden）在相鄰迭代幾乎相同。
- 需要一種能同時省下 attention 與 FFN、並且細到 token × layer 粒度的跳過機制。

## 核心方法與特色

- **Token 重要性估計**：結合兩個訊號——(1) 前一輪迭代中該 token 中間張量（K、V、hidden state）的變化量，(2) 前一輪的信心分數；變化大或信心變動的 token 視為重要。這兩個訊號都是上一輪的副產品，估計成本極低。
- **Early-skipping（早期層跳過）**：對被判定為不重要的 token，在網路的前若干層完全不計算（attention + FFN 都跳），直接以快取的 K/V/hidden state 代替；只有重要 token 走完整前向。淺層跳過的理由是淺層表徵最穩定（與 Elastic-Cache「深層先變」的觀察一致）。
- **Partial cache update**：cache 中只有重要 token 的項目被刷新，其餘保持不變，避免整層重算。
- **Training-free 且與 cache 方法正交**：可疊在 Fast-dLLM / dLLM-Cache 之上，因此宣稱相對 SOTA cache 方法再 1.85×。
- **代價**：跳過的 token 在淺層使用舊表徵，會累積近似誤差；重要性估計依賴上一輪訊號，在早期 step 需 warm-up；跳過比例與層數為超參數。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| LLaDA-8B | 8B | 最高 226.57 TPS |
| Dream-7B | 7B | 最高 308.51 TPS |

## PPA / 效能數據

> 軟體論文：speedup、throughput (tok/s)、latency、記憶體、benchmark 準確度變化。GPU：NVIDIA H200。

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| LLaDA-8B 吞吐 | 最高 226.57 TPS | 原始實作 | H200 |
| Dream-7B 吞吐 | 最高 308.51 TPS | 原始實作 | H200 |
| 相對原始實作加速 | 5.6×–16.8× | vanilla LLaDA / Dream | 五個 benchmark |
| 相對 SOTA cache 方法 | 最高 1.85× | 既有最佳 caching 方法（dLLM-Cache / Fast-dLLM 類） | — |
| 評測任務 | GSM8K、MATH、HumanEval、MBPP、BBH | — | 品質「preserved」 |
| 逐任務準確度 | 未查到 | — | — |

## 限制 / 備註

- 逐 benchmark 的準確度與 tok/s 未從可用來源取得；GitHub repo 未查到。
- 高 TPS 數字部分來自 H200 硬體與批次設定，跨論文比較時需注意 GPU 差異（多數 cache 論文用 A100）。
- 跳過淺層計算是近似，對長生成或需要精細推理的任務可能累積誤差。

## 與其他論文的關係

- 以 **dLLM-Cache** 與 **Fast-dLLM** 為主要對照，批評兩者只快取 KV、未處理 FFN 且粒度粗。
- 與 **Elastic-Cache**（從深層開始刷新、淺層重用）在「層級」維度互補；與 **DyLLM**（saliency 決定重算 token）在「token 級」維度思路相同，但 ES-dLLM 加上層維度。
- 「用前一輪信心估重要性」與 **Focus-dLLM** 的 past-confidence indicator 同源。
- 屬於 dLLM 版的 early-exit / layer-skip（類似 AR 的 AdaSkip、Unified Layer Skipping）。
