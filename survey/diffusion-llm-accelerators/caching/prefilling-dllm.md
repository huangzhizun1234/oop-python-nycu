# Prefilling-dLLM: Predictive Prefilling for Long-Context Inference in Diffusion Language Models

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2606.10537`（EMNLP 2026） |
| 作者 / 單位 | Jing Xiong, Qi Han, Shansan Gong, Yunta Hsieh, Chengyue Wu, Chaofan Tao, Chenyang Zhao, Ngai Wong（香港大學等） |
| 日期 | 2026-06（v2：2026-08） |
| 類別 | KV cache 與稀疏（長上下文 prefill / chunk 選擇） |
| 連結 | [arXiv](https://arxiv.org/abs/2606.10537) · [GitHub](https://github.com/menik1126/Prefilling-dLLM) |

## 一句話總結

第一個為 dLLM 做原生 prefill/decode 分離的推論引擎：把長 prefix 切成 N 個 chunk 各自 prefill 一次並快取 KV，解碼時只挑 top-K 相關 chunk（可再做 chunk 內 token 稀疏）參與注意力，並以能直接在非連續 chunk KV 上運算的 kernel，在 8K–32K 上下文達 9.1–28.0× 加速。

## 要解決的問題

- dLLM 每個 denoising step 都要把整個 prefix 重新編碼，計算量 O((L_p + L_d)² · T)，長上下文（8K 以上）時完全不可行。
- 即使有 prefix KV cache，全部 prefix KV 仍參與每步注意力，長 prompt 時注意力成本仍主導。
- 長 prompt 下 dLLM 也有 lost-in-the-middle 問題，一味保留全部上下文不一定有利品質。

## 核心方法與特色

- **Chunked prefill（一次性）**：prefix 長度 L_p 切成 N 個大小為 C 的 chunk，每個 chunk 前加一個 BOS token 作為注意力錨點（attention anchor），各 chunk 獨立 prefill 一次並快取 KV，之後在整個解碼過程中固定不變；複雜度中 prefix 部分變成 O(N·C²) 且只做一次。
- **Query-conditioned top-K chunk 選擇**：以少量 draft token 的 self-information / 相關性分數評估每個 chunk 對當前 query 的貢獻，保留 top-K 個 chunk；週期性的 chunk-level BOS 錨點同時緩解 lost-in-the-middle。
- **Intra-chunk token sparsity**：在被選中的 chunk 內可再依 token_capacity 淘汰低重要 token，把每步注意力的 key 數量固定在預算內。
- **非連續 chunk KV 的 attention kernel**：解碼時不把選中的 chunk KV gather 成連續 cache，而是用自訂 kernel 直接對散落的 chunk KV 平行做注意力，避免 gather 開銷；每步複雜度降為 O((L_d² + K·B)·T)，只與解碼長度平方相關。
- **Training-free、涵蓋多種 dLLM**：套用在 Dream-7B、UltraLLaDA 與 Fast-dLLM v2（block diffusion）上，在 LongBench / InfiniteBench 取得 dLLM 加速方法中最佳品質。
- **代價**：chunk 獨立 prefill 失去 chunk 間的交互（以 BOS 錨點與 top-K 補償）；chunk 大小 C、K、token_capacity 需設定；chunk 選擇是查詢相關的近似，可能漏掉關鍵資訊。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| Dream-7B | 7B | — |
| UltraLLaDA | 8B | 長上下文 LLaDA 延伸版 |
| Fast-dLLM v2 | 7B（Qwen2.5 backbone） | block diffusion 模型 |

## PPA / 效能數據

> 軟體論文：speedup、throughput (tok/s)、latency、記憶體、benchmark 準確度變化。GPU 型號未查到。

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| 8K 上下文加速 | 9.1× | 原始 dLLM（全 prefix 重算） | LongBench / InfiniteBench |
| 16K 上下文加速 | 16.1× | 同上 | — |
| 32K 上下文加速 | 28.0× | 同上 | — |
| 品質 | dLLM 加速方法中 SOTA | 其他 dLLM 加速方法 | LongBench、InfiniteBench |
| 逐任務分數 | 未查到 | — | — |

## 限制 / 備註

- LongBench / InfiniteBench 的逐任務分數與 GPU 型號未從 README 取得。
- 加速隨上下文長度線性放大，是長上下文專用方案；短 prompt 情境收益有限。
- prefill/decode 分離的架構意味著可以做到 serving 系統層級的 prefix 共享，但論文未查到多請求 serving 的實驗。

## 與其他論文的關係

- 承接 **Fast-dLLM**（作者 Chengyue Wu 同時是 Fast-dLLM 第一作者）的 prefix cache，把「一次 prefill、固定 KV」推到長上下文並做 chunk 級稀疏。
- 與 **Focus-dLLM**（信心引導 + sink-aware pruning，32K 上 29.6×）和 **SparseD**（head-specific pattern）同為長上下文 dLLM 稀疏注意力方案，數字等級相近但 Prefilling-dLLM 的粒度是 chunk。
- 「chunk + BOS 錨點 + top-K 檢索」與 AR 模型的 chunked / speculative prefill、RAG 式 KV 檢索思路相通；HERALD 等 dLLM serving 工作延伸為 CPU-GPU 協同 KV 檢索。
- 可與 **dLLM-Cache**、**Elastic-Cache** 等 response 側 cache 疊加，構成 prompt + response 完整快取方案。
