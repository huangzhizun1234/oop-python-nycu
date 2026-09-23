# Power Up Diffusion LLMs: Day-0 Support for LLaDA 2.0（SGLang dLLM Framework）

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | 非論文：LMSYS / SGLang 官方部落格（2025-12-19）；設計 RFC 為 GitHub issue sgl-project/sglang#12766；路線圖 issue #14199（2025 Q4–2026 Q1）與 #39499 |
| 作者 / 單位 | SGLang 團隊（LMSYS）與 Ant Group（inclusionAI）合作 |
| 日期 | 2025-12 |
| 類別 | 系統與 serving |
| 連結 | [Blog](https://www.lmsys.org/blog/2025-12-19-diffusion-llm/) · [RFC #12766](https://github.com/sgl-project/sglang/issues/12766) · [Roadmap #14199](https://github.com/sgl-project/sglang/issues/14199) · [Roadmap #39499](https://github.com/sgl-project/sglang/issues/39499) · [LLaDA2.0](https://github.com/inclusionAI/LLaDA2.0) |

## 一句話總結

SGLang 觀察到 block diffusion 的計算模式與自家 chunked-prefill 幾乎一樣，於是只改 prefill adder 與 chunked-request handler 兩個元件、在 TP worker 與 model runner 之間插入一層「diffusion algorithm」抽象，就讓 LLaDA 2.0（16B mini / 100B flash MoE）在 day-0 取得 batching、TP/EP、CUDA Graph、KV cache 與 RL 整合等生產級能力，LLaDA2.0-flash-CAP 以 0.95 threshold 解碼達 500 TPS、較 AR 基準快最多 1.9×。

## 要解決的問題

- dLLM 進展快，但推論與 RL 後訓練工具落後：Fast-dLLM、dInfer 等偏研究工具，缺乏 continuous batching、排程、多卡並行與 RL 生態整合，難以支撐大規模評測與 RL rollout。
- dLLM 的解碼策略比 AR 多樣（threshold、entropy、edit/insertion 等），框架需要讓使用者能自定義解碼演算法，而不是每個模型硬寫一套。
- 100B 級 MoE dLLM（LLaDA2.0-flash）需要 TP+EP、FP8 與 CUDA Graph 才能實用。

## 核心方法與特色

- **重用 chunked-prefill 管線**：block diffusion 每次對一個固定長度 block（含 [MASK]）做 forward，並讀取前面所有已完成 block 的 KV——這與 chunked prefill「一塊新 token 對前綴 KV 做 attention」的模式相同，差別只在 mask 形狀（block-wise causal 的矩形 vs token-wise causal 的梯形）；「Cache Query」計算完全一致。因此不需改核心架構即可繼承 SGLang 既有的 paged KV、radix cache、TP/EP、CUDA Graph 等優化。
- **Diffusion algorithm 抽象層**：在 TP Worker 與 Model Runner 之間插入抽象，偵測到 diffusion 模型時由 TP Worker 呼叫演算法的 `run` 函式，該函式驅動 block 內的迭代 forward 直到整個 block unmask 完成。演算法可插拔：low-confidence threshold 平行解碼、JointThreshold（LLaDA2.1 的 token editing）、GFusion（entropy-bounded sampling）、FastDiffuser 等皆以此註冊。
- **KV cache 寫入時機**：block 迭代期間不寫 cache（因為 token 尚在變動），整個 block unmask 完成後再做一次 forward，把該 block 的 KV 一次寫入 token pool；這使 KV cache 語意與 AR 完全一致，radix prefix cache 可直接重用。
- **Dynamic batching**：batch 大小由 `max-running-requests` 與剩餘 token 預算共同決定；chunked-request 元件把各請求截出的 block input ids 重組成一個 block diffusion batch。後續路線圖加入 overlap scheduling、「FDFO 與 early block departure」（完成的請求在 block 邊界離開）、block-level radix caching。
- **生產級週邊**：TP + EP（100B MoE 必需）、CUDA Graph（固定 block 與 piecewise prefill）、FP8（含 MoE 的 SwapAB）、streaming I/O、temperature / top-p / top-k、RL 用的 step maps；硬體支援 NVIDIA、AMD（Triton / AIter）、Ascend NPU（graph mode）。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| LLaDA2.0-mini（-preview / -CAP） | 16B MoE | day-0 支援 |
| LLaDA2.0-flash（-preview / -CAP） | 100B MoE | day-0 支援；500 TPS 數字來源 |
| LLaDA2.1 / LLaDA2.2 | 16B / 100B MoE | JointThreshold token editing、block routing / insertion-deletion（路線圖） |
| SDAR（dense 與 MoE） | 1.7B–30B（依系列） | 已加入支援 |
| Fast-dLLM v2 | 7B（Qwen2.5 backbone） | AR wrapper + hierarchical decoder（路線圖 / WIP） |
| DiffusionGemma | 26B | 文字與影像輸入 serving（路線圖） |
| Nemotron Labs Diffusion | 未查到 | FastDiffuser 解碼（路線圖） |
| Dream-v0-Base-7B | 7B | full-sequence baseline（非 block diffusion，後期支援） |

## PPA / 效能數據

> 軟體論文：speedup、throughput (tok/s)、latency、記憶體、benchmark 準確度變化。

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| LLaDA2.0-flash-CAP 吞吐 | 500 TPS | LLaDA2.0-flash 383 TPS | threshold decoder 0.95，小 batch（GPU 型號未查到；部落格搜尋摘要） |
| vs AR 基準 | 最多 1.9× | AR baselines 258 TPS 與 237 TPS | 小 batch |
| LLaDA2.0 官方宣稱 | 2.1× 推論加速；LLaDA2.0-flash-CAP 最高 535 tok/s | AR 基準 | 基於 dInfer + SGLang 客製引擎（LLaDA2.0 README） |
| dInfer + SGLang 後端（LLaDA2.0-flash-CAP） | 580.70 TPS（bs=1）/ 1,966.34 TPS（bs=32） | — | 8×H20，gen_len=1000（dInfer README） |
| 準確度 | RFC 標註「model accuracy verification WIP」（2025-11）；部落格宣稱與 HF 實作一致（未查到數字） | — | — |

## 限制 / 備註

- 初版只支援 **block diffusion** 模型（LLaDA2.0 系列）；full-attention dLLM（LLaDA-8B、RND1、Dream）在 RFC 階段明確不支援，Dream baseline 與「non-block diffusion」在後續路線圖才加入。
- 500 / 383 / 258 / 237 TPS 的硬體與 batch 條件僅來自搜尋摘要，未能直接讀取部落格全文（lmsys.org 被 proxy 阻擋），細節待確認。
- block 迭代中不寫 KV 意味著 block 內每步都要重新讀取前綴 KV，長上下文時 I/O 成本仍高（HERALD 即針對此點做 offloading）。
- 為非同行審查的工程部落格，無正式消融實驗。

## 與其他論文的關係

- 是 Ant Group **LLaDA2.0 / 2.1 / 2.2** 的官方 serving 路徑，與 **dInfer**（`dinfer.md`）互補並在 dInfer v0.2 整合為後端。
- **HERALD**（`herald.md`）直接建構在 SGLang dLLM 框架上做 CPU-GPU KV 檢索。
- 「重用 AR stack」路線與 **Sangam**（`sangam.md`）的結論、**vLLM DiffusionGemma**（`other-frameworks.md`，重用 speculative decoding 路徑）一致；差別在 SGLang 選擇 chunked-prefill 作為對映點。
- 路線圖中的 dynamic batching / early block departure 與 **BlockServe**（`blockserve.md`）的 block-grained continuous batching 概念相同。
