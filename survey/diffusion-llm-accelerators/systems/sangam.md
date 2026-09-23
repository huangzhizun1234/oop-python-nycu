# Sangam: Efficiently Serving Diffusion LLMs with the AR Stack

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2607.04206` |
| 作者 / 單位 | Nitin Kedia et al.（UT Austin, UT-InfraAI；共同作者含 Aditya Akella，依搜尋結果推測） |
| 日期 | 2026-07 |
| 類別 | 系統與 serving |
| 連結 | [arXiv](https://arxiv.org/abs/2607.04206) · [GitHub](https://github.com/UT-InfraAI/sangam) |

## 一句話總結

Sangam 主張「有紀律地重用 AR serving stack」就是 serving cached dLLM 的正確基礎：以 FlashInfer + CUDA Graph 等 AR 核心元件為底，加上專為「不可分割的 dLLM prefill」設計的 deficit token-budget scheduler 與 colocated / disaggregated / hybrid 三種執行模式，在 LLaDA-8B 與 Dream-7B 上以相同延遲承受約 2.5–3× 於 Fast-dLLM 的負載。

## 要解決的問題

- **雙向注意力破壞 AR 的精確 KV cache**：commit 一個位置會改變其他所有位置的 KV，因此 dLLM 只能用近似快取（Fast-dLLM 式 block-wise cache），且每個 block 完成後要重算。
- **無法 chunked prefill**：AR serving（Sarathi-Serve、vLLM）靠 chunked prefill 把長 prompt 切片與 decode 混批，消除 prefill 對 decode 的 stall；但 dLLM 的 prefill 需要對整個 prompt 做雙向注意力，**不可分割**，一個長 prefill 進來就會讓同批 decode 請求整步停擺（head-of-line blocking）。
- **Fast-dLLM 等研究工具缺乏 serving 能力**：沒有 continuous batching、排程、SLO 控制；而 vLLM / SGLang 當時不支援 full-attention dLLM（LLaDA、Dream）。

## 核心方法與特色

- **Deficit token-budget scheduler（虧損式 token 預算排程）**：每個 iteration 有固定 token 預算 τ（例如 1024）；排程器先接納所有進行中的 decode，再檢查「累積預算」是否足以容納一整個不可分割的 prefill，容納不了就把未用完的預算累積到下一輪（carry forward）。長 prefill 因此會在累積夠多預算後才被排入，達到「攤銷後無 stall」（amortized stall-free）——decode 平均不被 prefill 卡住，而 prefill 也不會餓死。這是 deficit round-robin 思想在 dLLM 上的移植。
- **單一實作支援三種 P/D 執行模式**：colocated（同一 worker 混跑 prefill 與 decode）、static disaggregated（prefill 與 decode 分開 worker）、hybrid（conditional disaggregation：prefill worker 不夠用時把 prefill 溢流到 decode worker，並用同一個 deficit scheduler 保護該 worker 的 decode）。
- **重用 AR 核心元件而非重寫**：attention 用 FlashInfer kernel、decode 迴圈用 CUDA Graph，使 per-token 吞吐與 SGLang 相當；快取採 Fast-dLLM 風格的 block-wise 近似 KV cache。Sangam 不是 vLLM/SGLang 的 patch，而是獨立實作，但其排程、批次與 kernel 抽象皆借自 AR stack。
- **以真實 trace 評估**：使用 ShareGPT（decode-heavy）與 arXiv（prefill-heavy）trace，在 AWS p5.48xlarge（8×H100 80GB）上比較不同模式與 QPS 下的 mean / p99 延遲。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| LLaDA-8B-Instruct | 8B | bfloat16，ShareGPT trace 為主 |
| Dream-v0-Instruct-7B | 7B | bfloat16，arXiv trace 為主 |

## PPA / 效能數據

> 軟體論文：speedup、throughput (tok/s)、latency、記憶體、benchmark 準確度變化。

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| 可承受負載（matched latency） | 約 2.5–3× 更高 | Fast-dLLM（in-system 重現） | LLaDA-8B 與 Dream-7B，H100（GitHub README） |
| Mean latency 降低（decode-heavy） | 9–20% | Hybrid 模式 | Colocated 模式，LLaDA-8B，ShareGPT |
| Mean latency 降低（prefill-heavy） | 8–20% | Colocated 模式 | Hybrid 模式，Dream-7B，arXiv trace |
| Mean / p99 latency vs QPS | 圖 8（數值未查到） | Fast-dLLM | LLaDA-8B，單張 H100 80GB，Colocated，τ=1024 |
| 可持續 QPS | 委託方提供「約 1.0 QPS」，未查到佐證（待確認） | — | — |
| Per-token throughput | 與 SGLang 相當 | SGLang | FlashInfer + CUDA Graph |
| 準確度變化 | 沿用 Fast-dLLM 近似快取，論文未提供額外精度數字（未查到） | — | — |

## 限制 / 備註

- 「3× 負載」是「在相同延遲下可承受的 QPS 倍數」，非 tok/s 倍數；且對照組是論文自行在系統內重現的 Fast-dLLM。
- 只評測 8B 級 full-attention dLLM；對 block-diffusion 模型（LLaDA2.0、SDAR）而言 chunked prefill 問題較輕，Sangam 的優勢可能縮小。
- deficit scheduler 的 τ 需要依 GPU 與模型調整；τ 過小會讓長 prefill 等待過久（影響 TTFT）。
- 未與 SGLang 官方 dLLM 框架或 dInfer 直接比較。

## 與其他論文的關係

- 建立在 **Fast-dLLM** 的 block-wise KV cache 之上，並把 **Sarathi-Serve / vLLM** 的 stall-free 排程理念改造成不需 chunked prefill 的 deficit 版本。
- 與 **dLLM-Serve**（`dllm-serve.md`）互補：dLLM-Serve 處理 logit 記憶體與 Refresh/Reuse 排程，Sangam 處理 prefill/decode 干擾與 P/D 分離。
- 與 **BlockServe**（`blockserve.md`）同期：BlockServe 針對 block 粒度的 continuous batching 與 straggler；Sangam 針對 prefill 不可分割與 disaggregation。
- 其「重用 AR stack」的結論與 **SGLang dLLM 框架**（重用 chunked-prefill 管線）、**vLLM DiffusionGemma**（重用 speculative decoding 路徑）的工程路線一致（見 `sglang-dllm.md`、`other-frameworks.md`）。
