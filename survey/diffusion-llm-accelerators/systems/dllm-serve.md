# Taming the Memory Footprint Crisis: System Design for Production Diffusion LLM Serving (dLLM-Serve)

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2512.17077`（v2） |
| 作者 / 單位 | 未查到完整作者列表（alphaXiv 頁面關聯作者 Yanglin Zhang，待確認） |
| 日期 | 2025-12 |
| 類別 | 系統與 serving |
| 連結 | [arXiv](https://arxiv.org/abs/2512.17077) · [GitHub](https://github.com/chosen-ox/dLLM-Serve) |

## 一句話總結

dLLM-Serve 指出 dLLM serving 的真正瓶頸是「記憶體足跡危機」——巨大的 logit 張量與 Refresh / Reuse 兩種 phase 的資源震盪——並用 Logit-Aware Activation Budgeting、Phase-Multiplexed Scheduler 與 Head-Centric Sparse Attention 三招把 LLaDA-8B 的吞吐在 RTX 4090 / L40S 上提升 1.6–1.8×、尾延遲降低近 4×。

## 要解決的問題

- **Logit 張量吃掉 KV 空間**：dLLM 每步 denoising 都要對「整段 response（含所有 [MASK]）」輸出 logits，張量大小 = batch × 序列長 × 詞彙表（LLaDA 詞彙表約 126K），一次就佔用數 GB 的瞬時記憶體，逼得 serving 系統只能開很小的 batch，形成吞吐牆（委託方提供數字：約 0.25 RPS，改善後 0.4–0.5 RPS；此數字未在搜尋結果中查到，待確認）。
- **Refresh vs Reuse 的資源震盪**：使用 KV cache 的 dLLM 會在「Refresh phase」（重新計算整段 KV，compute-bound，類似 prefill）與「Reuse phase」（只算當前 block，bandwidth-bound，類似 decode）間交替；若同一 batch 內所有請求同步進入同一 phase，GPU 會在算力飽和與頻寬飽和之間擺盪，兩種資源都無法同時用滿。
- **稀疏注意力的儲存問題**：既有 dLLM KV 稀疏化（如 Sparse-dLLM、dLLM-Cache）多在邏輯層做 token 篩選，但 per-head 的不規則保留在物理記憶體上會造成碎片與非合併存取，實際 serving 拿不到收益。

## 核心方法與特色

- **Logit-Aware Activation Budgeting**：把最後一層 vocabulary projection 的 logit 計算拆成序列化的 sub-batch（或子區段）依序計算，將瞬時 activation 峰值鎖在固定預算內；省下來的瞬時記憶體轉為「持久」的 KV cache 容量，使同樣 GPU 能容納更多並行請求。代價是 lm_head 多次 launch 的小額延遲。
- **Phase-Multiplexed Scheduler**：排程器追蹤每個請求目前處於 Refresh（compute-bound）或 Reuse（bandwidth-bound）phase，刻意把不同 phase 的請求交錯排進同一個 iteration，讓計算密集與頻寬密集的工作互補（概念類似 AR serving 的 prefill/decode multiplexing，但這裡是同一請求在生命週期內反覆切換 phase）。結果是 GPU 兩種資源同時被用到，並避免所有請求同時 refresh 造成的延遲尖峰。
- **Head-Centric Sparse Attention / Sparse KV Cache**：允許每個 attention head 保留不同的 token 子集（邏輯稀疏），但在物理層採用密集、連續的儲存布局（dense layout），把「哪些 token 被保留」的索引與「資料放在哪」解耦；因此可以做精確的 per-head 不規則裁剪，同時保持 coalesced memory access。
- **面向生產環境的完整系統**：提供 REST API server、FlashAttention 後端、約 1,200 行 Python 的精簡程式碼；支援 LLaDA-8B-Instruct 與 Dream-v0-Instruct-7B。
- **針對消費級與伺服器級 GPU 都驗證**：在 24 GB 的 RTX 4090 與 48 GB 的 L40S 上評測，強調記憶體受限場景。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| LLaDA-8B-Instruct | 8B | 主要評測模型 |
| Dream-v0-Instruct-7B | 7B | 程式碼支援；論文中是否評測未查到 |

## PPA / 效能數據

> 軟體論文：speedup、throughput (tok/s)、latency、記憶體、benchmark 準確度變化。

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| Throughput 提升 | 1.61×–1.81× | state-of-the-art baseline（論文未在摘要點名，推測為 Fast-dLLM 式 serving） | RTX 4090，LLaDA-8B |
| Throughput 提升 | 1.60×–1.74× | 同上 | NVIDIA L40S，LLaDA-8B |
| 峰值吞吐（RTX 4090） | 76.6 tok/s | 1.81× vs 最強 baseline | Burst 資料集 |
| 峰值吞吐（L40S） | 106.95 tok/s | 摘要記為 3.12× vs baseline（與 1.60–1.74× 的敘述不一致，可能是不同 baseline，待確認） | Burst 資料集 |
| 尾延遲（tail latency） | 降低近 4× | 同上 | 高競爭（heavy contention）負載 |
| 吞吐牆（RPS） | 0.25 → 0.4–0.5 RPS | — | 委託方提供，未查到佐證（待確認） |
| 準確度變化 | 論文宣稱「co-optimize generation quality」，具體數字未查到 | — | — |

## 限制 / 備註

- 評測集中在單卡、記憶體受限的 GPU（4090 / L40S），未見 H100 多卡或 MoE 模型結果；對於記憶體充裕的環境，logit budgeting 的收益可能較小。
- Phase-multiplexing 的前提是 batch 內請求 phase 足夠分散；在突發同步到達（burst）的情況下需要額外的錯位策略，論文以 Burst 資料集測試，但摘要未說明錯位如何達成。
- Head-centric 稀疏注意力屬近似方法，精度影響數字未在搜尋結果中取得。
- 1,200 行的實作以清晰為主，缺乏 vLLM/SGLang 級別的 paged KV、多卡並行等功能。

## 與其他論文的關係

- 與 **Fast-dLLM**（dual cache / block-wise KV refresh）的 Refresh / Reuse 兩相結構直接相關：dLLM-Serve 把 Fast-dLLM 的單請求快取機制推廣到多請求 serving，並針對其造成的資源震盪做排程。
- 與 **Sangam**（`sangam.md`）、**BlockServe**（`blockserve.md`）同屬 2025 底–2026 的 dLLM serving 系統；Sangam 關注 prefill/decode 干擾與 AR stack 重用，BlockServe 關注 block 粒度 continuous batching，dLLM-Serve 則聚焦記憶體足跡與 phase 排程，三者互補。
- Head-Centric Sparse Attention 延續 **Sparse-dLLM**、**dLLM-Cache** 等 KV 稀疏化工作，但重點放在「物理儲存布局」而非篩選準則。
- 後續的 characterization 論文 *Serving Masked Diffusion LLMs: Characterization and Design Principles from Real Hardware*（arXiv 2608.23807）引用並延伸其 logit 記憶體觀察。
