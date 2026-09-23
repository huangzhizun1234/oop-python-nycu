# dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive Caching

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2506.06295`（ICML 2026） |
| 作者 / 單位 | Zhiyuan Liu, Yicun Yang, Dongdong Zhang, Junjie Chen, Yuxiang Zou, Zhuang Wei, Ye Wang, Linfeng Zhang（上海交通大學 EPIC Lab 等） |
| 日期 | 2025-06 |
| 類別 | KV cache 與稀疏（特徵快取） |
| 連結 | [arXiv](https://arxiv.org/abs/2506.06295) · [GitHub](https://github.com/maomaocun/dLLM-Cache) |

## 一句話總結

把 dLLM 推論拆成「靜態 prompt」與「部分動態 response」，對 prompt 特徵用長間隔快取、對 response 用短間隔快取＋以 Value 向量相似度（V-verify）挑出少數需要重算的 token，達到最高 9.1× 加速且多數任務無損。

## 要解決的問題

- dLLM 的雙向注意力讓 AR 式 KV cache 不能直接用，每個 denoising step 都要對整段序列做完整 forward。
- 但實際上 prompt 部分完全不變，response 部分在相鄰 step 間也只有少數 token 的特徵有明顯變化，全部重算是巨大的浪費。
- 現有方法（同期 Fast-dLLM / dKV-Cache）只針對 KV，且以固定 block / 固定週期刷新，缺乏對「哪些 token 真的變了」的細粒度判斷。

## 核心方法與特色

- **Prompt cache（長間隔）**：prompt token 的 K、V、attention output、FFN output 等中間特徵在整個生成過程中幾乎不變，因此每隔 Kp 步（長間隔，例如數十到上百步）才完整重算一次，其餘步驟直接沿用。
- **Response cache（短間隔 + 部分更新）**：response token 的特徵每隔 Kr 步（短間隔）完整刷新一次；在兩次刷新之間，只挑出「變化最大的 ρ 比例 token」重算，其餘沿用上一步快取。
- **V-verify（以 Value 相似度選 token）**：對每個 response token，比較當前步驟與快取中 Value 向量的 cosine similarity，相似度最低的前 ρ（約 25%）token 被視為「動態 token」需重算。作者實驗顯示 value-based 選擇在 ρ≈0.25 時準確度最高、且明顯優於隨機選擇；cosine similarity 版本在 GSM8K 達 78.54%。
- **快取層級是「特徵」不只是 KV**：連 attention output 與 FFN output 都快取，因此省下的不只 attention，還包括 FFN 的計算（FLOPs 減少直接反映在延遲上）。
- **Training-free、超參數可依任務調整**：Kp、Kr 不需搜尋，準確度敏感任務（程式碼、數學）用小間隔，延遲優先任務用大間隔；長 prompt 任務（LongBench）加速最大。
- **代價**：需要額外儲存多層中間特徵，記憶體開銷比純 KV cache 高；ρ 固定意味著每步重算 token 數不隨實際變化量調整（後來 DyLLM、Elastic-Cache 針對此點改進）。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| LLaDA-8B（Base / Instruct） | 8B | 主要實驗 |
| Dream-7B | 7B | 第二個驗證模型 |
| LLaDA-V、MMaDA | 8B | repo 提供多模態 dLLM 支援 |

## PPA / 效能數據

> 軟體論文：speedup、throughput (tok/s)、latency、記憶體、benchmark 準確度變化。GPU 型號未查到（論文附錄有列，摘要未提）。

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| 最高加速 / FLOPs 減少 | 9.1× | 原始 dLLM pipeline | LLaDA 8B, LongBench-HotpotQA（長 prompt） |
| GSM8K 加速 | 2.8×，準確度 70.66% | 原始實作 | 依搜尋摘要，模型 / shot 設定未確認 |
| V-verify（cosine）GSM8K | 78.54% | 其他相似度指標 / 隨機 | LLaDA 8B |
| 最佳更新比例 ρ | ≈0.25 | 全部重算 / 隨機選 | value-based selection |
| 品質 | 多數任務無損，接近 ARM 推論延遲 | 原始 dLLM | 多 benchmark |

## 限制 / 備註

- 加速幅度與 prompt 長度強相關：短 prompt、長生成的情境加速較小（約 2–3×），長 prompt 才有 9×。
- Kp、Kr、ρ 為固定超參數，跨模型 / 資料集需調整（DyLLM 論文即以此為批評點）。
- 快取多層特徵的記憶體開銷在長上下文時可能成為瓶頸，Sparse-dLLM 之類的 cache eviction 工作正是針對此問題。
- 已有 Ascend NPU 移植版（dLLM-cache-Ascend）。

## 與其他論文的關係

- 與 **Fast-dLLM**（block-wise KV cache）、**dKV-Cache**（delayed KV cache）並列 2025 年中三大 dLLM cache 方法，差異在 dLLM-Cache 快取的是整層特徵並用相似度做細粒度部分更新。
- **d²Cache**、**Elastic-Cache**、**DyLLM**、**ES-dLLM** 都以 dLLM-Cache 為主要 baseline，分別以確定性先驗、注意力漂移、saliency、early-skip 取代固定 ρ / 固定間隔的設計。
- **Sparse-dLLM**、**Focus-dLLM** 補足其在長上下文的記憶體與注意力計算問題（cache eviction / 稀疏注意力）。
- 同團隊後續有 dLLM 系列 toolkit 與量化工作，dLLM-Cache 為其推論加速基礎元件。
