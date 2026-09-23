# Sparse-dLLM: Accelerating Diffusion LLMs with Dynamic Cache Eviction

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2508.02558`（AAAI 2026） |
| 作者 / 單位 | Yuerong Song, Xiaoran Liu, Ruixiao Li, Zhigeng Liu, Zengfeng Huang, Qipeng Guo, Ziwei He, Xipeng Qiu（復旦大學 / 上海創智學院 / 上海 AI Lab，OpenMOSS） |
| 日期 | 2025-08 |
| 類別 | KV cache 與稀疏（cache eviction） |
| 連結 | [arXiv](https://arxiv.org/abs/2508.02558) · [GitHub](https://github.com/OpenMOSS/Sparse-dLLM) |

## 一句話總結

利用 dLLM 注意力「跨層、跨步都稀疏且重要 token 穩定」的特性，在 delayed KV cache 上加入雙向（prefix + suffix）動態淘汰，只保留高注意力 token 的 KV，使吞吐最高提升 10× 而峰值記憶體與原模型相當。

## 要解決的問題

- dLLM 每步對整段序列做雙向注意力，計算量 O(L²)；長上下文時延遲與記憶體都不可接受。
- 既有 cache 方法（Fast-dLLM、dKV-Cache、dLLM-Cache）把整層 KV / 特徵存下來，記憶體佔用反而比原模型高，限制長上下文應用。
- AR 模型的 KV eviction（H2O、SnapKV 等）假設單向注意力與固定的 query，無法直接搬到雙向、每步 query 都變的 dLLM。

## 核心方法與特色

- **關鍵觀察：持續的跨層稀疏與時間穩定性**。分析 LLaDA / Dream 的注意力圖發現只有少數 pivotal token（含 attention sink）拿到大部分注意力，而且這些 token 在整個 decoding 過程中持續重要、低重要 token 持續不重要——因此可以放心「淘汰」而不是只「跳過」。
- **雙向動態 cache eviction**：對 prefix（prompt + 已解碼）與 suffix（尚未解碼的 mask token）兩側同時做淘汰。以注意力分數（經 kernel_size 的局部平滑）估計 token 顯著性，只保留 keep_ratio（預設 0.5）的 KV 進 cache，其餘丟棄；剩下的 token 用稀疏注意力計算。
- **Delayed bidirectional sparse caching**：cache 更新刻意延後一步，避免 token 剛被解碼時表徵不穩定造成錯誤淘汰（承襲 dKV-Cache 的 delayed 思想並擴展到淘汰決策）。
- **加速與省記憶體同時達成**：淘汰後注意力計算與 KV 儲存量都按 keep_ratio 縮減，所以峰值記憶體接近無 cache 的原模型，而不像 dLLM-Cache 需要額外儲存全部特徵。
- **Training-free、block-wise 相容**：與 block 解碼（block_length 32）及 Fast-dLLM 式 prefix cache 相容，可疊加。
- **代價**：被淘汰的 token 之後不會再被看到，若淘汰錯誤無法恢復；keep_ratio 與 kernel_size 需依模型調整。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| LLaDA-8B-Instruct（LLaDA-Chat） | 8B | 主要實驗、LongBench |
| LLaDA-1.5 | 8B | — |
| Dream-v0-Base-7B / Dream-v0-Instruct-7B | 7B | — |

## PPA / 效能數據

> 軟體論文：speedup、throughput (tok/s)、latency、記憶體、benchmark 準確度變化。GPU 型號未查到。

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| 最高吞吐提升 | 最高 10× | 原始 LLaDA / Dream | 依 benchmark 與生成長度 |
| LLaDA-8B-Instruct GSM8K | 4.57 → 26.45 TPS（5.8×） | 原始 LLaDA | GSM8K |
| 峰值記憶體 | 與原始模型相當 | 原始模型 / 其他 cache 方法 | — |
| LongBench（4k 輸入截斷） | 34.46 vs 34.55 | 原始 LLaDA-8B-Instruct | block 32、steps = gen length = 512 |
| GSM8K 4-shot vs 保留比例 | 77.03–78.01%（不同 r / kernel size） | — | LLaDA-8B-Instruct |
| 預設超參數 | kernel_size 3、keep_ratio 0.5、block_length 32 | — | repo 範例指令 |

## 限制 / 備註

- 逐 benchmark 的完整表格在 README 以圖片呈現，文字未擷取到；GPU 型號未查到。
- 淘汰是不可逆的，理論上對需要回看遠端 prompt 的任務有風險，但 LongBench 結果顯示影響很小。
- 「10×」為最佳條件（長序列）；GSM8K 這類中等長度為 5–6×。

## 與其他論文的關係

- 延伸 **dKV-Cache** 的 delayed caching，並加入類似 AR 模型 H2O / SnapKV 的注意力導向淘汰，是第一個把 cache eviction 與稀疏注意力結合到 dLLM 的工作。
- 與 **dLLM-Cache / Fast-dLLM** 相比訴求「省記憶體」；與 **SparseD**（預先算好每個 head 的稀疏 pattern）和 **Focus-dLLM**（以信心預測 unmask 區域 + sink-aware pruning）同屬稀疏注意力路線，後兩者在長上下文（32K–64K）加速更高。
- **DPad / Streaming-dLLM** 從「丟掉遠端 suffix」的角度做類似的 suffix 稀疏化，但用位置距離而非注意力分數。
- 後續 Dynamic-dLLM、WaveFilter、Fine-Grained Cache Eviction（ACL 2026 Findings）等工作把 Sparse-dLLM 當 baseline。
