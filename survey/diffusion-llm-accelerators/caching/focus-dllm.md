# Focus-dLLM: Accelerating Long-Context Diffusion LLM Inference via Confidence-Guided Context Focusing

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2602.02159`（ACL 2026 Long） |
| 作者 / 單位 | Lingkun Long, Yushi Huang, Shihao Bai, Ruihao Gong, Jun Zhang, Ao Zhou, Jianlei Yang（北京航空航天大學 / HKUST / SenseTime Research） |
| 日期 | 2026-02 |
| 類別 | KV cache 與稀疏（稀疏注意力） |
| 連結 | [arXiv](https://arxiv.org/abs/2602.02159) · [GitHub](https://github.com/Longxmas/Focus-dLLM) · [ACL Anthology](https://aclanthology.org/2026.acl-long.556/) |

## 一句話總結

利用「相鄰 denoising step 的 token 信心高度相關」預測下一步會被解碼的位置，只為這些位置估計注意力重要性並做 sink-aware 剪枝，在 32K 上下文下達到 29.6× 無損加速。

## 要解決的問題

- 長上下文 dLLM 推論的成本由雙向全注意力主導；要做稀疏注意力必須知道「哪些 query 重要」，但 dLLM 每步要解哪些 token 事先未知，無法像 AR 那樣只為當前 query 估計。
- 直接對所有 mask token 估計注意力重要性的成本本身就接近全注意力。
- 粗暴剪枝會誤刪 attention sink（少數被所有 token 高度關注的位置），導致品質崩潰。

## 核心方法與特色

- **Past confidence-guided indicator**：實證發現 token 信心在相鄰 step 間有強正相關，因此用上一步（t−1）的信心分數預測本步（t）最可能被 unmask 的位置，並向兩側擴展一個小視窗以維持語意連貫。這讓「重要 query」可以在本步 forward 之前就決定。
- **Sink-aware pruning**：動態辨識 attention sink，並利用「跨層 sink 位置一致」的特性把辨識成本分攤；剪枝時保留 sink 與被預測 unmask 區域高度關注的 prompt block，其餘 prompt token 的 K/V 以 block 為單位剪掉。
- **只針對 prompt 側稀疏化**：長上下文情境下成本幾乎全在 prompt 的 K/V，Focus-dLLM 把注意力計算縮到「預測 unmask 區域 × 保留的 prompt block」，因此加速隨上下文長度線性放大。
- **Training-free、與 KV cache / block 解碼相容**：不改權重，可與 Fast-dLLM 式 prefix cache 疊加。
- **代價**：依賴信心的時間連續性，在信心跳動大的早期 step 預測可能失準；需要額外維護 sink 與信心統計；對短上下文任務收益小。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| LLaDA-8B-Instruct | 8B | LongBench |
| UltraLLaDA | 8B | 長上下文延伸版 LLaDA，Focus-dLLM 在其上取得最高平均分 |
| Dream-7B | 7B | — |

## PPA / 效能數據

> 軟體論文：speedup、throughput (tok/s)、latency、記憶體、benchmark 準確度變化。GPU 型號未查到。

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| 最高加速 | 29.6× | 原始（全注意力）dLLM | 32K 上下文，「lossless」 |
| 加速趨勢 | 隨上下文長度增加而顯著放大 | — | LongBench 各長度 |
| LongBench 平均分 | UltraLLaDA 上為所有方法最高（優於 vanilla 與其他加速框架） | vanilla / 既有加速方法 | — |
| 逐任務分數 | 論文未提供於 README / 未查到 | — | — |

## 限制 / 備註

- 具體 LongBench 各子任務分數、GPU 型號與 32K 以外長度的加速倍率未從可用來源取得。
- 只稀疏化 prompt 側注意力，對 response 內部的冗餘（suffix mask、已解碼 token 重算）沒有處理，需搭配其他方法。
- 「29.6×」是相對無 cache 的原始實作；相對已有 prefix cache 的 baseline 增益會小很多。

## 與其他論文的關係

- 與 **SparseD**（預先計算 head-specific pattern 並重用）和 **Sparse-dLLM**（注意力導向 KV 淘汰）同屬 dLLM 稀疏注意力路線，Focus-dLLM 的差異是「先預測 query 再估計重要性」。
- **Prefilling-dLLM** 以 chunk 級 top-K 選擇長 prompt，思路接近但粒度更粗，且做 prefill/decode 分離。
- 「信心在相鄰 step 高度相關」的觀察也被 **ES-dLLM**（用前一輪信心估 token 重要性）與 **DyLLM** 利用。
- 可與 **Fast-dLLM**、**dLLM-Cache** 的 prefix cache 疊加作為長上下文推論的完整方案。
