# TEAM: Temporal-Spatial Consistency Guided Expert Activation for MoE Diffusion Language Model Acceleration

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2602.08404`（ICML 2026） |
| 作者 / 單位 | Linye Wei, Zixiang Luo, Pingzhi Tang, Meng Li（北京大學 PKU-SEC-Lab） |
| 日期 | 2026-02 |
| 類別 | KV cache 與稀疏（MoE 專家啟用剪枝 + 平行解碼） |
| 連結 | [arXiv](https://arxiv.org/abs/2602.08404) · [GitHub](https://github.com/PKU-SEC-Lab/TEAM-MoE-dLLM) |

## 一句話總結

針對 MoE 型 dLLM「每步啟用大量專家、但最終只接受少數 token」的浪費，利用專家路由在時間（跨 denoising step）與空間（跨 token 位置）上的一致性，保守地只啟用必要專家、同時對熱門 token 做投機性多候選探索，以更少專家解出更多 token，在 SDAR-30B-A3B 上平均 1.94×、最高 2.2× 加速。

## 要解決的問題

- MoE dLLM 每個 denoising step 對所有 mask token 做 forward、每個 token 都要路由到 top-k 專家，但一步之後只有少數 token 被接受，大部分專家計算是白費。
- 專家啟用數量直接決定 MoE 的記憶體頻寬與延遲（尤其是 batch 小時），既有 dLLM cache 方法只處理 dense 模型的 KV / 特徵，未觸及專家路由。
- 需要 plug-and-play、不重訓的方案。

## 核心方法與特色

- **時間一致性（temporal consistency）**：同一 token 在相鄰 denoising step 的專家路由結果高度一致，因此已解碼 token 與大部分 mask token 可直接沿用上一步的路由或縮減候選專家。
- **空間一致性（spatial consistency）**：鄰近位置 token 傾向路由到相同專家，可用鄰居的路由結果推測 / 共享專家啟用，減少實際啟用的專家集合。
- **保守專家選擇（conservative activation）**：對已解碼 token 與尚不會被接受的 mask token，只啟用「必要」專家（依一致性推得），把平均啟用專家數降低 35–39%。
- **投機性探索（speculative exploration）for hot tokens**：對本步最可能被接受的熱門 token，反而擴大探索多個候選，使每次迭代接受的 token 數提升 1.49–1.74×；「少專家 + 多接受」同時縮短每步時間並減少步數。
- **三策略協同、plug-and-play**：三個互補策略可直接套在 SDAR 等 MoE dLLM 上，不需微調。
- **代價**：依賴路由一致性的假設，在路由本身不穩定的模型 / 早期 step 上可能失準；投機探索增加少量額外計算；目前僅在 SDAR 30B-A3B 驗證。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| SDAR-30B-A3B | 30B（啟用 3B）MoE，block diffusion | 主要（唯一）實驗模型 |
| LLaDA-MoE | 7B-A1B | 論文摘要與 README 未提及（未查到） |

## PPA / 效能數據

> 軟體論文：speedup、throughput (tok/s)、latency、記憶體、benchmark 準確度變化。GPU 型號未查到。

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| 平均加速 | 1.94× | vanilla MoE dLLM | SDAR-30B-A3B，GSM8K / MATH / HumanEval / MBPP |
| 最高加速 | 2.2× | vanilla | HumanEval |
| 啟用專家數減少 | 平均 35–39% | vanilla top-k 路由 | — |
| 每迭代接受 token 數 | 1.49–1.74× | vanilla | 熱門 token 投機探索 |
| 準確度 | 「negligible degradation」 | vanilla | 逐任務數字未查到 |

## 限制 / 備註

- 只驗證在 SDAR-30B-A3B；LLaDA-MoE、LLaDA2.0-mini 等其他 MoE dLLM 的結果未查到。
- 加速倍率（約 2×）低於 dense 模型的 cache 方法，但這是在 MoE 專家維度上的正交節省，可與 KV cache 疊加。
- README 標示為「initial implementation」，完整程式與逐任務表格未擷取到。

## 與其他論文的關係

- 「時間一致性」與 **dKV-Cache**、**DyLLM**、**Sparse-dLLM** 對 KV / 注意力的時間穩定觀察是同一現象在 MoE 路由上的版本。
- 「熱門 token 多候選投機探索」延伸 **Fast-dLLM** 的信心閾值平行解碼與投機解碼思路。
- 與 dMoE（learnable block experts）、Saliency-Harnessing Accurate Routing for Diffusion MoE 等後續 MoE dLLM 工作互為競爭；與 SDAR / LLaDA-MoE 的模型工作為基礎。
- 可與 **Fast-dLLM**、**dLLM-Cache** 的 KV / 特徵快取疊加，是 MoE dLLM 專屬的額外加速維度。
