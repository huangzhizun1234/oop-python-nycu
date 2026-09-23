# Hardware Acceleration of Block-Diffusion LLM for Edge Devices

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2609.01084` |
| 作者 / 單位 | Wei-Hsing Huang, Kiseok Lee, Ming-Yen Lee, Weiyu Sun, Cheng-Jhih Shih, Gayatri Tanksali, Arpit Khandelwal, Pin-Jun Chen, Yingyan Celine Lin, Shimeng Yu（Georgia Tech（依作者群推測，待確認）） |
| 日期 | 2026-09 |
| 類別 | 硬體加速器 |
| 連結 | [arXiv](https://arxiv.org/abs/2609.01084) · GitHub：未查到 |

## 一句話總結

針對 batch-one 的 block-diffusion LLM（Fast-dLLM v2 類）邊緣推論，共同設計「寬 I/O、可帶精度標籤讀取的 LPDDR 記憶體系統 (WIFiV-LPDDR)」、「低秩 + INT8 殘差、依 query 決定每筆精度的 prefix KV (BRQ-KV)」與「依飄移程度決定用 canonical 權重 / 低位元 delta / 直接沿用快取狀態的 FFN (DAT-FFN)」，映射到 input-stationary 混合精度 systolic array，在模擬的 Jetson-class 平台上 1.5B/7B 模型平均省能 3.79x/3.96x、延遲加速 2.88x/4.44x，benchmark 掉分 <1 個百分點。

## 要解決的問題

- 邊緣裝置是單流 (batch-one) 推論：權重從 DRAM 讀進來只服務一個請求，無法像資料中心用 batch 攤平權重流量，因此 FFN 權重與 prefix KV 的記憶體流量直接決定延遲與能耗。
- 全注意力 dLLM（LLaDA 類）每個去噪步都重算整條序列；原生 block diffusion（Fast-dLLM v2、SDAR）讓完成的 block 不可變、可精確快取，但每個 block 內的多步精煉 (refinement) 仍然每步都要串流讀取 prefix KV 與全部 FFN 權重。
- 目前的 LPDDR 介面與加速器不支援「同一份資料依需求以不同精度讀取」，也不會利用「精煉步驟間活化值變化很小」這個時間局部性。

## 核心方法與特色

- **WIFiV-LPDDR（Wide-I/O, precision-tagged LPDDR）**：把 LPDDR 通道加寬、並讓每次讀取帶精度標籤，使同一筆 KV/權重可以只讀取需要的位元平面（例如只讀低秩部分、或低秩 + INT8 殘差），讓 BRQ-KV / DAT-FFN 的「每筆資料不同精度」真的能轉成減少的 DRAM 流量，而不是讀滿再丟棄。
- **BRQ-KV（Bit-Reversible / 可回復精度的 prefix KV）**：prefix KV 存成「canonical 低秩分量 + INT8 殘差」；attention 時依當前 query 決定每個 KV entry 要用低秩近似還是低秩 + 殘差的全精度。精度分層是可逆的（reversible tiers），不需要保存多份副本。代價：需要在 attention 前多一個 query-dependent 的精度決策與低秩重建。
- **DAT-FFN（Drift-Adaptive Tiered FFN）**：追蹤精煉步驟間 FFN 輸入的「飄移 (drift)」，把 FFN 計算分成三檔：飄移大 → 用 canonical（完整）權重重算；飄移中 → 用相鄰步驟校正過的低位元 delta；飄移小 → 直接沿用快取的 FFN 輸出（cached-state carry）。整個過程活化值維持不量化，只在權重/delta 上降位元，並用「correction ladder」逐級修正。省下的是 FFN 權重流量與 MAC 數，代價是需要 drift 偵測與快取狀態的 SRAM。
- **Input-stationary 混合精度 systolic array**：BRQ-KV 與 DAT-FFN 都映射到同一個 input-stationary 陣列，讓 INT8 殘差、低位元 delta 與全精度 canonical 路徑共用 PE，避免為每種精度各做一套資料通路。
- **端到端、可調的品質-效率設定**：DAT-FFN 的門檻依 benchmark 各自設定（per-benchmark settings），論文主張在所報告設定下各 benchmark 掉分低於 1 個絕對百分點。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| Fast-dLLM v2 1.5B | 1.5B | 原生 block-diffusion（由 Qwen2.5-1.5B-Instruct 改造）；1.5B 數字對應此模型 |
| Fast-dLLM v2 7B | 7B | 由 Qwen2.5-7B-Instruct 改造；7B 數字對應此模型 |

（模型名稱依搜尋摘要「uses Fast-dLLM v2 as the baseline algorithm」推定，待全文確認。）

## PPA / 效能數據

> 硬體論文：製程、面積、功耗、頻率、能效 (TOPS/W)、對比 GPU 的 speedup / energy。

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| 能耗降低（算術平均） | 3.79x（1.5B）/ 3.96x（7B） | 模擬的 Jetson-class 平台上、未套用本文技術的 block-diffusion 推論 | full stack（WIFiV-LPDDR + BRQ-KV + DAT-FFN），batch-one，所報告 DAT-FFN 設定 |
| 延遲加速（算術平均） | 2.88x（1.5B）/ 4.44x（7B） | 同上 | 同上 |
| Benchmark 掉分 | <1 個絕對百分點 | 全精度 Fast-dLLM v2 | per-benchmark DAT-FFN 設定 |
| 製程 / 面積 / 功耗 / 頻率 | 論文未在公開摘要提供 / 未查到 | — | 標題強調 "modeled Jetson-class platforms"，偏向系統級模擬而非流片 |
| Systolic array 規模、LPDDR 規格 | 未查到 | — | — |

## 限制 / 備註

- 結果來自「modeled Jetson-class platforms」，是模型化的效能/能耗估計，不是實體晶片或 FPGA 量測；製程、面積、頻率等 PPA 細節在摘要中沒有，本次無法取得全文。
- 只針對「原生 block diffusion」（完成 block 不可變）的模型；LLaDA/Dream 這類全注意力、會回頭修改 token 的 dLLM 不能直接套用 BRQ-KV 的 prefix 不變假設。
- DAT-FFN 的三檔門檻是 per-benchmark 調的，實際部署時需要一個泛用的門檻策略。
- 對比基準是同平台上的未優化 block-diffusion 推論，不是對商用 GPU（Orin NX 實機）的直接量測比較。

## 與其他論文的關係

- 直接建立在 **Fast-dLLM v2**（block-diffusion + block-wise KV cache）之上，把演算法層的「完成 block 精確快取」轉化為硬體層的記憶體流量優化。
- 與 **DART / npu-dllm-sampling.md** 互補：DART 針對取樣階段與桌面/伺服器 GPU 比較；本文針對邊緣 batch-one 的 DRAM 頻寬瓶頸（KV + FFN 權重流量）。
- DAT-FFN 的「精煉步驟間活化飄移小 → 重用」與軟體側的 dLLM-Cache、D2 Cache 等 cross-step 特徵重用是同一觀察，本文把它落實到權重讀取精度與 systolic array 資料通路。
- 與 **DiPe (dipe-edge.md)** 同樣鎖定 Jetson Orin 級邊緣裝置，但 DiPe 是在現成 GPU 上做量化 + 跨步聚合，本文則是提出新的記憶體系統與加速器。
