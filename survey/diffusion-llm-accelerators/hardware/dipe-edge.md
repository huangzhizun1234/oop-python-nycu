# DiPe: Real-Time Diffusion LLM Inference on Edge Devices for Planning Tasks（Demo Abstract）

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | 無 arXiv；SenSys-Adjunct 2026（ACM/IEEE International Conference on Embedded AI and Sensing Systems, Posters & Demos）Demo Abstract |
| 作者 / 單位 | Yejia Liu, Xiaomin Ouyang（The Hong Kong University of Science and Technology, HKUST） |
| 日期 | 2026（SenSys 2026） |
| 類別 | 硬體加速器 / 邊緣部署（商用邊緣 GPU 上的系統示範） |
| 連結 | [HKUST Research Portal](https://researchportal.hkust.edu.hk/en/publications/demo-abstract-dipe-real-time-diffusion-llm-inference-on-edge-devi/) · [SenSys 2026 Accepted Demos](https://sensys.acm.org/2026/accepted_demos.html) · GitHub：未查到 |

## 一句話總結

DiPe 是一個在 NVIDIA Jetson Orin NX 上即時執行 diffusion LLM 做規劃任務（示範為旅遊規劃）的系統，結合模型量化與「跨去噪步的資訊聚合 (cross-step information aggregation)」在維持精度下取得顯著加速；為 2 頁 demo abstract，公開摘要未給具體數字。

## 要解決的問題

- dLLM 對規劃 (planning) 任務有結構性優勢：並行生成 token、雙向注意力能同時考量整份計畫的前後約束（時間、地點、預算），比 AR 模型更容易產生全域一致的計畫。
- 但 dLLM 的迭代去噪讓延遲很高，在 Jetson Orin NX 這種記憶體（8/16 GB）與算力受限的邊緣裝置上難以即時；每一步都要對整段序列做完整 forward，權重流量與步數同時是瓶頸。

## 核心方法與特色

- **模型量化**：以低位元權重（具體位元數未查到，Orin NX 記憶體限制下推測為 4-bit 或 8-bit）降低權重流量與記憶體佔用，使 dLLM 能放進 Orin NX 並提高每步吞吐。
- **Cross-step information aggregation（跨步資訊聚合）**：利用相鄰去噪步之間 token 預測 / 中間特徵的高度相似性，把多步的資訊聚合起來以減少實際需要執行的去噪步數或每步的計算量（機制類似 dLLM-Cache / 特徵重用與早期確定 token 的合併），在維持規劃準確度下獲得加速。具體演算法細節在 demo abstract 摘要中未描述。
- **即時互動示範**：在 Jetson Orin NX 上做旅遊規劃，與會者可即時輸入需求並看到裝置端 dLLM 生成計畫。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| 未查到 | — | 公開摘要未指明；Orin NX 記憶體限制下推測為 LLaDA-8B 量化版或更小的 dLLM（待確認） |

## PPA / 效能數據

> 邊緣系統示範：speedup、latency、精度；無新硬體 PPA。

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| Speedup | 「significant speedup」；數值未查到 | 未量化 / 未聚合的 dLLM 推論 | Jetson Orin NX |
| 精度 | 「maintaining accuracy」；數值未查到 | — | 規劃任務 |
| 延遲 / tok/s / 功耗 | 未查到 | — | — |
| 平台 | NVIDIA Jetson Orin NX | — | 商用邊緣 GPU（Ampere，8/16 GB） |

## 限制 / 備註

- 只是 demo abstract（通常 2 頁），公開摘要不含任何數字、模型名、量化位元；本檔僅能描述方法方向。HKUST research portal 與 ACM DL 皆被 proxy 擋住，全文未能取得。
- 不是新硬體，而是在現成 Jetson GPU 上的量化 + 演算法加速；歸在「硬體」類是因為它提供了 Orin NX 級邊緣裝置上 dLLM 可行性的實證。
- 「cross-step aggregation」與既有的 cross-step 快取（dLLM-Cache、D2 Cache）、DiPe 自身可能的多步合併之間的差異待全文確認。

## 與其他論文的關係

- 與 **block-diffusion-edge-hw.md** 同樣鎖定 Jetson Orin 級平台：後者提出新的記憶體系統與加速器（模擬），DiPe 則在實機 Orin NX GPU 上用軟體手段達成即時。
- 與 **mobile-npu-llada-cpp.md** 同屬「dLLM 邊緣部署實證」：一個在手機 Hexagon NPU、一個在 Jetson GPU。
- 「跨步資訊聚合」在概念上與軟體側的 **dLLM-Cache**、**Fast-dLLM** 的並行解碼 / KV cache 重用相通，可視為把這些技術落到規劃任務與邊緣硬體的應用案例。
- 應用面與 dLLM 用於規劃/agent 任務的研究（如 "The Bitter Lesson of Diffusion Language Models for Agentic Workflows"）相關，DiPe 提供正面案例。
