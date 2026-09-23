# Serving Masked Diffusion LLMs: Characterization and Design Principles from Real Hardware

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2608.23807` |
| 作者 / 單位 | Farhana Amin, Sabiha Afroz, Mona Moghadampanah, Dimitrios S. Nikolopoulos（Virginia Tech） |
| 日期 | 2026-08 |
| 類別 | 硬體特性分析 / 系統與 serving |
| 連結 | [arXiv](https://arxiv.org/abs/2608.23807) · GitHub：未查到 |

## 一句話總結

第一篇在真實 GPU（單張 H200）上、以並發 serving 負載量測 masked dLLM（LLaDA-8B-Instruct + D2F LoRA）的特性研究：發現瓶頸不在 GPU（單請求只有 24.4% 時間是 kernel 執行、75.6% 是 CPU 側逐 block 控制流），請求難度離散地落在 11 個固定步數層級且無法在進入時預測，而「同步批次、每個去噪步共用一次 forward」在 batch 16 時可達逐請求派發的 16.0x 吞吐。

## 要解決的問題

- 近期的 dLLM serving 系統（Sangam、HERALD、Optimus、dInfer 等）多直接沿用 AR serving 的假設（continuous batching、依 token 進度做 admission/eviction、預測請求長度），但沒有人先在真實硬體上量測 dLLM 在並發負載下的行為。
- dLLM 的執行單位是「去噪步」而非「token」，且 D2F 這類 block-wise pipeline 解碼讓每個請求的步數依內容而異；若 serving 系統照 AR 方式設計，可能把不成立的假設帶進來。

## 核心方法與特色

- **量測設定**：LLaDA-8B-Instruct 加 D2F（Discrete Diffusion Forcing）LoRA adapter，讓模型能做 block-wise、可提前終止的並行解碼；單張 NVIDIA H200；資料集 GSM8K 與 HumanEval；分析 tier 結構、可預測性、批次與排程。
- **發現 1：瓶頸在 CPU 而非 GPU**：單請求 wall-clock 中只有 24.4% 是 GPU kernel 執行，75.6% 是 CPU 端逐 block 的控制流（信心門檻判斷、mask 更新、block 推進、kernel launch）。這與 DART 在硬體層觀察到的「取樣非 GEMM 階段佔 70%」是同一現象的軟體端表現。
- **發現 2：請求難度是離散的且不可預測**：固定解碼門檻下，請求需要的去噪步數只會落在 11 個值：178 + 29k（178、207、236 … 468）；所有測試過的 admission-time 信號（prompt 長度等）都預測不了層級（最佳 R² = 0.150）。因此 AR 式「預測長度再排程」在 dLLM 上不成立。
- **發現 3：逐請求派發的代價隨 batch 線性放大**：相對於同步批次，逐請求派發的 overhead 在 batch 2 / 4 / 8 分別為 2.4x / 5.1x / 8.1x；同步批次（把 batch 內請求 pad 到相同形狀、每個去噪步跑一次共用 forward、之後各請求獨立取樣）在 batch 16 達到 16.0x 吞吐。
- **發現 4：短 benchmark 會掩蓋變異**：生成預算 <320 tokens 的 benchmark 會把請求在延遲分佈拉開之前就截斷，低估 serving 的尾延遲變異。
- **設計原則**：(a) 以「去噪步」為並行單位，同步批次共用 forward；(b) admission/eviction 必須與已共享的 forward pass 互動設計，而不是 AR 式逐 token 進出；(c) 不要依賴 admission-time 的長度/難度預測；(d) 減少 CPU 端 per-block 控制流（融合、CUDA graph、GPU 端取樣）是首要優化。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| LLaDA-8B-Instruct + D2F LoRA | 8B | D2F 讓 LLaDA 做 block-wise pipeline 並行解碼；所有量測皆此模型 |

## PPA / 效能數據

> 特性分析論文：GPU 利用率、吞吐、延遲分佈；無新硬體 PPA。

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| GPU kernel 時間佔比 | 24.4%（CPU 側控制流 75.6%） | 單請求 wall-clock | LLaDA-8B-Instruct + D2F，H200，batch 1 |
| 步數層級 | 11 個離散值：178 + 29k（178 … 468） | — | 固定解碼門檻，GSM8K |
| 難度可預測性 | 最佳 R² = 0.150 | admission-time 信號 | 無信號可用 |
| 逐請求派發 overhead | 2.4x / 5.1x / 8.1x | 同步批次 | batch 2 / 4 / 8 |
| 同步批次吞吐 | 16.0x | 逐請求派發 | batch 16 |
| 短 benchmark 偏差 | 生成預算 <320 tokens 會低估延遲變異 | — | — |
| 絕對 tok/s、延遲 | 未查到 | — | — |

## 限制 / 備註

- 只有一個模型（LLaDA-8B + D2F）、一張 GPU（H200）、兩個資料集；LLaDA-2.0、Dream、SDAR 等其他 dLLM 的 tier 結構可能不同（11 個層級與 29 的步距顯然來自 D2F 的 block 大小 / 門檻配置）。
- 「CPU 瓶頸 75.6%」是在 batch 1、未做 kernel 融合的實作上量到；有 CUDA graph / GPU 端取樣的實作（dInfer、Fast-dLLM v2 kernel）數字會不同。
- 沒有提出新系統，只給設計原則；後續 serving 系統可據此驗證。

## 與其他論文的關係

- 對 **Sangam**（用 AR stack 服務 dLLM）、**HERALD**、**Optimus**、**dInfer** 等 serving 系統是「先量測再設計」的反思：指出 AR 式 admission/長度預測不適用。
- 與 **DART (npu-dllm-sampling.md)** 互相印證：DART 在硬體層說取樣（非 GEMM）佔 70% 延遲，本文在軟體層說 CPU 控制流佔 75.6%；兩者都主張 dLLM 的瓶頸在 GEMM 之外。
- 建立在 **D2F**（Discrete Diffusion Forcing）之上；tier 離散化直接來自 D2F 的 block-wise 解碼結構。
- 對硬體加速器設計的啟示：加速器（如 DART、block-diffusion-edge-hw）應把 per-block 控制流（信心門檻、mask 更新、block 推進）搬進硬體或 GPU 端，否則 GEMM 加速後 CPU 端會成為新瓶頸。
