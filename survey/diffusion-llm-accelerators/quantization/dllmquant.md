# DLLMQuant: Quantizing Diffusion-based Large Language Models

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2508.14090`（ICML 2026 poster，會議版標題 "DLLMQuant: A Post-Training Quantization Framework Tailored for Diffusion-Based Large Language Models"） |
| 作者 / 單位 | Chen Xu, Dawei Yang（Houmo AI，依記憶，待確認） |
| 日期 | 2025-08 |
| 類別 | 量化 |
| 連結 | [arXiv](https://arxiv.org/abs/2508.14090) · [ICML 2026](https://icml.cc/virtual/2026/poster/62533) · GitHub：未查到官方 repo |

## 一句話總結

第一個專為 dLLM 設計的 PTQ 框架：用 TMAS 依 timestep 與 mask 比例抽樣校準資料、用 IA-AQ 依 bidirectional attention 互動強度分配激活量化資源、用 CGQ 依 mask 狀態與 token 置信度加權 weight 誤差補償，在 4-bit 下讓 LLaDA 的 GSM8K 回升 10 分以上。

## 要解決的問題

- 直接把 AR LLM 的 PTQ 套到 dLLM 會嚴重掉分：AWQ 在 LLaDA W4A4 下準確度掉 16%。
- 作者歸納三個 dLLM 專屬的難點：
  1. **迭代生成 + 動態 mask 比例**：不同 decoding step 的 token 分佈差異很大，既有 PTQ 校準只用「全可見」的乾淨文字，無法涵蓋這些分佈。
  2. **誤差在迭代中累積放大**：每一步的量化誤差會透過後續 denoising 被放大。
  3. **bidirectional attention 與 mask token**：masked / unmasked token 對輸出的重要性不同，均一量化會浪費精度在不重要的位置。

## 核心方法與特色

- **TMAS（Temporal-Mask Adaptive Sampling）校準抽樣**：不用固定的乾淨文字校準，而是同時考慮「時間步」與「mask 比例」兩個因子，從整個 denoising 軌跡中抽樣激活，讓校準集覆蓋不同 timestep 的分佈，直接對症問題 1。
- **IA-AQ（Interaction-Aware Activation Quantization）**：利用 dLLM bidirectional attention 的互動訊號（attention score）辨識哪些 token 對其他 token 影響大，量化時優先保護這些 token（動態分配量化資源），減少高影響 token 的激活誤差。
- **CGQ（Certainty-Guided Quantization）**：在 GPTQ 類的 Hessian 誤差補償中，把「token 是否為 mask」與「token 置信度分數」納入加權，讓 weight 量化的誤差最小化目標對齊 dLLM 真正重要的位置（高置信、將被 unmask 的 token），而不是對所有位置一視同仁。
- **可疊加在既有 PTQ 之上**：三個模組可以插到 GPTQ / AWQ / QuaRot / DuQuant 等現有流程中，以「原方法 vs. 原方法 + DLLMQuant」形式報告增益。
- **代價**：TMAS 需要跑多步 denoising 來收集校準激活，校準成本高於 AR PTQ；IA-AQ 需要 attention 訊號，屬 dynamic 量化，會增加少量 runtime 開銷。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| LLaDA-8B（Base / Instruct） | 8B | 主要模型；報告 GSM8K 提升 >10 分 |
| LLaDA-1.5-8B | 8B | |
| Dream-7B | 7B | |

## PPA / 效能數據

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| AWQ 直接套用之掉分 | −16% 準確度 | FP16 | LLaDA，W4A4（動機實驗） |
| GSM8K 提升 | > +10 分 | 原 PTQ 方法（未加 DLLMQuant） | LLaDA，4-bit |
| 九項任務平均 | 平均 +2% | 原 PTQ 方法 | 3 個 dLLM（LLaDA-8B / LLaDA-1.5 / Dream-7B） |
| 生成速度 | 1.6×–1.7× | FP16 | 4-bit 實作，token 生成 |
| 記憶體 | 3.2× 降低（LLaDA-8B 約 16 GB → 約 5 GB） | FP16 | 4-bit |
| HumanEval / GSM8K 推理能力 | QuaRot 量化後推理能力退化；DLLMQuant 可維持接近 FP | QuaRot | 生成型任務 |
| 各 benchmark 具體分數 | 論文未於可存取來源提供 / 未查到 | — | — |

## 限制 / 備註

- 校準流程需模擬多步 denoising，校準時間長於 AR PTQ（具體時間論文未於可存取來源提供）。
- IA-AQ 依賴 attention 訊號做動態分配，在真正的低 bit kernel 上實作與加速能否兼得需要驗證；1.6×–1.7× 的 speedup 條件（GPU 型號、batch）未查到。
- 未包含 2-bit 極低 bit 設定（由 Quant-dLLM 補上）。

## 與其他論文的關係

- 與 **Quantization Meets dLLMs (2508.14896)** 同期互補：後者是 benchmark，本篇是首個 dLLM 專用 PTQ 演算法；兩者都指出 W4A4 是 dLLM 量化的分水嶺。
- **STaR-Quant (2606.04945)** 把 DLLMQuant 當作主要基線之一，並把「mask/unmask 分佈差異」與「時間誤差累積」進一步以 SGAT / TAC 兩個模組處理。
- **Quant-dLLM (2510.03274)** 的 MCS 與本篇的 TMAS 想法相近（都讓校準資料模擬 timestep-dependent masking），但目標是 2-bit weight-only。
- 建立在 GPTQ（Hessian 誤差補償）、QuaRot / DuQuant（rotation）等 AR PTQ 之上。
