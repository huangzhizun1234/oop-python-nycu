# dLLM: Simple Diffusion Language Modeling

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2602.22661` |
| 作者 / 單位 | Zhanhui Zhou, Lingjie Chen, Hanghang Tong, Dawn Song（UIUC、UC Berkeley；依作者所屬推測，待確認） |
| 日期 | 2026-02 |
| 類別 | 模型（訓練 / 推論 / 評測框架） |
| 連結 | [arXiv](https://arxiv.org/abs/2602.22661) · [GitHub](https://github.com/ZHZisZZ/dllm) · HF: dllm-hub |

## 一句話總結

一個把擴散語言模型的訓練、推論、評測統一成標準化 pipeline 的開源框架（Apache-2.0）：可直接重現、微調、部署、評測 LLaDA / LLaDA-MoE / LLaDA2.x / Dream，內建 MDLM 與 BD3LM 兩種訓練演算法、Fast-dLLM 式 cache 與信心閾值解碼，並提供用小算力從 BERT 或 AR 小模型做出 dLLM 的可重現配方。

## 要解決的問題

- 2025 年後 dLLM 論文迅速收斂到一組共同元件（遮罩擴散 / block 擴散目標、remasking 取樣器、KV cache 近似、信心平行解碼、lm-eval 評測），但這些元件散落在各自 ad-hoc 的 codebase 中、實作不透明，難以重現與公平比較——尤其加速論文常因取樣設定不同而數字互相矛盾。
- 缺乏「低成本入門」路徑：從零訓練 dLLM 需數 T tokens，一般研究者無法驗證想法。

## 核心方法與特色

- **統一抽象**：把 dLLM 拆成 (1) 訓練目標（masked diffusion / MDLM、block diffusion / BD3LM）、(2) 統一取樣器（抽象化 remasking 策略、步數、block 大小等推論細節）、(3) 評測（基於 lm-evaluation-harness，涵蓋 MMLU-Pro、GSM8K、MATH、Countdown、Sudoku、程式）。新方法只需替換其中一個元件。
- **訓練支援**：基於 transformers Trainer 的 SFT 與預訓練、LoRA、DeepSpeed ZeRO-1/2/3、FSDP，並支援 GRPO 強化學習（推理任務）；可直接微調 LLaDA-8B、Dream-7B、LLaDA-MoE、LLaDA2.0 / 2.1。
- **推論加速整合**：內建 Fast-dLLM 的 KV cache 與 confidence-threshold 平行解碼，作為所有支援模型的通用加速選項；提供互動式 chat 介面。
- **小模型配方與 checkpoint**：BERT-Chat（把任意 BERT 編碼器轉成可對話的 dLLM）、Tiny-A2D（0.5B / 0.6B 的 AR→diffusion 轉換）、Edit Flows（含插入 / 刪除 / 替換的編輯式擴散），並釋出對應權重，讓單卡等級算力也能做 dLLM 研究。
- **代價 / 定位**：本身不提出新模型或新加速演算法，沒有 PPA 數字；價值在於為加速研究提供可重現的 baseline 與評測協定。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| LLaDA-8B-Base / Instruct | 8B | 可重現 / 微調 |
| LLaDA-MoE-7B-A1B | 7B / 1.4B 啟用 | 支援 |
| LLaDA2.0 / LLaDA2.1（mini / flash） | 16B-A1B / 100B-A6B | 支援 |
| Dream-v0-Base-7B | 7B | 支援 |
| BERT-Chat 系列 | BERT 級（~0.1–0.3B） | 框架釋出 |
| Tiny-A2D | 0.5B / 0.6B | AR→diffusion 小模型，框架釋出 |
| Edit Flows | — | 編輯式擴散範例 |

## PPA / 效能數據

> 軟體論文：speedup、throughput (tok/s)、latency、記憶體、benchmark 準確度變化。

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| 論文 / README 提供的加速或準確率數字 | 論文未提供（框架論文；重現既有模型結果） | — | — |
| 內建加速選項 | Fast-dLLM KV cache + 信心閾值解碼 | 原始 LLaDA / Dream 取樣 | 具體倍率依 Fast-dLLM 論文（LLaDA-8B 8.1×–27.6×） |
| 支援的訓練演算法 | MDLM、BD3LM | — | — |
| 支援的分散式訓練 | DeepSpeed ZeRO-1/2/3、FSDP、LoRA | — | — |
| 評測 benchmark | MMLU-Pro、GSM8K、MATH、Countdown、Sudoku、Code 等 | lm-eval-harness | — |

## 限制 / 備註

- 沒有任何 PPA 數據；對本 survey 的用途是「統一的 baseline / 評測環境」，可用來公平比較不同加速方法（同一模型、同一取樣器、同一 harness）。
- 目前推論加速只整合 Fast-dLLM 一種；dInfer、JetEngine、SGLang 等高吞吐引擎未包含。
- Tiny-A2D / BERT-Chat 為教學級小模型，非 SOTA。

## 與其他論文的關係

- 對 LLaDA、LLaDA-MoE、LLaDA2.x、Dream 提供統一實作；對 BD3-LM 提供訓練演算法；對 Fast-dLLM 提供推論整合。
- 與 Edit Flows、LLaDA2.1 token editing、Seed Diffusion edit-based 階段同屬「編輯式擴散」方向的實作入口。
- 與 dInfer（螞蟻）、JetEngine（SDAR）分工：dLLM 偏訓練與評測的可重現性，後兩者偏高吞吐 serving。
- 可作為本 survey 中量化 / 快取 / 平行解碼論文的共同重現平台。
