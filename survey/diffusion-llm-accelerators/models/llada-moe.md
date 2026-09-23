# LLaDA-MoE: A Sparse MoE Diffusion Language Model

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2509.24389` |
| 作者 / 單位 | Fengqi Zhu, Zebin You, Yipeng Xing et al.（中國人民大學、螞蟻集團 inclusionAI、上海交大、浙江大學） |
| 日期 | 2025-09 |
| 類別 | 模型 |
| 連結 | [arXiv](https://arxiv.org/abs/2509.24389) · [HF (inclusionAI/LLaDA-MoE-7B-A1B-Base)](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Base) · [dInfer](https://github.com/inclusionAI/dInfer) |

## 一句話總結

第一個從零預訓練的 MoE 擴散語言模型：7B 總參數、每 token 只啟用 1.4B，在 ~20T tokens 上訓練後，以約 1B 啟用參數的推論成本超越 LLaDA-8B / LLaDA-1.5 / Dream-7B 等 dense dLLM，並逼近 Qwen2.5-3B-Instruct；配合螞蟻自家 dInfer 引擎達 1,100+ tok/s。

## 要解決的問題

- dLLM 推論每一步都要對整段序列做完整 forward，且步數多，因此「每步的計算量」是速度瓶頸；MoE 能把每步啟用參數降到 1/5，但 masked diffusion 的訓練目標（隨機遮罩比例、雙向注意力）是否與 MoE 的 routing / load balancing 相容，此前無人驗證。
- 同時要證明擴散模型的 scaling 不必靠 dense 堆參數，稀疏架構在 20T tokens 規模下仍能發揮 MoE 的優勢。

## 核心方法與特色

- **MoE 直接套進 masked diffusion 目標**：Transformer 骨幹（RMSNorm、SwiGLU、RoPE、QK-LayerNorm），16 層、hidden 2048，每層 64 個專家、router 選 top-8，專家維度 1024；總非嵌入參數 7B、啟用 1.4B。訓練目標與 LLaDA 相同（隨機遮罩比例的 ELBO），額外加上 load-balancing loss (0.01) 與 z-loss (0.001) 穩定 routing。
- **四階段 20T tokens 訓練**：預訓練 Stage 1（10T）→ Stage 2（10T）→ 退火 Stage 1（500B, 4k context）→ 退火 Stage 2（500B, 延長到 8k context），之後做 SFT 得到 Instruct 版；資料規模是 LLaDA-8B（2.3T）的近 9 倍。
- **推論成本換算**：每步只算 1.4B 參數，使同樣的 diffusion 步數下 FLOPs 約為 LLaDA-8B 的 1/5；這是「模型層面」的加速，可與 KV cache、平行解碼等「系統層面」加速疊加。
- **搭配 dInfer 推論框架**：螞蟻同時開源 dInfer（2510.08666），把推論拆成模型 / 迭代管理 / 解碼策略 / KV cache 管理四個模組，對 LLaDA-MoE 提供 prefix / dual KV cache、信心閾值平行解碼、iteration smoothing 等，官方宣稱比 Fast-dLLM 快 10×、比 vLLM 上的 Qwen2.5-3B 快 2–3×。
- **代價**：MoE 的記憶體佔用仍是 7B（需載入全部專家），只降計算不降權重讀取；且從零訓練 20T tokens 的成本遠高於 AR→dLLM 轉換路線（SDAR、LLaDA2.0）。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| LLaDA-MoE-7B-A1B-Base | 7B 總 / 1.4B 啟用 | 64 experts, top-8, 16 層 |
| LLaDA-MoE-7B-A1B-Instruct | 7B / 1.4B | SFT 版 |
| LLaDA-8B-Base / Instruct | 8B dense | 對照 |
| LLaDA-1.5 | 8B dense | 對照 |
| Dream-v0-7B Base / Instruct | 7B dense | 對照 |
| Qwen2.5-3B Base / Instruct | 3B（AR） | 啟用參數相近的 AR 對照 |

## PPA / 效能數據

> 軟體論文：speedup、throughput (tok/s)、latency、記憶體、benchmark 準確度變化。

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| 訓練 tokens | ≈20T（10T+10T+0.5T+0.5T） | LLaDA-8B 2.3T | 四階段 |
| Base 平均分 | 46.94 | LLaDA-8B 43.53；Dream-7B 46.66；Qwen2.5-3B 50.34 | MMLU/MMLU-Pro/GSM8K/MATH/HumanEval/MBPP 六項平均 |
| Base MMLU / GSM8K | 64.59 / 66.41 | LLaDA-8B 65.90 / 70.70；Dream 69.50 / 77.79 | — |
| Base MATH / HumanEval / MBPP | 36.10 / 45.73 / 52.40 | LLaDA-8B 27.30 / 33.50 / 38.20 | — |
| Instruct 平均分（9 項） | 53.12 | LLaDA-8B-Inst 42.65；LLaDA-1.5 45.56；Dream-Inst 46.51；Qwen2.5-3B-Inst 53.51 | 含 Drop / IFEval / BFCL-Live |
| Instruct MMLU / MMLU-Pro | 67.18 / 44.64 | Qwen2.5-3B-Inst 69.11 / 44.13 | MMLU-Pro 反超 Qwen |
| Instruct GSM8K / MATH | 82.41 / 58.68 | LLaDA-1.5 83.30 / 42.60；Qwen2.5-3B-Inst 86.28 / 67.02 | — |
| Instruct HumanEval / MBPP | 61.59 / 70.02 | LLaDA-1.5 52.40 / 42.80；Qwen2.5-3B-Inst 60.37 / 65.81 | 程式任務超越 Qwen |
| Instruct IFEval / BFCL-Live | 59.33 / 63.09 | LLaDA-1.5 58.23 / 66.20；Qwen 58.20 / 50.40 | agent 任務明顯強 |
| 推論 throughput（dInfer） | 1,100+ tok/s（HumanEval, bs=1）；六項 benchmark 平均 800+ tok/s | 比 Fast-dLLM 快 10×；比 vLLM Qwen2.5-3B 快 2–3× | 單節點 8×H800，dInfer 框架（非本論文，為 2510.08666 數據） |
| 每步啟用參數 | 1.4B | LLaDA-8B 8B | 理論每步 FLOPs ≈1/5 |

## 限制 / 備註

- 論文本身未報告 tok/s；速度數字全部來自螞蟻後續的 dInfer 論文與 README，屬同一團隊自評，尚無獨立重現。
- MMLU、GSM8K 等知識/數學任務仍略低於 dense LLaDA-8B 與 Dream-7B（Base 版），MoE 的優勢主要在 Instruct 後的程式、agent、對齊任務。
- 需載入 7B 權重，記憶體與 dense 7B 相同；對硬體加速器而言 expert routing 帶來的不規則存取是額外成本。
- 後續有 LLaDA-MoE v2（2608.03457, 2026-08）進一步 scaling，本文未涵蓋。

## 與其他論文的關係

- 直接延續 LLaDA（同一 RUC 團隊 + 螞蟻），保留相同 masked diffusion 目標與推論方式，只把 FFN 換成 MoE 並放大資料量。
- 是 dInfer（2510.08666）與 LLaDA2.0 的前置：dInfer 以 LLaDA-MoE 為主要展示模型；LLaDA2.0 則改走「AR MoE（Ling）→ block diffusion 轉換」路線，規模到 100B。
- 與 Dream-7B、LLaDA-1.5 為主要 dense dLLM 對照；與 SDAR-30B-A3B（Qwen3 MoE 轉換）形成「從零訓練 MoE dLLM vs. AR MoE 轉換」的對比。
- 為後續 dLLM MoE 專用加速（如 dMoE 2605.30876、TEAM）提供基礎模型。
