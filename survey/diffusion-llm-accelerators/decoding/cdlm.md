# CDLM: Consistency Diffusion Language Models For Faster Sampling

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2511.19269` |
| 作者 / 單位 | Minseo Kim, Chenfeng Xu, Coleman Hooper, Harman Singh, Ben Athiwaratkun, Ce Zhang, Kurt Keutzer, Amir Gholami（SqueezeAI Lab, UC Berkeley；部分作者依記憶來自 Together AI，待確認） |
| 日期 | 2025-11 |
| 類別 | 平行與投機解碼（訓練式：consistency distillation + block-causal） |
| 連結 | [arXiv](https://arxiv.org/abs/2511.19269) · [GitHub](https://github.com/SqueezeAILab/CDLM) · [HF](https://huggingface.co/minseo25/CDLM-Dream) |

## 一句話總結

把 consistency model 的想法帶進 dLLM：從雙向 teacher 蒸餾出一個 block-causal 的 student，讓它每步能一次「定案」多個 token，同時因為 block-causal mask 可以直接用標準 KV cache，延遲降低 3.6–14.5x。

## 要解決的問題

- dLLM 兩個瓶頸同時存在：(1) refinement 步數多；(2) 全雙向 attention 讓標準 KV cache 不可用（每步都要重算所有 token 的 K/V）。
- 既有方法多半只解其中一個：Fast-dLLM 的近似 cache 有品質損失，dParallel 減步數但仍是雙向、不能 cache。
- 需要一種訓練方式同時改變「步數」與「attention 結構」。

## 核心方法與特色

- **Consistency modeling → multi-token finalization**：模仿 consistency distillation，訓練 student 在任何中間 noise 狀態都直接輸出「軌跡終點」（最終序列），因此一步就能 commit 一整批 token，而不必等信心逐步傳播。
- **Block-wise causal attention mask**：微調時強制 block 之間為 causal、block 內雙向。已完成的 block 的 K/V 從此固定，可用標準 KV cache（與 AR 完全相同的 cache 語意），推論引擎不需特殊近似。
- **從雙向 teacher 蒸餾**：先用原始 Dream / LLaDA（雙向、全步數）離線生成軌跡（preprocess 階段），再訓練 block-causal student 對齊，兩階段皆在 4 GPU 上完成；模型本體架構不變，只換 mask 與權重。
- **推論：block 內 confidence-threshold 平行解碼**：每個 block 內用信心門檻決定要 finalize 哪些 token，跨 block 順序推進，並套用 KV cache。
- **代價**：block-causal 犧牲了「後文影響前文」的能力，長程雙向一致性可能略降；需要蒸餾訓練。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| Dream-7B（Instruct） | 7B | teacher → CDLM-Dream |
| LLaDA-8B（Instruct） | 8B | teacher → CDLM-LLaDA |

## PPA / 效能數據

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| 延遲降低 | 3.6x–14.5x（arXiv 摘要）；README 寫 3.6x–12.8x | 原始 Dream / LLaDA 解碼 | 數學（GSM8K、MATH）與程式（HumanEval、MBPP） |
| 準確度 | 「competitive」 | 原始模型、Fast-dLLM、dLLM-Cache | 逐項數字未查到 |
| 訓練資源 | 4 GPU（型號未載明） | — | preprocess + train 兩階段 |
| tok/s | 未查到 | — | — |

## 限制 / 備註

- 摘要與 README 的加速上限（14.5x vs 12.8x）不一致，可能為不同版本；逐 benchmark 數字需查原文。
- 與 dParallel 的 consistency-distillation baseline 相比（GSM8K 64 步 69.9%），CDLM 宣稱能維持準確度，但兩者設定不同，不能直接對比。
- 程式碼與 **D2F**（Discrete Diffusion Forcing）高度相關，方法上也共享 block-causal 思路。

## 與其他論文的關係

- 建立在 **D2F** 與 **Block Diffusion / SDAR / Fast-dLLM v2** 的 block-causal 路線上，差異在用 consistency 目標把步數壓到極低。
- 與 **dParallel** 競爭同一問題（訓練式減步數）；dParallel 保留全雙向、無 cache，CDLM 改 block-causal 換取 cache。
- 與 **Fast-dLLM**（近似 KV cache）比較：CDLM 的 cache 是精確的，不需 refresh。
- 與 **Jacobi Forcing** 殊途同歸：一個把 dLLM 改成 block-causal，一個把 AR 改成 block-parallel，最終形態都接近「block-causal 平行解碼器」。
