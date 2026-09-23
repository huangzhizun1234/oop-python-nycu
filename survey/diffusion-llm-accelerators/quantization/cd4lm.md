# CD⁴LM: Consistency Distillation and aDaptive Decoding for Diffusion Language Models

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2601.02236` |
| 作者 / 單位 | Yihao Liang et al.（單位未查到） |
| 日期 | 2026-01 |
| 類別 | 量化 / 壓縮（蒸餾，few-step） |
| 連結 | [arXiv](https://arxiv.org/abs/2601.02236) · [GitHub](https://github.com/yihao-liang/CDLM) |

## 一句話總結

把連續擴散的 consistency distillation 搬到離散 token 空間：DSCD 訓練一個「軌跡不變」的學生模型，能從任意噪聲狀態直接跳到乾淨分佈；再配合依置信度動態跳步的 CAD 解碼，在 LLaDA-8B-Instruct 上以 3.62× 平均加速（GSM8K 5.18×）達到不掉分甚至略升的準確度。

## 要解決的問題

- dLLM 訓練時只學「固定 schedule 下的局部轉移」（從 t 到 t−1），但高效推論需要「長跳」——一次 unmask 很多 token、跳過大量中間狀態。這種 static-to-dynamic misalignment 使得直接減少步數時品質崩潰。
- 純 training-free 的平行解碼（如 Fast-dLLM 的 confidence threshold）受限於原模型在未見狀態上的預測品質；需要讓模型本身學會跨多步的一致性。

## 核心方法與特色

- **DSCD（Discrete-Space Consistency Distillation）**：以 LLaDA-8B-Instruct 為 teacher，學生模型被訓練成對同一條軌跡上不同噪聲程度（不同 mask 比例）的狀態都輸出一致的乾淨 token 分佈（trajectory-invariant），因此可以從任意中間狀態「一步到位」；訓練用 temperature 2.0 的軟標籤與課程式權重（λ 由 0.9 漸降到 0.5）平衡 teacher 蒸餾與一致性目標。
- **CAD（Confidence-Adaptive Decoding）**：推論時依每個 token 的置信度（預設門檻 0.95）動態決定本步要 unmask 多少 token，高置信時激進跳步、低置信時多花步數；block length 32 的 semi-autoregressive 解碼。
- **訓練與推論解耦**：DSCD 讓模型「有能力」長跳，CAD 決定「何時」長跳，兩者組合把 accuracy–efficiency Pareto 前緣整體外推。
- **資料**：GSM8K 訓練集與 OpenCodeInstruct（預設 200k 子集）用於蒸餾，訓練需 8× ≥40 GB GPU。
- **代價**：需要蒸餾訓練（非 training-free）；學生模型與 teacher 同尺寸（8B），只減步數、不減參數量與記憶體。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| LLaDA-8B-Instruct | 8B | teacher 與學生初始化 |

## PPA / 效能數據

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| GSM8K | 77.6%，5.18× wall-clock speedup | LLaDA baseline 77.4% | LLaDA-8B-Instruct，gen_length 256、block 32、threshold 0.95 |
| HumanEval | 40.9%，3.30× | 38.7% | 同上 |
| MBPP | 39.0%，2.96× | 36.9% | 同上 |
| MATH500 | 38.6%，5.33× | 37.3% | 同上 |
| 平均 | 3.62× speedup，平均準確度提升 | LLaDA baseline | 四個 benchmark |
| GPU 型號 / tok/s | 論文未於可存取來源提供 / 未查到 | — | — |

## 限制 / 備註

- 對比基準是 fixed-step LLaDA，未查到與 Fast-dLLM、dParallel 等 training-free / 可學習平行解碼方法的直接比較數字。
- 只在 LLaDA-8B-Instruct 上驗證；Dream / SDAR 等其他 dLLM 未測。
- 加速來自減少 NFE，不減少每步計算量，與量化、KV cache 方法正交。

## 與其他論文的關係

- 與 **T3D (2602.12262)**（trajectory self-distillation + DDO，SDAR / LLaDA）、**DiDi-Instruct (2509.25035)**、**OPTD (2608.02942)** 同屬 dLLM few-step 蒸餾路線；CD⁴LM 強調離散 consistency，T3D 強調 reverse-KL 式的 mode-seeking。
- 與 **dParallel (2509.26488)**（可學習平行解碼）目標相同（減少步數），方法互為競爭。
- 與 **Fast-dLLM (2505.22618)** 的 confidence-aware parallel decoding 互補：CAD 可視為其「有蒸餾支撐」的版本，可再疊加 KV cache。
- 與量化類（DLLMQuant、Quant-dLLM）正交，可同時使用以同時減步數與減位元。
