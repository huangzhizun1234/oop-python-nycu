# DMax: Aggressive Parallel Decoding for dLLMs

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2604.08302`（v3 2026-05） |
| 作者 / 單位 | Zigeng Chen, Gongfan Fang, Xinyin Ma, Ruonan Yu, Xinchao Wang（xML Lab, National University of Singapore） |
| 日期 | 2026-04 |
| 類別 | 平行與投機解碼（訓練式：自我修正的 uniform dLLM） |
| 連結 | [arXiv](https://arxiv.org/abs/2604.08302) · [GitHub](https://github.com/czg1225/DMax) · [HF](https://huggingface.co/Zigeng/DMax-16B) |

## 一句話總結

把 masked dLLM 擴充成能「修正自己錯誤預測」的 uniform dLLM（On-Policy Uniform Training），並用 mask/token embedding 內插做 Soft Parallel Decoding，讓 LLaDA-2.0-mini 的每次 forward 產出 token 數（TPF）從 2.8 拉到 6.2 而準確度不變，2 張 H200 上超過 1000 tok/s。

## 要解決的問題

- 平行解碼的根本風險是 error accumulation：masked dLLM 一旦 unmask 就不能改，平行度一高，早期錯誤會汙染後續上下文。LLaDA-2.0-mini 在 GSM8K 把 TPF 推到 5.8 時準確度從 92.6% 崩到 42.1%。
- WINO 類 training-free 撤銷法只能靠信心啟發式，且撤銷會增加步數。
- Uniform（可從錯 token 恢復）dLLM 從頭訓練成本極高，需要能便宜地從現有 masked dLLM 轉換。

## 核心方法與特色

- **On-Policy Uniform Training**：用模型自己（on-policy）在平行解碼時真正產生的錯誤預測作為訓練輸入，教它從「mask + 自己的錯 token」的混合狀態恢復乾淨序列；這統一了 masked dLLM（輸入含 mask）與 uniform dLLM（輸入含錯 token）兩種目標，只需在 LLaDA-2.0-mini 上做 SFT 級別的微調（8 GPU，dFactory/VeOmni）。
- **Soft Parallel Decoding（embedding 內插）**：每個尚未定案的位置不再是「mask 或 token」二元狀態，而是 mask embedding 與預測 token embedding 依信心加權的內插；上一步的信心作為先驗傳給下一步，讓模型逐步從 mask 端「滑」向 token 端，形成 progressive self-refinement。
- **自我修正能力**：因為訓練時看過自己的錯誤，推論時可以在後續步驟覆寫先前低信心的預測，等於把 WINO 的 revoke 內建進模型，而不需額外驗證分支。
- **激進的 threshold**：官方 quick start 以 `threshold=0.0`、block 32、gen_length 2048 執行，即幾乎每步都接受大量 token；提供通用 / Math / Coder 三個 16B 版本。
- **代價**：需要微調且訓練資料是自身軌跡（釋出 math 與 code 兩套）；目前僅在 LLaDA-2.0-mini（MoE 16B）上驗證。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| LLaDA-2.0-mini | 16B（MoE） | 基座；DMax-16B / DMax-Math-16B / DMax-Coder-16B 皆由此微調 |

## PPA / 效能數據

（trade-off 圖數值由官方圖讀取）

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| 平均 TPF | 2.8 → 6.2 | LLaDA-2.0-mini 原始 | 準確度維持 |
| 數學 / 推理 TPF | 6.0 | — | GSM8K、MATH500、Minerva_Algebra、ASDIV |
| 程式 TPF | 6.6 | — | HumanEval、MBPP（含 Plus） |
| GSM8K | TPF 2.04 → 5.47；DMax 93.5% @≈4.1 TPF、90.4% @≈6.0 TPF | LLaDA-2.0-mini 92.6% @2.0 TPF，42.1% @5.8 TPF | 準確度 vs TPF 曲線 |
| MATH500 | DMax 78.0% @≈3.5 TPF、71.6% @≈6.6 TPF | 基準 75.8% @2.6 TPF；15.2% @6.3 TPF | 同上 |
| HumanEval | DMax 88.4% @≈6.5 TPF、83.5% @≈7.3 TPF | 基準 84.2% @4.4 TPF；3.7% @6.6 TPF | 同上 |
| MBPP | TPF 2.71 → 5.86；DMax 79.2% @≈5.9 TPF | 基準 80.6% @2.7 TPF；2.3% @6.0 TPF | 同上 |
| Throughput | > 1000 tok/s（平均 1338 TPS，batch size 1） | — | 2x NVIDIA H200，dInfer 引擎 |

## 限制 / 備註

- 高 TPF 端（≥6）仍有 2–4 個百分點的準確度下降（GSM8K 92.6→90.4、MATH500 75.8→71.6），「preserving accuracy」指的是中段設定。
- 依賴 dInfer + sglang/vllm 客製推論堆疊，數字含系統層優化，與純演算法比較時需注意。
- 目前只有 16B MoE 一種基座，對 dense 7–8B 模型的效果未驗證。

## 與其他論文的關係

- 同團隊 **dParallel**（ICLR 2026）的續作：dParallel 讓信心更快收斂，DMax 更進一步允許錯了再改。
- 把 **WINO** 的「可撤銷」從解碼演算法內化成模型能力；與 **STaRR / TACG** 這些 remask 啟發式互補。
- 訓練上與 **uniform-state diffusion（UDLM）** 系列相關，是「用 masked dLLM 初始化 uniform dLLM」的實用方案。
- 使用 **LLaDA-2.0-mini** 與 **dInfer** 生態，與 Fast-dLLM v2 / SDAR 這類 block diffusion 模型是不同基座路線。
