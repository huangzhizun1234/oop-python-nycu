# R²-dLLM: Accelerating Diffusion Large Language Models via Spatio-Temporal Redundancy Reduction

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2604.18995` |
| 作者 / 單位 | Zhenbang Du, Kejing Xia, Xinrui Zhong, Yonggan Fu, Nicolai Oswald, Binfei Ji, Brucek Khailany, Pavlo Molchanov, Yingyan (Celine) Lin（Georgia Tech EIC Lab / NVIDIA） |
| 日期 | 2026-04 |
| 類別 | KV cache 與稀疏（token 剪枝 / 解碼冗餘消除，兼平行解碼） |
| 連結 | [arXiv](https://arxiv.org/abs/2604.18995) · [GitHub](https://github.com/GATECH-EIC/R2-dLLM) |

## 一句話總結

指出 dLLM 解碼有「空間冗餘」（信心相近的 token 群與位置模糊造成一次只解一個）和「時間冗餘」（已穩定的預測被反覆 remask 重算），提出訓練無關的聚合 / 定稿規則加上冗餘感知 SFT，把解碼步數最多減少 88%、最高 10.1× 加速。

## 要解決的問題

- dLLM 每步依信心挑 token 解碼，但常出現多個 token 信心幾乎相同（confidence cluster）或同一個 token 在相鄰位置都有高信心（positional ambiguity），保守策略每步只解一個，造成大量步數浪費。
- 許多 mask 位置的預測在早期就已穩定，但因未被選中而被反覆 remask、重新預測，同樣結果算了很多次。
- 純 training-free 規則需要手調閾值且跨模型不穩定。

## 核心方法與特色

- **空間冗餘：local confidence / token aggregation**：把相鄰且信心相近的 token（confidence_cluster_size、spatial_threshold）視為一個群一次解碼；對「同一預測在鄰近位置重複出現」的情況做 token cluster 聚合，消除位置模糊導致的重複解碼。
- **時間冗餘：finalize temporally stable tokens**：追蹤每個 mask 位置的預測在連續 temporal_steps 步內是否一致（temporal_threshold），一致者直接定稿，不再參與後續 remask 與重算。
- **Redundancy-aware SFT**：把上述高效解碼軌跡當作監督訊號微調模型（釋出 R²-dLLM-LLaDA 與 R²-dLLM-Dream 權重），使模型本身傾向產生「可一次解多 token、早期即穩定」的信心分布，減少對手調閾值的依賴。
- **統一框架**：推論規則可單獨 training-free 使用，也可配合 SFT 權重得到更高加速。
- **代價**：SFT 需要訓練資源（作者用 4×H200）；聚合規則在需要細緻位置控制的生成上可能誤合併；仍有多個超參數。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| LLaDA-8B-Instruct | 8B | 基底；釋出 R²-dLLM-LLaDA 微調版 |
| Dream-v0-Instruct-7B | 7B | 基底；釋出 R²-dLLM-Dream 微調版 |

## PPA / 效能數據

> 軟體論文：speedup、throughput (tok/s)、latency、記憶體、benchmark 準確度變化。GPU：訓練 4×NVIDIA H200 141GB；延遲量測單張 H200。

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| 解碼步數減少 | 最多 88% | 既有解碼策略 | 各模型 / 任務 |
| 最高加速 | 10.1× | 原始解碼 | 依搜尋摘要，具體 benchmark 未確認 |
| 評測任務 | GSM8K、MATH、HumanEval、MBPP | — | LLaDA-8B-Instruct、Dream-7B-Instruct |
| 品質 | 「competitive」，與原模型相當 | 原模型 | — |
| 逐任務數字 | 未查到 | — | — |

## 限制 / 備註

- 逐 benchmark 的準確度 / 步數 / tok/s 未從 README 或搜尋摘要取得。
- 此篇主要減少「步數」（時間軸），與 KV cache（減少每步計算）正交，本檔歸入此類是因其時間冗餘處理與 cache/remask 密切相關。
- 需要 SFT 才能達到最佳效果，非純 training-free。

## 與其他論文的關係

- 「時間冗餘」的觀察與 **dKV-Cache**（已解碼 token 表徵穩定）、**DyLLM**（多數 token 表徵不變）一致，但 R²-dLLM 把它用在「定稿預測」而非「快取 KV」。
- 「空間冗餘」對應 **Fast-dLLM** 信心閾值平行解碼的改良：不是單一閾值，而是局部聚合。
- 與 **Streaming-dLLM** 的 early exit、STDec（spatio-temporal stability decoding）、Local Determinism Propagation 等同期工作在思路上高度重疊。
- SFT 路線與 dParallel、TAD（trajectory self-distillation）等「訓練模型以支援激進平行解碼」的工作相呼應。
