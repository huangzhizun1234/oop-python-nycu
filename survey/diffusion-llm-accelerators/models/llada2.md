# LLaDA2.0: Scaling Up Diffusion Language Models to 100B

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2512.15745`（模型 2025-11 釋出、論文 2025-12） |
| 作者 / 單位 | Tiwei Bie, Maosong Cao, Kun Chen, Lun Du et al.（40+ 位作者，Ant Group inclusionAI 主導；合作：人民大學、浙江大學、西湖大學、HKUST） |
| 日期 | 2025-12 |
| 類別 | 模型 |
| 連結 | [arXiv](https://arxiv.org/abs/2512.15745) · [GitHub](https://github.com/inclusionAI/LLaDA2.X) · [LLaDA2.1 arXiv](https://arxiv.org/abs/2602.08676) · [dInfer](https://github.com/inclusionAI/dInfer) |

## 一句話總結

把預訓練好的 AR MoE 模型（Ling-mini-2.0 / Ling-flash-2.0）透過「block size 先放大再縮回」的三階段 WSD 訓練，系統性轉換成 block diffusion 語言模型，做出 16B-A1B 與 100B-A6B 兩個 dLLM；100B 版與 Qwen3-30B-A3B 分數相當，加上 CAP 訓練後在 SGLang 上達 535 tok/s（2.1× AR）。

## 要解決的問題

- 從零訓練 dLLM（LLaDA、LLaDA-MoE）資料效率低、成本高，無法追上 AR 的 scaling 節奏；而直接把 AR 權重換成全雙向注意力（Dream）會破壞預訓練權重分布，仍需數百 B tokens。
- 全序列 dLLM 無法用 KV cache，長輸出時每步都重算整段，速度優勢在實際 serving 中兌現不了；需要一種既保留雙向去噪、又能逐 block 快取的架構，且要能在 100B 規模上訓練穩定。

## 核心方法與特色

- **Block Diffusion Language Model (BDLM) 作為統一框架**：block size = 1 就是 AR（next-token），block size = 整段就是全序列 masked diffusion；在 block 之間保持因果、block 內部雙向去噪。這讓 AR 與 dLLM 成為同一參數化下的兩端，可以用「調 block size」平滑轉換。
- **三階段 block-level WSD 訓練**：(1) Warmup：block size 由 1 → 4 → 32 → 64 → 4096 逐步放大，讓 AR 權重漸進適應雙向建模；(2) Stable：以 4096（等於全序列）做大規模 masked diffusion 預訓練，超過 100B tokens；(3) Decay：把 block size 收斂回 32，得到部署用的 block diffusion 模型。論文說明 block size 32 在速度與品質之間最平衡。
- **AR→dLLM 轉換而非從零訓練**：mini 來自 Ling-mini-2.0（16B-A1B）、flash 來自 Ling-flash-2.0（100B-A6B，約 6.1B 啟用），繼承 AR 模型的知識與 MoE 架構，32K context。
- **CAP（Confidence-Aware Parallel）訓練**：在 SFT 損失外加一項 confidence loss（L = L_SFT + λ·L_conf），讓模型對「可平行解碼的 token」給出更尖銳的信心分布；推論時用信心閾值（0.85–0.95）決定每步一次接受幾個 token，直接提高 tokens-per-forward (TPF)。
- **推論引擎**：dInfer 與 SGLang 整合，支援 block 級 KV cache 重用、平行解碼、TP8；官方所有速度數字都在此引擎上量測。
- **代價**：block diffusion 犧牲了全序列 dLLM 的任意順序 / 全域規劃能力；Warmup + Stable + Decay 仍需數百 B tokens 的訓練。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| LLaDA2.0-mini | 16B 總 / ≈1.4B 啟用（MoE） | 由 Ling-mini-2.0 轉換 |
| LLaDA2.0-flash | 100B 總 / ≈6.1B 啟用（MoE） | 由 Ling-flash-2.0 轉換，「最大的 dLLM」 |
| LLaDA2.0-mini-CAP / flash-CAP | 同上 | 加 CAP 訓練，平行解碼強化版 |
| LLaDA2.0-mini/flash-preview | 同上 | 早期預覽版（對照） |
| Qwen3-30B-A3B-Instruct-2507、Ling-flash-2.0、Qwen3-8B、Ling-mini-2.0 | AR 對照 | — |

## PPA / 效能數據

> 軟體論文：speedup、throughput (tok/s)、latency、記憶體、benchmark 準確度變化。

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| flash 平均分 | 73.18 | Qwen3-30B-A3B-Instruct-2507 73.60；Ling-flash-2.0 72.15 | 知識/推理/程式/數學/agent/對齊多項平均 |
| flash HumanEval / MBPP | 94.51 / 88.29 | Qwen3-30B-A3B 93.29 / 86.65 | 程式任務超越 AR |
| flash AIME 2025 | 60.00 | Qwen3-30B-A3B 61.88；Ling-flash-2.0 55.89 | — |
| flash LiveCodeBench / BigCodeBench | 42.29 / 41.58 | Gemini Diffusion 30.9 / 45.4 | 第三方彙整，待確認 |
| mini 平均分 | 64.34 | Qwen3-8B 63.42；Ling-mini-2.0 65.77 | — |
| mini HumanEval / MMLU | 86.59 / 80.53 | Qwen3-8B 84.76 / 80.94 | — |
| 推論 TPS（flash-CAP） | **535 tok/s** | LLaDA2.0-flash 無 CAP 383；Ling-flash-2.0 256；Qwen3-30B-A3B 237 → **2.1×** | SGLang, TP8, H20, 推理/程式任務（第三方筆記註明條件） |
| TPF vs. 閾值（block 32） | 閾值 0.95 → 2.55 TPF（品質 70.15）；0.90 → 2.9；0.85 → 3.31 | AR = 1.0 | 品質隨閾值降低而下降 |
| dInfer 8×H20, flash-CAP | bs=1 平均 580.7 tok/s（HumanEval 753.1, GSM8K 591.9, IFEval 222.6）；bs=32 平均 1,966 tok/s（HumanEval 2,558.5） | — | dInfer README |
| Stable 階段資料量 | >100B tokens | — | Warmup / Decay 為「中等規模」，論文未給精確值 |
| 後續 LLaDA2.1（token editing） | 892 TPS（程式任務）；flash Q-mode 平均 73.54 / TPF 3.64 | LLaDA2.0-flash 72.43 / TPF 3.08 | 2602.08676，備註 |

## 限制 / 備註

- 所有速度皆為螞蟻自家引擎（dInfer / SGLang）上的自報數字，尚無獨立重現；TPS 高度依賴任務熵（IFEval 僅 222 tok/s vs HumanEval 753）。
- 啟用參數 6B 的 flash 與啟用 3B 的 Qwen3-30B-A3B 比速度，每步 FLOPs 其實較高，加速來自 TPF>1 而非模型更輕。
- Block size 固定 32，長推理鏈仍需逐 block 串行；GPQA 等純推理任務 flash（62.31）低於 Ling-flash-2.0（69.16），顯示轉換有損。
- CAP 的信心閾值需按任務調整，閾值太低會明顯掉分。
- 系列後續：LLaDA2.1（2026-02，token editing 加速）、LLaDA2.2-flash（2026-07）、LLaDA2.0-Uni（多模態），本文只涵蓋 2.0。

## 與其他論文的關係

- 承接 LLaDA / LLaDA-MoE（同體系）但方法論改為 BD3-LM 式 block diffusion 與 AR→dLLM 轉換；與 SDAR（Qwen3 → block diffusion，50B tokens）、Fast-dLLM v2（Qwen2.5 → block diffusion，~1B tokens）、Efficient-DLM（Qwen3 → block-wise attention）是同一路線的競爭者，LLaDA2.0 的特色是規模最大（100B）且用「block size 放大再縮回」的 WSD 課程。
- CAP 是 Fast-dLLM「信心閾值平行解碼」的訓練期版本（把推論策略內化成訓練目標）。
- 推論依賴 dInfer（2510.08666）與 SGLang；LLaDA2.1 的 token editing、以及 dLLM 框架、chitu 等推論引擎都以 LLaDA2.x 為支援對象。
- 與 Gemini Diffusion / Seed Diffusion / Mercury 等閉源商用 dLLM 相比，是目前開源中規模最大的 dLLM，速度數字（535 tok/s）處於中間。
