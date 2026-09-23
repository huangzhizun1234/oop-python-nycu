# Seed Diffusion: A Large-Scale Diffusion Language Model with High-Speed Inference

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2508.02193`（Seed Diffusion Preview 技術報告） |
| 作者 / 單位 | Yuxuan Song, Zheng Zhang, Cheng Luo, Pengyang Gao, Fan Xia, Hao Luo, … , Hao Zhou（ByteDance Seed；合作含清華 AIR） |
| 日期 | 2025-08（Preview 2025-07-31 發布） |
| 類別 | 模型 |
| 連結 | [arXiv](https://arxiv.org/abs/2508.02193) · [Blog](https://seed.bytedance.com/seed_diffusion) · [Demo](https://studio.seed.ai/exp/seed_diffusion) |

## 一句話總結

ByteDance 的大規模離散擴散程式模型：透過「遮罩→編輯」兩階段課程、限制順序的軌跡蒸餾、以及 on-policy 步數最小化訓練，配合 block 級平行取樣，在 H20 GPU 上達到 2,146 tok/s（同規模 AR 的 5.4×），程式 benchmark 與 AR 相當、程式編輯任務更佳，速度超越 Mercury 與 Gemini Diffusion。

## 要解決的問題

- 標準 masked diffusion 只學「遮罩→還原」，推論時已生成的 token 無法再修正，且訓練時的隨機遮罩比例與推論時「多步、左到右偏好」的實際軌跡不一致，導致要用很多步才能維持品質，速度優勢被吃掉。
- 純平行取樣忽略了語言的因果結構（如程式的語法依賴），而純 AR 又太慢；需要在「任意順序」與「左到右」之間找到既快又不掉品質的解碼順序。
- 前人（Mercury、Gemini Diffusion）已證明可行，但速度仍在 1–1.5K tok/s；ByteDance 目標是在更弱的 H20（受出口管制的降規 GPU）上突破 2K tok/s。

## 核心方法與特色

- **兩階段課程（Two-Stage Curriculum, TSC）**：第一階段為標準 mask-based diffusion（隨機遮罩、還原）建立生成能力；第二階段改為 edit-based diffusion（對已生成序列施加插入 / 刪除 / 替換等擾動再學會修正），讓模型能修正早期錯誤，緩解「一旦解碼就固定」的問題，也提升程式編輯類任務。
- **限制順序學習（constrained-order trajectory distillation）**：從眾多可能的去噪順序中，篩選出符合因果 / 語法依賴的高品質軌跡（例如先生成函式簽名再填內容），用這些軌跡做蒸餾，讓模型的平行解碼順序不違反語言結構。
- **On-policy 軌跡優化**：以模型自身取樣的軌跡為訓練資料，直接以「最少步數且輸出正確」為目標優化，鼓勵每步接受更多 token；這是把 Fast-dLLM 式的「信心平行解碼」內化為訓練目標。
- **Block 級平行擴散取樣（semi-AR）**：推論時以 block 為單位由左到右生成、block 內平行去噪，block 間可用 KV cache；並針對 H20 做系統層級優化（自研推論框架）。
- **代價**：僅支援程式任務（Preview）、模型規模未揭露、技術細節（步數、block 大小、蒸餾資料）在報告中描述有限；速度為自家框架自報。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| Seed Diffusion Preview | 未揭露 | 程式專用，閉源 demo |
| 對照：Mercury Coder、Gemini Diffusion | 未揭露 | 速度對比 |
| 對照：Qwen2.5-Coder / Seed-Coder 等同規模 AR | ~8B 級（依記憶，待確認） | 品質 / 速度對比 |

## PPA / 效能數據

> 軟體論文：speedup、throughput (tok/s)、latency、記憶體、benchmark 準確度變化。

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| 推論速度 | **2,146 tok/s** | 同規模 AR 模型 **5.4×**（官方 blog）；「significantly faster」than Mercury（1,109 on H100）與 Gemini Diffusion（1,479） | NVIDIA H20 GPU，自研推論框架，batch 未揭露 |
| 程式 benchmark（HumanEval、MBPP、BigCodeBench、LiveCodeBench、MBXP、NaturalCodeBench、Aider、CanItEdit） | 「與同規模 AR 相當」；程式編輯（CanItEdit / Aider）「展現優勢」 | Qwen2.5-Coder / Seed-Coder 級 AR | 論文有完整表格，數值未查到 |
| 速度-品質 Pareto | 宣稱為程式模型新 SOTA 前緣 | Mercury、Gemini Diffusion | 論文圖 |
| 兩階段課程 / on-policy 對步數的影響 | 論文有消融，數值未查到 | — | — |

## 限制 / 備註

- 參數量、訓練資料、步數、block 大小皆未公開；速度來自 H20 + 自研框架，與 Mercury（H100）、Gemini（未知硬體）不可直接換算。
- 僅 Preview 且限程式任務，一般能力「即將推出」；未開放 API 或權重。
- ByteDance Seed 後續開源 Stable-DiffCoder（2601.15892，8B，block diffusion 續訓）可視為其開源近親，但非同一模型。
- 對本 survey 的價值：展示「訓練期優化解碼軌跡（on-policy、順序約束）」是除了 KV cache / 平行解碼之外，另一條把 dLLM 推向 2K tok/s 的路徑。

## 與其他論文的關係

- 直接與 Mercury、Gemini Diffusion 競爭，並以兩者為速度對照。
- 方法上結合了 block diffusion（BD3-LM / LLaDA2.0 / SDAR 的 semi-AR 取樣）、edit-based diffusion（與 Edit Flows、LLaDA2.1 token editing 同方向）與 on-policy 步數優化（與 DiffuCoder coupled-GRPO、d1、TraceRL 等 dLLM RL 相關）。
- 其 on-policy「最少步數」目標與 Fast-dLLM v2 / LLaDA2.0-CAP 把平行解碼內化到訓練的思路一致。
- Stable-DiffCoder（同團隊，2026-01）是開源的後續，採 block diffusion CPT + block-wise clipped noise schedule（承自 BD3-LM）。
