# Gemini Diffusion（Google DeepMind 實驗性文字擴散模型）

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | 無論文；官方模型頁 deepmind.google/models/gemini-diffusion（Google I/O 2025 發布） |
| 作者 / 單位 | Google DeepMind |
| 日期 | 2025-05 |
| 類別 | 模型 |
| 連結 | [官方頁](https://deepmind.google/models/gemini-diffusion/) · （後續開源版 DiffusionGemma 技術報告 arXiv 2608.00146，2026-06） |

## 一句話總結

Google 在 I/O 2025 展示的實驗性擴散語言模型：自報取樣速度 1,479 tok/s（不含 0.84 秒固定開銷），約為同品質 AR 模型 Gemini 2.0 Flash-Lite 的 5 倍，程式 benchmark 幾乎持平、但推理任務明顯落後。

## 要解決的問題

- 與 Mercury 相同：AR 逐 token 生成造成的延遲下限。Google 的定位是「以噪聲到文字的平行細化」達到極低延遲，並利用擴散模型可在生成過程中修正錯誤（editing）的特性，提升程式 / 數學等需要一致性的輸出。
- 作為 Gemini 產品線的探索，目標是驗證擴散架構在 Gemini 等級的資料與基礎設施上能否達到與 Flash-Lite 同級的品質。

## 核心方法與特色

- **噪聲到文字的平行細化**：官方描述為「learn to generate outputs by converting random noise into coherent text or code」，一次生成整個 token 區塊並逐步細化，而非逐 token；每步可回頭修改先前的錯誤。這是標準的離散（遮罩）擴散推論形態，但官方未說明遮罩機制、步數或 block 化。
- **速度來源**：每次 forward 產生多個 token；官方以 1,479 tok/s 的「平均取樣速度」呈現，並在圖表註明排除 0.84 秒的啟動開銷（overhead），代表其延遲曲線是「先等 ~0.8 秒、再瞬間吐出整段」。
- **編輯式生成**：官方強調「iterative refinement」讓模型在生成中修正錯誤，適合程式與數學等對一致性要求高的任務；這對應學術界的 remasking / self-correction 研究方向。
- **與 Flash-Lite 的品質對齊**：Google 選擇與自家最小/最快的 Gemini 2.0 Flash-Lite 對比，而非 Flash / Pro，說明其定位是「速度型」模型。
- **代價 / 未知**：無參數量、訓練資料、架構、去噪步數、是否使用 KV cache 等任何資訊；僅提供 waitlist demo，至 2026 中未開放 API。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| Gemini Diffusion（experimental） | 未揭露 | 僅 demo / waitlist |
| Gemini 2.0 Flash-Lite | 未揭露（AR） | 官方唯一對照 |
| DiffusionGemma（2608.00146） | 開源版；官方標示 experimental，MMLU / coding 低於 Gemma 4 | 2026-06 後續，備註 |

## PPA / 效能數據

> 軟體論文：speedup、throughput (tok/s)、latency、記憶體、benchmark 準確度變化。

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| 取樣速度 | **1,479 tok/s**（不含 0.84 s 固定開銷） | Gemini 2.0 Flash-Lite（約 5× 慢；一份社群筆記記為 181 tok/s，待確認） | 官方自報；硬體與 batch 未揭露 |
| 第三方上手實測 | ≈857 tok/s | — | 社群量測，非官方 |
| HumanEval | 89.6 | Flash-Lite 90.2 | 官方表 |
| MBPP | 76.0 | Flash-Lite 75.8 | 官方表 |
| LiveCodeBench v6 | 30.9 | Flash-Lite 28.5 | 官方表 |
| LBPP (v2) | 56.8 | Flash-Lite 56.0 | 官方表 |
| BigCodeBench | 45.4 | Flash-Lite 45.8 | 官方表 |
| SWE-Bench Verified | 22.9 | Flash-Lite 未查到 | 官方表（非 agentic 設定） |
| AIME 2025 | 23.3 | Flash-Lite 20.0 | 官方表 |
| GPQA Diamond | 40.4 | Flash-Lite 56.5 | 官方表，明顯落後 |
| BIG-Bench Extra Hard | 15.0 | Flash-Lite 21.0 | 官方表，明顯落後 |
| Global MMLU (Lite) | 未查到 | — | 官方表有列，數值未取得 |

## 限制 / 備註

- 所有數字皆為 Google 自報，無論文、無第三方複現；速度圖排除固定 overhead，實際首 token 延遲不低。
- 推理類 benchmark（GPQA Diamond −16 分、BBEH −6 分）明顯落後 AR，顯示當時擴散模型在長鏈推理上的差距；程式與短答案任務則持平。
- 從 2025-05 發布到 2026 年中始終停留在 waitlist demo；2026-06 才以 DiffusionGemma 形式開源，且官方承認品質低於 Gemma 4。
- 對本 survey 的價值：提供「大廠基礎設施上的 dLLM 速度上限」參考點（~1.5K tok/s），與 Mercury（H100, 1.1K）、Seed Diffusion（H20, 2.1K）並列。

## 與其他論文的關係

- 與 Mercury（Inception Labs, 2025-02/06）同為首批商用 dLLM；Seed Diffusion 論文摘要直接以「significantly faster than contemporary Mercury and Gemini Diffusion」定位自己。
- 開源模型 LLaDA2.0-flash（535 tok/s）、Nemotron-Labs-Diffusion（850 tok/s on GB200）等以其為速度標竿；LLaDA2.0-flash 在 HumanEval（94.5 vs 89.6）與 LiveCodeBench（42.3 vs 30.9）上高於它。
- Google 內部相關研究（MD4「Simplified and generalized masked diffusion」、AR2Diff 轉移學習）可能是其技術基礎，但官方未確認。
- 後續 DiffusionGemma（2608.00146）為其開源延伸。
