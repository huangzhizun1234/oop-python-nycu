# Ultra-Fast Language Generation via Discrete Diffusion Divergence Instruct (DiDi-Instruct)

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2509.25035`（ICLR 2026） |
| 作者 / 單位 | Haoyang Zheng, Xinyang Liu, Cindy Xiangrui Kong, Nan Jiang, Zheyuan Hu, Weijian Luo, Wei Deng, Guang Lin（Purdue University 為主，依記憶，待確認） |
| 日期 | 2025-09 |
| 類別 | 量化 / 壓縮（蒸餾，few-step） |
| 連結 | [arXiv](https://arxiv.org/abs/2509.25035) · [GitHub](https://github.com/haoyangzheng-ai/didi-instruct) · [OpenReview](https://openreview.net/forum?id=mtdyZsa47V) |

## 一句話總結

以「積分 KL 散度最小化」為理論基礎，把預訓練的 masked discrete diffusion LM 蒸餾成 few-step 學生，在 OpenWebText 上以 8–128 NFE 達到優於 teacher 與 GPT-2 的困惑度、最高 64× 加速，且蒸餾訓練時間比競爭方法少 20× 以上。

## 要解決的問題

- dLLM 需要數百到上千次 NFE 才能達到好品質，遠慢於 AR；直接減少步數品質急降。
- 既有離散擴散蒸餾（如 SDTT、Di[M]O 類 distribution matching）要嘛訓練不穩、要嘛訓練成本高、要嘛在極少步（≤16 NFE）品質差；連續擴散的 score distillation（如 Diff-Instruct）無法直接用於離散 token 空間。

## 核心方法與特色

- **Integral KL-divergence minimization**：把「學生的 few-step 生成分佈」與「teacher 的多步邊際分佈」之間沿整條擴散時間積分的 KL 作為目標，推導出可用 policy-gradient 風格估計的實用訓練演算法（學生自己採樣、teacher 提供 reward-like 訊號），適用於離散 token 而不需要連續 score。
- **Grouped reward normalization**：對同一批次 / 同組樣本的 reward 做正規化以降低梯度方差、穩定訓練（類似 GRPO 的思想）。
- **Intermediate-state matching**：不只比對最終輸出，也讓學生在中間噪聲狀態與 teacher 對齊，提高覆蓋率（避免 mode collapse）。
- **Reward-guided ancestral sampler**：推論時用 teacher / reward 訊號引導祖先採樣，進一步提升少步生成品質。
- **代價**：需要訓練（非 training-free）；目前驗證規模為 169M 的 MDLM，尚未擴展到 LLaDA-8B 等大模型；加速倍數以 NFE 比較，不含 kernel 層面優化。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| MDLM（masked diffusion LM，teacher） | 169M | OpenWebText 預訓練 |
| DiDi-Instruct 學生 | 169M | 由 teacher 初始化後蒸餾 |
| GPT-2（對照） | 124M–1.5B（具體型號未查到） | AR 基線 |

## PPA / 效能數據

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| 困惑度（8 NFE） | 62.2 | 優於先前加速 dLLM 與 GPT-2 基線 | OpenWebText validation，169M |
| 困惑度（128 NFE） | 18.4 | 同上 | 同上 |
| 加速 | 最高 64× | teacher（多步 MDLM） | 以 NFE 計 |
| 蒸餾訓練 wall-clock | 減少 > 20× | 競爭 dLLM 蒸餾方法 | — |
| Entropy 損失 | 約 1%（可忽略） | teacher | 生成多樣性 |
| GPU / tok/s | 論文未於可存取來源提供 / 未查到 | — | — |

## 限制 / 備註

- 規模僅 169M、任務為無條件語言建模（困惑度），與 8B 級 dLLM 在 GSM8K / HumanEval 的實用場景仍有距離。
- 困惑度優於 teacher 的部分可能來自 mode-seeking 帶來的 entropy 下降，作者以約 1% 的 entropy 損失佐證其可忽略。

## 與其他論文的關係

- 承接 **Di[M]O (2503.15457)**（masked diffusion 一步蒸餾，主要用於影像）與 **SDTT / Self-Distillation Through Time (2410.21035)**，把離散蒸餾推到語言建模並改善訓練效率。
- 與 **CD⁴LM (2601.02236)**、**T3D (2602.12262)** 同為 dLLM few-step 蒸餾，但後兩者以 8B 級 LLaDA / SDAR 為對象、評估下游推理任務。
- 與 **Diff-Instruct** 系列（連續擴散的 score distillation）在理論上同源，本篇是其離散版本。
- 與量化、KV cache、剪枝方法正交。
