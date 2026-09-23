# Diffusion Language Models Know the Answer Before Decoding (Prophet)

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2508.19982`（OpenReview 有投稿頁；NeurIPS 2025 網站有對應頁面、ML Anthology 索引為 ICLR 2026，正式 venue 待確認） |
| 作者 / 單位 | Pengxiang Li, Yefan Zhou, Dilxat Muhtar, Lu Yin, Shilin Yan, Li Shen, Yi Liang, Soroush Vosoughi, Shiwei Liu（依記憶：Dartmouth College / University of Oxford 等，待確認） |
| 日期 | 2025-08 |
| 類別 | 平行與投機解碼（early commit / 早停） |
| 連結 | [arXiv](https://arxiv.org/abs/2508.19982) · [GitHub](https://github.com/pixeli99/Prophet) |

## 一句話總結

觀察到 dLLM 在解碼到一半時「答案其實已經定了」，Prophet 用 top-2 候選的 confidence gap 當訊號，一旦夠大就一次把剩餘所有 mask token 全部 commit（all-in），training-free 地把解碼步數最多砍 3.4x。

## 要解決的問題

- masked dLLM（LLaDA、Dream）預設每步只 unmask 少量 token，256 token 通常要跑 256 步（或 block 內每步一個），大量後段步驟只是在「確認」早已穩定的預測，是純粹的浪費。
- 既有 few-step / confidence-threshold 方法把「要不要提早 commit」當成固定超參數，沒有一個能反映「模型是否已經確定」的動態訊號；直接減少步數會掉分。
- 作者實測：GSM8K 上高達 97%、MMLU 上高達 99% 的樣本，只用一半的 refinement 步數就已經能正確解碼（early answer convergence）。

## 核心方法與特色

- **Early answer convergence 的實證分析**：記錄 LLaDA-8B 每一步的 denoised 預測（x0_history），統計答案 token 在 25%/50% 步數時就已正確的比例，證明後半段步驟大多不改變答案，這是 early commit 的正當性來源。
- **Top-2 confidence gap 作為 commit 判準**：每步對仍被 mask 的位置看 logits 的 top-1 與 top-2 之差（gap），gap 大代表模型「不猶豫」；當所監控位置（例如答案區）的 gap 全都超過門檻時，觸發 all-in。這比只看 top-1 機率更能區分「高信心但有競爭者」與「真的確定」。
- **Phase-aware 門檻**：官方實作把解碼進度分三段（0–33% / 33–67% / 67–100%），門檻分別為 7.5 / 5.0 / 2.5（logit gap）。早期要求更高信心以免過早 commit，後期放寬以盡早結束。
- **All-in decoding**：觸發後一次 forward 把剩餘所有 mask 位置以 argmax 全部填入，不再迭代；因此加速來自「省掉後段步數」，額外成本只有每步算一次 gap，overhead 可忽略。
- **Training-free、與其他加速正交**：不改模型、不需微調，可疊在 KV cache、block-wise semi-AR 解碼之上；代價是若 gap 門檻設太低，會在上下文尚未完整時提早鎖定錯誤答案。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| LLaDA-8B-Instruct | 8B | 主要實驗模型，steps=256、gen_length=256、block_length=32 |
| Dream-7B（Dream-v0-Instruct-7B） | 7B | 次要驗證模型 |

## PPA / 效能數據

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| 解碼步數減少 | up to 3.4x | 完整 256 步解碼 | LLaDA-8B / Dream-7B，多任務（論文摘要） |
| 加速（planning 類任務） | up to 2.67x | 完整解碼 | 官方 README（GPU 未載明） |
| 加速（一般任務） | up to 2.34x | 完整解碼 | 官方 README |
| 一半步數即正確的樣本比例 | GSM8K 97%、MMLU 99% | — | LLaDA-8B 分析實驗 |
| 準確度變化 | 「維持或略微提升」 | 完整解碼 | 論文敘述；逐項數字未查到 |
| 額外 overhead | negligible | — | 每步僅計算 top-2 gap |

## 限制 / 備註

- 官方 README 的加速數字（2.67x / 2.34x）與摘要的「步數 up to 3.4x」口徑不同（步數 vs. wall-clock），引用時須區分。
- 門檻是 logit gap 的絕對值，對不同模型 / 溫度可能需重新調整；需要指定「答案位置」（`answer_start_pos`）來監控，對開放式生成不一定好界定。
- 分析與加速主要針對「答案短、推理長」的任務（GSM8K、MMLU），對長篇自由生成的效果論文著墨較少。

## 與其他論文的關係

- 與 **TACG**（2607.03236）同屬「何時 commit」的問題，TACG 引用 Prophet 並改用跨步 trajectory 訊號（EMA logits + history gate）而非單步 gap。
- 與 **Fast-dLLM** 的 confidence-threshold 平行解碼互補：Fast-dLLM 決定「每步 unmask 幾個」，Prophet 決定「何時停止迭代、一次全填」。
- 與 **LSP**（Longest Stable Prefix, 2603.05454）共享第一作者 Pengxiang Li，LSP 從 KV cache 連續性角度改進 commit 排程。
- 與 **dParallel / Learn2PD** 這類需要訓練的步數壓縮方法相比，Prophet 完全 training-free，但加速上限較低。
