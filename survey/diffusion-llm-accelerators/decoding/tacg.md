# TACG: Trajectory-Aware Commit Gating for Diffusion Language Model Decoding

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2607.03236`（repo 標題另作 "From Confidence to Commitment: ..."） |
| 作者 / 單位 | Chengcheng Wang, Tingzhang Luo, Wenhao Li, Jianyuan Guo, Chang Xu（依記憶：University of Sydney，待確認） |
| 日期 | 2026-07 |
| 類別 | 平行與投機解碼（commit 閘控） |
| 連結 | [arXiv](https://arxiv.org/abs/2607.03236) · [GitHub](https://github.com/Clarence-CV/TACG-DLLM) |

## 一句話總結

把「token 是什麼」與「現在能不能 commit」分開：token 身分永遠取自模型當前 posterior，但 commit 與否由跨步 trajectory 訊號（EMA logits 對比 + 短期持續性閘門）決定，training-free、零額外 forward，在 LLaDA / Dream / LLaDA2-Mini 上減少步數、提高 TPF 且通常不掉分。

## 要解決的問題

- dLLM 每步都輸出一整條預測分佈的「軌跡」，但幾乎所有解碼器只看當前快照就 commit，把「信心高」與「可以定案」混為一談。
- 結果：在上下文不完整時出現的瞬間 top-1 尖峰會被錯誤鎖定；而連續多步都被支持、只是沒衝到門檻的候選卻被拖延。
- 既有修正法（WINO 撤銷、STaRR 動態門檻）仍主要依賴信心值本身。

## 核心方法與特色

- **Temporal Implicit Logits Guidance (TILG)**：維護每個位置 logits 的指數移動平均（EMA）作為「自我參考」，在 natural-parameter（logit）空間把當前 logits 與 EMA 對比（類似 classifier-free guidance 的差分形式），放大「持續被支持」的候選、壓抑「突然冒出」的候選。這只需保存一份 EMA 張量，不需額外模型或 forward。
- **History Gate (HG)**：要求某位置的 top-1 提案在最近若干步保持一致（短期 persistence）才允許 commit；瞬間翻轉的位置會被擋下再等。
- **Capped extra-promotion budget**：每步除了通過閘門的 token，額外允許有限數量（budget）的高分候選提前 commit，以維持平行度、避免閘門太保守拖慢速度。
- **身分錨定於 base posterior**：TILG / HG 只影響「何時 commit」，不改變「commit 什麼」（仍取原始 posterior 的 argmax），因此不會引入 guidance 造成的偏差，是「stability-constrained commit rule」。
- **代價**：需要幾步的暖機才有 trajectory；EMA 衰減率、持續步數、budget 三個超參需調。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| LLaDA-8B-Instruct | 8B | HumanEval / MBPP / GSM8K / MATH500 |
| Dream-v0-Instruct-7B | 7B | 同上 |
| LLaDA2-Mini（LLaDA-2.0-mini） | 16B（MoE） | 同上 |

## PPA / 效能數據

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| 準確度 | 「typically improves or preserves」 | 既有解碼器（含 KLASS 等） | 三模型、四 benchmark；逐項數字未查到 |
| Denoising 步數 | 減少 | — | 數值未查到 |
| Tokens per forward (TPF) | 提高 | — | 數值未查到 |
| 額外 forward / 輔助網路 | 0 | — | 只多一份 EMA logits |
| GPU / tok/s | 未查到 | — | — |

## 限制 / 備註

- 具體加速倍率與準確度數字未查到（論文摘要僅定性描述）。
- 官方 repo 含 KLASS 對照腳本，暗示主要比較對象為 KLASS 類 training-free 解碼器。
- 引用 Prophet 作為相關工作。

## 與其他論文的關係

- 與 **STaRR** 是最接近的競爭者（都是 trajectory-aware、training-free 的 commit / remask 規則）。
- 對 **Prophet** 的「單步 top-2 gap」作出批評式延伸：用多步歷史判斷 readiness。
- 與 **KLASS**（KL-based 平行解碼）、**Fast-dLLM** 同為推論期規則，可直接替換其 commit 判準。
- 「Where and When to Commit: Candidate-Aware Decoding」（2607.28166）為同月類似主題工作。
