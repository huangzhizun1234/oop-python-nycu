# STaRR: Spatial-Temporal Token-Dynamics-Aware Responsive Remasking for Diffusion Language Models

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2601.04205`（v1 曾以 "STDD: Spatio-Temporal Dynamics-Driven Token Refinement" 為題） |
| 作者 / 單位 | Xinhao Sun, Maoliang Li, Zihao Zheng, Jiayu Chen, Hezhao Xu, Yun Liang, Xiang Chen（依記憶：北京大學，待確認） |
| 日期 | 2026-01 |
| 類別 | 平行與投機解碼（動態 remasking 門檻） |
| 連結 | [arXiv](https://arxiv.org/abs/2601.04205) · [GitHub](https://github.com/Huaijin2005/STaRR) |

## 一句話總結

用每個 token 信心在「時間軸（跨步變異）」與「空間軸（相對鄰居的偏離）」上的動態統計，取代靜態信心門檻來決定該 remask 還是該 commit，training-free 地在 LLaDA / Dream 上平均 4.1x、最高 8.9x 加速且準確度相當。

## 要解決的問題

- dLLM 的 remasking 策略（每步決定哪些位置保留、哪些重新 mask）主導了速度–品質折衷；主流做法用一條固定的信心門檻（如 Fast-dLLM 的 0.9）。
- 靜態門檻忽略了信心的演化：有些 token 信心雖不高但已經穩定多步（其實可以 commit），有些 token 信心暫時很高卻在鄰居更新後會改變（其實該再等），造成大量不必要的 remask → 步數浪費。

## 核心方法與特色

- **Temporal variance**：追蹤每個位置在最近幾步的信心序列，計算變異數；變異小代表預測已收斂，即使絕對信心不高也可以放行。這捕捉了「穩定性」而非「瞬時峰值」。
- **Spatial deviance**：比較某位置信心與同一步中鄰近 / 全序列信心分佈的偏離程度；信心明顯高於周圍的位置更可能是可靠錨點，明顯低於周圍的則優先 remask。
- **Dynamic Threshold Remasking**：把上述兩個指標合成一個逐步、逐 token 的動態門檻，隨解碼進度自動放寬或收緊，而非全程一條線。
- **Responsiveness Optimization**：確保 token 一旦滿足條件就「及時」被 commit（不多等一步），並避免對已穩定 token 重複計算，讓每一步都盡量做有用的 unmask。
- **可與 dual cache 疊加**：官方 repo 提供 STaRR + dual cache（Fast-dLLM 式 prefix/suffix cache）的評測腳本；代價是需維護每個位置的歷史信心（少量記憶體）與統計計算。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| LLaDA-8B-Instruct | 8B | GSM8K / MATH500 / HumanEval / MBPP |
| Dream-v0-Instruct-7B | 7B | 同上 |

## PPA / 效能數據

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| 平均加速 | 4.1x | 原始（逐 token）解碼 | LLaDA-8B-Instruct / Dream-7B，四個 benchmark |
| 最高加速 | 8.9x | 原始解碼 | 論文摘要（對應任務未查到） |
| 準確度 | 「comparable」 | 原始解碼 | 逐項數字未查到 |
| GPU / tok/s | 未查到 | — | — |

## 限制 / 備註

- 加速基準為 vanilla 逐 token 解碼；相對 Fast-dLLM 等 threshold 方法的增益未查到具體數字。
- 需要跨步歷史，第一步沒有時間資訊，早期步驟仍要依賴靜態門檻。
- 官方 repo 由個人帳號維護，README 未含結果表。

## 與其他論文的關係

- 直接針對 **Fast-dLLM** 靜態信心門檻的改良，屬 training-free 動態 remask 家族。
- 與 **TACG**（2607.03236）高度相似（都用跨步 trajectory 訊號），TACG 用 EMA logits 與 history gate，STaRR 用變異數與空間偏離。
- 與 **WINO** 的 revoke 思路同源：WINO 決定「撤銷什麼」，STaRR 決定「門檻怎麼動」。
- 與 **Prophet** 的 top-2 gap 互補：Prophet 看單步的候選差距，STaRR 看多步的穩定性。
