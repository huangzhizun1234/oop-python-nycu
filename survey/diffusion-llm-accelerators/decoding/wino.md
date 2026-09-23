# Wide-In, Narrow-Out: Revokable Decoding for Efficient and Effective DLLMs (WINO)

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2507.18578`（ICLR 2026） |
| 作者 / 單位 | Feng Hong et al.（Semantic Scholar 顯示末位作者姓 Yu；單位依記憶為上海交通大學等，待確認） |
| 日期 | 2025-07 |
| 類別 | 平行與投機解碼（可撤銷解碼） |
| 連結 | [arXiv](https://arxiv.org/abs/2507.18578) · [GitHub](https://github.com/Feng-Hong/WINO-DLLM) |

## 一句話總結

讓 dLLM 的解碼「可撤銷」：每步大膽 draft 很多 token（wide-in），同時用雙向上下文驗證並把可疑的 token 重新 mask 回去（narrow-out），training-free 地同時拿到 6–10x 加速與更高準確度。

## 要解決的問題

- 傳統 dLLM 解碼一旦 unmask 就永久固定，因此為了安全只能每步解少量 token（保守 → 慢），或平行解很多但錯誤無法回收（激進 → 品質崩）。
- 早期步驟的上下文（大量 mask）品質很差，此時做出的高信心預測很多其實是錯的，但既有方法不會回頭修。
- 需要一種能在「大量平行解碼」與「錯誤修正」之間解耦的機制。

## 核心方法與特色

- **Wide-in（投機式 drafting）**：每步用較寬鬆的門檻一次 unmask 多個 token，主動製造高平行度，而不是像 confidence-threshold 方法那樣保守。
- **Narrow-out（動態錯誤收縮）**：在同一輪 forward 的另一個分支中，用互補的 mask 配置重新評估已 draft 的 token；當某 token 在更完整的上下文下信心不足或與新上下文矛盾時，將其重新 mask（revoke），留待後續步驟重解。draft 與 verify 併成一個 batch，不增加序列步數。
- **利用雙向注意力做驗證**：dLLM 的雙向 attention 天生能用「後文」檢查「前文」，WINO 把這個能力從生成挪到驗證，這是 AR 模型做不到的。
- **速度與品質同時改善**：因為錯誤可被撤銷，模型可用更激進的平行度；而撤銷後重解的 token 是在更好的上下文下決定的，所以準確度反而上升（GSM8K +2.58%）。
- **代價**：每步多一份驗證分支的計算（batch 變大）；revoke 過多時步數反而增加，需調 draft / revoke 門檻。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| LLaDA-8B-Instruct | 8B | 語言任務：GSM8K、MATH-500、HumanEval、MBPP、Countdown、Sudoku、ARC-E、ARC-C |
| MMaDA | 8B | 多模態 dLLM，六個多模態 benchmark（lmms-eval），含 Flickr30K captioning |

## PPA / 效能數據

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| GSM8K 準確度 | +2.58% | LLaDA-8B-Instruct 標準解碼 | 同時 6.10x TPS 加速 |
| GSM8K 解碼步數 | 減少 > 6x | 標準解碼 | LLaDA-8B-Instruct |
| 整體加速 | 6x–10x | LLaDA / MMaDA 標準解碼 | 多任務，同時準確度提升 |
| Flickr30K captioning | up to 10x 加速，CIDEr 超越 baseline | MMaDA 標準解碼 | 多模態 |
| MBPP | 與 LLaDA 持平 | — | 唯一未提升準確度的任務 |
| GPU / 絕對 tok/s | 未查到 | — | — |

## 限制 / 備註

- 加速數字主要以步數 / TPS 相對值呈現，絕對 tok/s 與 GPU 型號未查到。
- 後續延伸：**WINO+**（"Roll Out and Roll Back: Diffusion LLMs are Their Own Efficiency Teachers", 2605.16941）用離線 WINO 軌跡做訓練；**ReMix-DLLM**（CVPR 2026）為同團隊 training-free 後續。
- 較新的 remasking 研究（如 "Re-evaluating Confidence Remasking", 2606.12232）指出 revoke 策略在不同模型上效果不一，需個別調參。

## 與其他論文的關係

- 開創 dLLM「revokable decoding」路線，**STaRR**（2601.04205）與 **TACG** 都是在「何時 remask / 何時 commit」上做更細的動態門檻。
- 與 **Fast-dLLM** 的 confidence-threshold 平行解碼直接競爭：Fast-dLLM 只能前進，WINO 可倒退。
- 與 **DMax**（2604.08302）目標相同（激進平行 + 自我修正），但 DMax 用訓練讓模型在 embedding 空間自我修正，WINO 完全 training-free。
- **FreeDave / Spiffy** 的 draft-and-verify 是 lossless（保證與逐 token 解碼一致），WINO 則是 lossy 但常常更準。
