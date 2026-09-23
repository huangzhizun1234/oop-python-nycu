# PSD: Pushing the Pareto Frontier of Diffusion LLMs via Parallel Speculative Decoding

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2605.15609` |
| 作者 / 單位 | Shengyin Sun, Yiming Li, Renxi Liu, Xinqi Li, Hui-Ling Zhen, Weizhe Lin, Chen Chen, Xianzhi Yu, Mingxuan Yuan, Chen Ma（依記憶：City University of Hong Kong 與 Huawei Noah's Ark Lab，待確認） |
| 日期 | 2026-05 |
| 類別 | 平行與投機解碼 |
| 連結 | [arXiv](https://arxiv.org/abs/2605.15609) · GitHub：未查到 |

## 一句話總結

把 dLLM 的兩種加速軸——「空間」（一步 unmask 多個 token）與「時間」（一次驗證跨多個 denoising 步的 draft）——在同一次 forward 裡相乘，training-free 地把 tokens-per-forward 推到 5.5x 且準確度與 greedy 相當。

## 要解決的問題

- 既有方法只走一條軸：Fast-dLLM / WINO 類在空間軸（每步多 unmask），Spiffy / FreeDave 類在時間軸（投機驗證多步）；兩者各自的加速上限不高，且沒有人把它們合在一起。
- 單純把兩者串接會讓驗證 batch 爆炸，需要一個結構化的 draft 與接受規則。

## 核心方法與特色

- **Stage 1 – Spatial parallel unmasking**：用可配置的 transfer policy（信心門檻 / top-k 等）在當前步一次揭露多個 token，得到空間平行度。
- **Stage 2 – Temporal speculative drafting**：從當前狀態出發，依信心由高到低把剩餘位置逐層填入，構造多個不同「深度」的候選未來狀態（第 d 層 = 假設再走 d 步之後的樣子）。這些 draft 直接來自模型當步的預測，不需額外模型。
- **Stage 3 – Batched verification with hierarchical acceptance**：把所有深度的候選狀態疊成一個 batch，一次 forward 得到各自的更新預測；由深到淺檢查，保留「其投機 token 全部與 verifier 更新後預測一致」的最深分支，等於一次 forward 跳過多步。
- **Pareto 觀點**：兩軸的平行度可獨立調整，論文以 TPF–accuracy 的 Pareto 前緣呈現，顯示組合後的前緣整體外推。
- **代價**：batch 內候選數 = 深度數，forward 的 FLOPs 隨之增加；在 GPU 已飽和時，TPF 的提升不會完全轉成 wall-clock 加速。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| 三個 dLLM（論文摘要未點名；依相關工作慣例推測為 LLaDA-8B / Dream-7B / SDAR 系列，待確認） | 7–8B | 推理與程式生成任務 |

## PPA / 效能數據

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| Tokens per forward | up to 5.5x | 逐 token 解碼（TPF=1） | 三個 dLLM，推理與程式任務 |
| 準確度 | 與 greedy 解碼相當 | greedy 逐 token | 論文摘要 |
| Wall-clock 加速 / tok/s / GPU | 未查到 | — | — |

## 限制 / 備註

- 所有具體數字（逐模型、逐 benchmark）未查到；上表僅為摘要層級。
- TPF 與實際 throughput 的差距取決於驗證 batch 大小，論文是否報告 wall-clock 未確認。
- 未查到官方程式碼。

## 與其他論文的關係

- 明確站在 **Spiffy**（時間軸）與 **Fast-dLLM / dParallel**（空間軸）之上，作為兩者的乘積。
- 與 **FreeDave** 的多分支驗證思路相近，但 FreeDave 只驗證一步，PSD 驗證多深度。
- 與 **SimSD**（同月）形成對照：SimSD 用外部小 dLLM draft，PSD 是 self-drafting。
- **BlockPilot**（2606.31315）等後續工作嘗試用學習的策略取代 PSD 的手調 transfer policy。
