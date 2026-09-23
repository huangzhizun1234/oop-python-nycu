# BlockBatch: Multi-Scale Consensus Decoding for Efficient Diffusion Language Model Inference

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2605.29233` |
| 作者 / 單位 | Xiaoyou Wu et al.（Georgia Institute of Technology） |
| 日期 | 2026-05 |
| 類別 | 平行與投機解碼（多 block-size 分支共識） |
| 連結 | [arXiv](https://arxiv.org/abs/2605.29233) · GitHub：未查到 |

## 一句話總結

不再為每個請求選一個固定 block size，而是同時跑多個 block size 的分支，靠信心閘控合併、leader 同步與週期性全序列 KV refresh 讓它們達成共識，training-free 地比 Fast-dLLM 再少 26.6% 步數、快 1.33x。

## 要解決的問題

- Block-wise（semi-AR）dLLM 解碼存在粒度兩難：小 block 保留局部條件、品質好但步數多；大 block 平行度高但容易過早 commit，且近似 KV cache 的誤差會累積。
- 既有加速（Fast-dLLM 等）每個請求只用一個 block size，不同 block size 之間「有時對、有時錯」的互補性完全沒被利用。

## 核心方法與特色

- **Block size 作為分支維度**：對同一請求同時以數個 block size 解碼（batch 內平行），作者觀察到不同 block size 的 KV-cache 軌跡相關但不相同，token 層級會出現「分歧 → 後期共識」的現象。
- **Confidence-gated merging**：各分支在同一位置的預測若一致且信心足夠，直接合併為共識 token，所有分支同步接受；分歧位置則保留為 mask 等待更多上下文。
- **Leader-based synchronization**：以某一分支（通常是進度最快 / 信心最高者）為 leader，其他分支週期性對齊其已 commit 的前綴，避免分支漂移導致 batch 無法共享。
- **Full-sequence KV refresh**：block 內的局部 cache 更新會累積誤差，BlockBatch 週期性做一次全序列 forward 重算 K/V，把所有分支重新錨定到全域一致的狀態，兼顧 cache 速度與正確性。
- **代價**：batch 內有多個分支，單步 FLOPs 增加；加速來自步數減少（26.6%）而非單步變快，故 wall-clock 只有 1.33x。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| LLaDA-1.5-8B | 8B | — |
| LLaDA-8B-Instruct | 8B | — |
| Dream-7B（Base） | 7B | — |

## PPA / 效能數據

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| Denoising 步數 | −26.6% | Fast-dLLM | GSM8K / MATH / HumanEval / MBPP，三個模型平均 |
| 端到端加速 | 1.33x（平均） | Fast-dLLM | 同上 |
| 準確度 | 維持（preserving accuracy） | Fast-dLLM | 逐項數字未查到 |
| GPU / tok/s | 未查到 | — | — |

## 限制 / 備註

- 加速倍率相對溫和（1.33x），且是相對 Fast-dLLM 而非 vanilla；主要價值在揭示 block size 互補性。
- 多分支 batch 佔用更多記憶體與算力，對已用大 batch 的 serving 場景不一定划算。
- 未查到官方程式碼與詳細數字。

## 與其他論文的關係

- 直接建立在 **Fast-dLLM**（block-wise + 近似 KV cache）之上，KV refresh 是對其 cache 誤差的修補。
- 與 **Swordsman**（2602.04399）都在處理「block 怎麼切」：Swordsman 依 entropy 自適應切一個分割，BlockBatch 同時跑多個分割取共識。
- 與 **SemBlock**（2606.04964）、**Multi-Block Diffusion LMs**（2606.29215）同屬 2026 年的 block 粒度研究。
- 「多分支共識」的想法與 **FreeDave / PSD** 的 batch 驗證有形式上的相似，但目的不同（多尺度 vs 多深度）。
