# Fast-dLLM v2: Efficient Block-Diffusion LLM

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2509.26328` |
| 作者 / 單位 | Chengyue Wu, Hao Zhang, Shuchen Xue, Shizhe Diao, Yonggan Fu, Zhijian Liu, Pavlo Molchanov, Ping Luo, Song Han, Enze Xie（NVIDIA、香港大學、MIT） |
| 日期 | 2025-09 |
| 類別 | 模型 |
| 連結 | [arXiv](https://arxiv.org/abs/2509.26328) · [GitHub](https://github.com/NVlabs/Fast-dLLM/tree/main/v2) · [Project](https://nvlabs.github.io/Fast-dLLM/v2/) |

## 一句話總結

只用約 1B tokens 微調（比 Dream 少 500×），把 Qwen2.5-Instruct 改造成 block diffusion 模型：block 大小 32、內含 8-token sub-block，以互補式遮罩保留 AR 訓練目標、用「block 級 + sub-block 級」階層式 KV cache 做平行解碼，7B 版吞吐比 Qwen2.5-7B-Instruct 高 2.54× 而準確率相當。

## 要解決的問題

- Fast-dLLM v1 是 training-free 方法（近似 KV cache + 信心平行解碼），只能加速 LLaDA / Dream 這類全序列 dLLM，且加速後仍慢於 AR；根本問題是全雙向 dLLM 架構本身不支援精確 KV cache。
- Dream 式 AR→全雙向轉換需 580B tokens；作者想以「最小改動、最少資料」把現成 AR LLM 變成可平行生成又保有 AR 品質的 block diffusion 模型。

## 核心方法與特色

- **Block diffusion + 互補式注意力遮罩**：把序列切成 32-token block；訓練時 block 間因果、block 內雙向。以「互補遮罩」（complementary mask）為同一 block 產生兩份互補的遮罩模式，讓每個 token 在一個 batch 內既當被預測的目標、又當可見上下文，等於一次訓練覆蓋全部 token，因而 1B tokens 即可收斂。
- **不改架構、保留 token shift**：沿用 Qwen2.5 的 next-token 預測頭（token-shift），使 AR 權重分布幾乎不變；訓練時把噪聲序列與乾淨序列串接（2L×2L 特製 mask），一次 forward 完成。
- **階層式快取（hierarchical cache）**：(1) block-level cache：已完成 block 的 KV 永久快取（與 AR 相同）；(2) sub-block cache：把 32-token block 再切成 8-token sub-block，部分解碼的 block 內以 dual cache 快取已確定的 sub-block，讓 block 內平行去噪也能重用計算。與 v1 不同，v2 在 block 完成後不再更新其 KV。
- **平行解碼 pipeline**：block 內用信心閾值一次接受多個 token，配合 sub-block cache 減少每步的重算量，整體對 AR 達到 2.5× 吞吐。
- **代價**：block 32 / sub-block 8 限制單步平行度；仍需微調（非 training-free）；block 之間串行，失去全域規劃。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| Fast-dLLM v2 1.5B | 1.5B | Qwen2.5-1.5B-Instruct 微調，~1B tokens |
| Fast-dLLM v2 7B | 7B | Qwen2.5-7B-Instruct 微調，~1.3B tokens |
| Qwen2.5-1.5B / 7B-Instruct | 1.5B / 7B（AR） | 基礎與對照 |
| Dream 7B、LLaDA 8B | 7B / 8B | 全序列 dLLM 對照 |
| （Fast-dVLM，2604.06832） | — | 同方法延伸到 VLM，備註 |

## PPA / 效能數據

> 軟體論文：speedup、throughput (tok/s)、latency、記憶體、benchmark 準確度變化。

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| 微調資料量 | ≈1B tokens（1.5B 版）/ ≈1.3B（7B 版） | Dream 580B → **500× 少** | 64×A100，8 h（1.5B）/ 12 h（7B），context 2048，batch 256 |
| Block / sub-block | 32 / 8 tokens | — | — |
| 吞吐加速 | **2.54×**（7B）；「up to 2.5×」 | Qwen2.5-7B-Instruct（AR） | 第三方筆記：217.5 vs 39.5 tok/s @ bs=1（與 2.54× 不一致，待確認） |
| 7B benchmark | HumanEval 63.4、GSM8K 83.7、MMLU 66.6 | Qwen2.5-7B-Instruct 相當（「comparable accuracy」） | GitHub v2 README 節錄 |
| 7B 平均分 | 60.3 | Qwen2.5-7B-Instruct 58.2（第三方筆記，待確認） | HumanEval / MBPP / GSM8K / MATH / IFEval / MMLU |
| 準確率損失 | 「without compromising generation quality」 | — | — |
| v1 對照（LLaDA-8B, training-free） | 8.1×（GSM8K 79.3→78.5）至 27.6× | LLaDA 原始 6.7 tok/s | 供對比，非本文 |

## 限制 / 備註

- 官方頁與 GitHub 未列出完整 tok/s 表與 GPU 條件；第三方整理的 217.5 / 39.5 tok/s 與論文 2.54× 不吻合，可能來自不同 batch / 長度設定。
- block 32 內部的平行度受信心閾值限制，長推理輸出時每 block 仍需多步。
- 與 Efficient-DLM（同 NVIDIA 團隊、同年 12 月）相比，Fast-dLLM v2 資料量少但基礎模型較舊（Qwen2.5），Efficient-DLM 以 Qwen3 + 300–500B tokens 換得更高加速（4.5× vs Dream）。
- 方法被延伸到視覺語言模型（Fast-dVLM，最高 6.18× vs AR）。

## 與其他論文的關係

- 是 Fast-dLLM（v1，2505.22618，training-free KV cache + 平行解碼）的續作：v1 在推論期近似，v2 改在訓練期把模型變成天然支援 KV cache 的 block diffusion。
- 架構承自 BD3-LM（block diffusion、2L×2L 訓練 mask、clean/noisy 串接）；與 SDAR、LLaDA2.0、Efficient-DLM 同屬 AR→block diffusion 轉換，Fast-dLLM v2 是其中資料量最少的。
- 互補遮罩與「一次 forward 覆蓋所有 token」的思想被 Efficient-DLM、Stable-DiffCoder 等後續工作引用。
- dLLM 框架（2602.22661）整合了 Fast-dLLM 的 cache 與信心閾值解碼作為通用推論加速選項。
