# Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2505.22618`（ICLR 2026；v1/v2 一併接受） |
| 作者 / 單位 | Chengyue Wu, Hao Zhang, Shuchen Xue, Zhijian Liu, Shizhe Diao, Ligeng Zhu, Ping Luo, Song Han, Enze Xie et al.（NVIDIA / HKU / MIT） |
| 日期 | 2025-05 |
| 類別 | KV cache 與稀疏（＋平行解碼） |
| 連結 | [arXiv](https://arxiv.org/abs/2505.22618) · [GitHub](https://github.com/NVlabs/Fast-dLLM) · [Demo](https://fast-dllm.hanlab.ai/) |

## 一句話總結

以「block-wise 近似 KV cache（含 prefix + suffix 的 DualCache）」加上「信心閾值驅動的平行解碼」，在不重新訓練的前提下把 LLaDA / Dream 的吞吐量最高提升 27.6×，準確度損失僅 1–2 分。

## 要解決的問題

- dLLM 使用雙向注意力，每個 denoising step 都要把整段（prompt + 全部 mask token）重算一次，無法像 AR 模型那樣直接沿用 KV cache，因此開源 dLLM 實際推論速度反而比 AR 慢。
- dLLM 理論上可一次解多個 token，但「同時解 k 個 token」時各 token 的邊際分布獨立取樣，會破壞 token 間的相依關係（condition independence 問題），k 越大品質越差。
- 需要一個 training-free 的方案同時解決「每步重算」與「多 token 解碼品質下降」。

## 核心方法與特色

- **Block-wise approximate KV cache（PrefixCache）**：把生成長度切成 block（例如 32 token），在解某個 block 期間，prompt 與已完成 block 的 KV 被凍結重用，只在 block 邊界重算一次。依據是相鄰 step 間 KV 的 cosine similarity 極高（接近 1），所以「近似」而非精確的 cache 幾乎不影響輸出；代價是每個 block 邊界仍需一次 full forward。
- **DualCache**：除了 prefix，連「尚未輪到的 suffix mask token」的 KV 也一併快取（因為 mask token 的 KV 在 block 內幾乎不變），使每步實際計算只剩當前 block，速度比只快取 prefix 再進一步提升，長序列時效果更明顯。
- **Confidence-aware parallel decoding**：每步不固定解 k 個 token，而是把所有 softmax 信心值超過全域閾值 τ（常用 0.9）的 token 一次解出（至少解 1 個）。作者給出理論結果：當各 token 信心都超過 1−ε 時，用邊際分布同時解碼與依序聯合解碼的結果是一致的（在小 ε 下等價），因此高信心 token 平行解碼幾乎不損品質。
- **兩者相乘**：KV cache 單獨約 3.2–3.6× 加速，平行解碼單獨約 2.5–5.8×，合併後在 LLaDA GSM8K（512 tokens）達 11×，長 prompt + 長生成（8-shot、1024 tokens）達 27.6×。
- **代價 / 限制**：block 內 KV 是近似值；閾值 τ 過低會出現品質下降；需在 block 邊界重算，因此加速上限受 block size 與生成長度影響；batch=1 的評測條件下數字最好。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| LLaDA-8B-Instruct | 8B | 主要實驗模型，GSM8K / MATH / HumanEval / MBPP |
| Dream-v0-Base-7B | 7B | 第二個驗證模型 |

## PPA / 效能數據

> 軟體論文：speedup、throughput (tok/s)、latency、記憶體、benchmark 準確度變化。所有實驗在單張 NVIDIA A100 80GB 上進行（batch size 1）。

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| 最高吞吐提升 | 27.6×（19.3 tok/s） | LLaDA 原始實作 | LLaDA-8B-Instruct, GSM8K 8-shot, gen length 1024, A100 |
| 對應準確度 | 76.0%（基準 77.3%） | LLaDA 原始 | 同上 |
| KV cache 單獨加速 | 約 3.2–3.6× | 無 cache | LLaDA / Dream，256–512 tokens |
| 平行解碼單獨加速 | 約 2.5–5.8× | 逐 token 解碼 | 閾值 τ=0.9 |
| 合併加速（LLaDA） | 11×（GSM8K, 512）；9.2×（MBPP, 512） | LLaDA 原始 | KV cache + parallel |
| 合併加速（Dream-Base） | 5.6×（GSM8K, 512）；7.8×（MBPP, 512） | Dream 原始 | KV cache + parallel |
| 準確度變化 | 多數 benchmark 在 1–2 分以內 | 原始模型 | GSM8K / MATH / HumanEval / MBPP |
| LLaDA GSM8K 256 | 基準約 6.7 tok/s → 約 54 tok/s（約 8×），準確度 79.3→78.5（依記憶，待確認） | — | 5-shot, gen 256, block 32 |

## 限制 / 備註

- 「27.6×」是最佳條件（長 prompt + 1024 生成長度）的數字；一般 256 生成長度下的整體加速約 5–10×。
- 近似 KV cache 在 block 邊界仍需 full recompute；block size 太大會累積誤差、太小則 cache 效益低。
- 信心閾值策略在低信心 token 多的任務（開放式生成）上會退化成接近逐 token 解碼。
- 後續 Fast-dLLM v2（block diffusion、Qwen2.5 backbone，需微調）與 Fast-dVLM / Fast-dDrive 皆在同一 repo，本檔只涵蓋 v1。

## 與其他論文的關係

- 與 **dKV-Cache**、**dLLM-Cache** 同期（2025-05/06）提出 dLLM 的 KV / feature cache，三者被後續幾乎所有 caching 論文當作 baseline。
- **DPad**、**Streaming-dLLM**、**Elastic-Cache**、**d²Cache** 等都直接建立在 Fast-dLLM 的 block-wise cache + 信心閾值平行解碼之上，並宣稱在其上再疊加加速（例如 DPad 與 Fast-dLLM 相乘達 61×）。
- 信心閾值平行解碼（threshold decoding）後來成為 dLLM 推論的預設策略之一，與 KV cache 在 LLaDA 官方 repo 及多數 serving 系統（如 dInfer、LMDeploy dLLM 支援）中被採用。
