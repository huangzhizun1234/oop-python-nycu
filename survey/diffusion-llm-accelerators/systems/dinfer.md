# dInfer: An Efficient Inference Framework for Diffusion Language Models

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2510.08666` |
| 作者 / 單位 | Yuxin Ma et al.（Ant Group / inclusionAI） |
| 日期 | 2025-10（v0.1 開源）；2025-12 釋出 v0.2（支援 block diffusion LLaDA2.0、FP8、SGLang 整合） |
| 類別 | 系統與 serving |
| 連結 | [arXiv](https://arxiv.org/abs/2510.08666) · [GitHub](https://github.com/inclusionAI/dInfer) |

## 一句話總結

dInfer 把 dLLM 推論拆成「模型 / 迭代管理器 / 解碼策略 / KV-cache 管理器」四個可抽換模組，在每個模組各放一個新演算法（iteration smoothing、hierarchical / credit decoding、vicinity KV refresh），再配上 TP+EP、torch.compile、CUDA Graph 與 loop unrolling 等系統優化，讓 LLaDA-MoE 在 8×H800、batch size 1 下 HumanEval 超過 1,100 tok/s，約為 Fast-dLLM 的 10 倍。

## 要解決的問題

- 開源 dLLM（LLaDA、LLaDA-MoE、LLaDA2.0）愈來愈多，但缺乏一個像 vLLM/SGLang 那樣「標準化、可擴充、效率高」的推論框架；既有的 Fast-dLLM 等是研究型程式碼，演算法與系統層耦合在一起，不容易組合不同的快取、解碼與迭代策略。
- dLLM 每一步 denoising 都要對整段序列做一次 forward，且雙向注意力使 KV cache 會逐步 stale；若每次都 refresh 全部 cache 就退化成 prefill 成本，若不 refresh 又會掉精度。
- batch size 1 時 GPU 利用率極低，單卡 dLLM 難以打敗高度優化的 AR 模型（例如 Qwen2.5-3B on vLLM）。

## 核心方法與特色

- **四模組解耦架構**：`Model`（LLaDA / LLaDA-MoE / LLaDA2.0 等）、`Diffusion Iteration Manager`（控制 denoising 迴圈與 block 進度）、`Decoding Strategy`（決定每步 unmask 哪些 token）、`KV-Cache Manager`（決定何時、對哪些位置 refresh cache）。每個模組獨立實作，同一套演算法可以套到不同模型，方便做 A/B 比較與組合。
- **Iteration smoothing（迭代平滑）**：在相鄰兩次 denoising 迭代之間傳遞前一步的機率資訊（而非只傳 hard token），讓尚未 unmask 的位置也能利用前一步的「軟」預測，使收斂更平順、可在更少迭代內達到相同品質。
- **Hierarchical decoding 與 credit decoding（階層式 / 信用解碼）**：hierarchical decoding 把整個 block 以「先粗後細」的層級方式挑選 unmask 位置，避免相鄰 token 同時被高信心 unmask 造成的依賴衝突；credit decoding 則記錄每個位置在多次迭代中「持續預測同一 token」的 trace credit，累積足夠信用就提前 commit，減少總迭代數。兩者都屬 training-free 平行解碼。
- **Vicinity KV-cache refresh（鄰近刷新）**：不再每個 block 都 refresh 整段 KV cache，而是只重新計算「當前解碼 block 附近」的 KV（cache staleness 主要來自鄰近位置），其餘沿用舊值；以少量精度風險換取大幅減少 refresh 的 prefill-like 計算。
- **系統層優化**：支援 Tensor Parallel 與 Expert Parallel（讓 MoE 在 batch size 1 也能吃滿 8 張卡）、PyTorch compile 與 CUDA Graph 去除 kernel launch 開銷，並提出 **loop unrolling** 把多個 diffusion 迭代展開到同一個 CUDA stream 內，消除迭代之間的 stream bubble。v0.2 進一步加入 block-diffusion 模型（LLaDA2.0）的 batch 推論、FP8 量化與 SGLang 後端整合。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| LLaDA-MoE-7B-A1B (Base / Instruct) | 7B（1B active） | 論文主要評測模型，MoE 用 EP |
| LLaDA-8B (Base / Instruct) | 8B | 框架支援 |
| LLaDA-1.5 | 8B | 框架支援 |
| LLaDA2.0-mini (-preview / -CAP) | 16B MoE | v0.2 起支援（block diffusion） |
| LLaDA2.0-flash (-preview / -CAP) | 100B MoE | v0.2 起支援；README 提供 8×H20 數據 |
| Qwen2.5-3B（AR 對照，跑 vLLM） | 3B | 對照組 |

## PPA / 效能數據

> 軟體論文：speedup、throughput (tok/s)、latency、記憶體、benchmark 準確度變化。

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| Throughput（HumanEval） | > 1,100 tok/s | — | LLaDA-MoE，8×H800，batch size 1（abstract 數字；委託方提供之「1,011 tok/s vs Fast-dLLM 91 tok/s」疑為早期版本數字，依記憶，待確認） |
| Throughput（六個 benchmark 平均） | > 800 tok/s | — | LLaDA-MoE，8×H800，batch size 1 |
| Speedup vs Fast-dLLM | 約 10× | Fast-dLLM（同一 LLaDA-MoE） | 相同模型、相近準確度 |
| Speedup vs AR | 約 2–3× | Qwen2.5-3B on vLLM | 品質相當 |
| LLaDA2.0-flash-CAP 平均 TPS | 580.70（bs=1）/ 1,966.34（bs=32） | — | 8×H20，gen_len=1000（GitHub README） |
| LLaDA2.0-flash-CAP HumanEval | 753.10 / 2,558.51 tok/s | — | bs=1 / bs=32，8×H20 |
| LLaDA2.0-flash-CAP GSM8K | 591.90 / 2,111.79 tok/s | — | bs=1 / bs=32，8×H20 |
| LLaDA2.0-flash-CAP MBPP | 773.00 / 2,262.45 tok/s | — | bs=1 / bs=32，8×H20 |
| LLaDA2.0-flash-CAP IFEval | 222.60 / 931.89 tok/s | — | bs=1 / bs=32，8×H20 |
| LLaDA2.0-flash-CAP CruxEval-O | 562.90 / 1,967.08 tok/s | — | bs=1 / bs=32，8×H20 |
| 準確度變化 | 「無明顯損失」 | 原始 LLaDA-MoE | 論文未提供逐項數字（未查到） |

## 限制 / 備註

- 1,100 tok/s 是 **8 張 H800 服務一個 batch=1 請求** 的數字（靠 TP+EP 榨出 batch-1 利用率），不是單卡吞吐，也不是多請求 serving 吞吐；與 Fast-dLLM 的 10× 差距一部分來自系統工程（CUDA Graph、compile、loop unrolling）而非純演算法。
- vicinity refresh 與 credit decoding 都是近似方法，在長輸出或高 threshold 下可能有精度風險；論文宣稱「similar accuracy」，但逐 benchmark 的 accuracy 表未在搜尋摘要中取得。
- v0.1 針對 full-attention LLaDA / LLaDA-MoE；block diffusion（LLaDA2.0）在 v0.2 才支援，且 v0.2 後部分功能改走 SGLang 後端。
- 8×H20 的 LLaDA2.0-flash 數字沒有對照基準（README 未給 Fast-dLLM / vLLM 對照）。

## 與其他論文的關係

- 直接以 **Fast-dLLM**（KV cache + threshold parallel decoding）為主要對照，並吸收其 dual cache / threshold decoding 概念，再加上 iteration smoothing、hierarchical / credit decoding（credit decoding 同期亦有 CreditDecoding 論文 arXiv 2510.06133）。
- 是 Ant Group **LLaDA-MoE / LLaDA2.0** 系列的官方推論引擎；LLaDA2.0 技術報告所稱「2.1× 推論加速、LLaDA2.0-flash-CAP 535 tok/s」即基於 dInfer + SGLang 的客製引擎。
- 與 **SGLang dLLM 框架**（見 `sglang-dllm.md`）互補：dInfer 專注 batch-1 低延遲與演算法組合，SGLang 專注多請求 serving；v0.2 起兩者整合。
- 與 **dLLM-Serve、Sangam、BlockServe** 等 serving 系統屬競爭/互補關係：後者關注多請求排程與記憶體，dInfer 關注單請求的 kernel 與演算法效率。
