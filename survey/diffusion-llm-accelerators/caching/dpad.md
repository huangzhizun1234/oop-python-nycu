# DPad: Efficient Diffusion Language Models with Suffix Dropout

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2508.14148`（ICLR 2026） |
| 作者 / 單位 | Xinhua Chen, Sitao Huang, Cong Guo, Chiyue Wei, Yintao He, Jianyi Zhang, Hai "Helen" Li, Yiran Chen（Duke University） |
| 日期 | 2025-08 |
| 類別 | KV cache 與稀疏（token 剪枝） |
| 連結 | [arXiv](https://arxiv.org/abs/2508.14148) · [GitHub](https://github.com/Crys-Chen/DPad) |

## 一句話總結

把 dLLM 尚未解碼的 suffix mask token 視為「草稿紙」（scratchpad），發現其重要性隨距離急劇衰減，因此只保留當前 block 附近一小段 suffix、其餘依距離衰減機率丟棄，訓練無關即可帶來 1.2–4×，與 Fast-dLLM 疊加後最高 61.4× 加速，且 strict-match 準確度反而大幅提升。

## 要解決的問題

- dLLM 每步對所有 suffix mask token（可能上千個）計算注意力與 logits，但這些 token 多數在很久之後才會被解碼，當前步驟的預測會被丟棄。
- 遠端 suffix token 只是把「已解碼 prefix 的訊號」收集起來，資訊高度冗餘，卻佔用 O(L²) 的計算。
- KV cache 方法只減少 prefix 重算，對 suffix 的冗餘計算沒有處理。

## 核心方法與特色

- **Scratchpad 觀點與 Diffusion Lottery Tickets (DLT) 假說**：suffix token 像草稿紙，收集 prefix 的訊號，但重要性隨距離衰減；即使把高注意力的遠端 suffix token 剪掉，模型也會動態把注意力移到附近 token，因此只需一個稀疏子集的 suffix 就足夠。
- **Sliding window**：只保留當前 block 之後固定長度的 suffix 視窗，使每步計算量不隨整體生成長度 L 成長。
- **Distance-decay dropout**：在視窗內以距離為參數的高斯取樣，越遠的 suffix token 越可能被丟棄；丟棄是「在注意力計算前決定」的確定性（seed 固定）策略，並修改 RoPE 以支援非連續位置。
- **與 Fast-dLLM 相乘**：DPad 減少序列長度、Fast-dLLM 減少步數與 prefix 重算，兩者正交，合併後 MBPP 上 LLaDA-1.5 從 62.34s 降到 4.41s（14.1×），1024 token 情境達 61.39×。
- **Strict-match 準確度提升**：去掉遠端 suffix 後模型較少產生冗長 / 格式錯誤的尾巴，LLaDA-1.5 GSM8K strict-match 從 61.87% 升到 78.47%（+16.6 分；論文另報 LLaDA-Instruct +26.46%）。
- **代價**：對需要長距離規劃的生成（例如 suffix 後段需先固定）可能有風險；視窗大小與衰減參數需設定；只實作在 block-wise 解碼上。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| LLaDA-1.5 | 8B | 主要實驗 |
| LLaDA-8B-Instruct | 8B | 論文附加實驗（strict-match +26.46%） |
| Dream-v0-Base-7B | 7B | 主要實驗 |

## PPA / 效能數據

> 軟體論文：speedup、throughput (tok/s)、latency、記憶體、benchmark 準確度變化。GPU：NVIDIA A100-PCIe-80GB；4-shot；latency 為每題延遲（秒）。

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| 最高加速 | 61.39× | Vanilla LLaDA-1.5 | GSM8K, 1024 tokens, 1-shot, DPad + Fast-dLLM（parallel + prefix cache） |
| Dream 最高加速 | 30.58× | Vanilla Dream | HumanEval, 1024 tokens, 0-shot, DPad + parallel |
| LLaDA-1.5 GSM8K | 27.61s → 18.26s（DPad 1.51×）→ 6.23s（+parallel 4.43×）；flexible 80.59→80.89%，strict 61.87→78.92% | Vanilla | 4-shot |
| LLaDA-1.5 MATH | 25.12s → 8.57s（2.93×）；strict 32.72→35.96% | Vanilla | — |
| LLaDA-1.5 HumanEval | 34.80s → 5.26s（6.61×）；40.85→39.63%（DPad 單獨 44.51%） | Vanilla | — |
| LLaDA-1.5 MBPP | 62.34s → 4.41s（14.14×）；38.20→41.60% | Vanilla | — |
| Dream GSM8K | 22.30s → 5.24s（4.25×）；flexible 75.06→74.83% | Vanilla | — |
| Dream HumanEval | 28.49s → 4.06s（7.01×）；51.22→52.44% | Vanilla | — |
| Dream MBPP | 49.15s → 9.86s（4.98×）；52.40→54.80% | Vanilla | — |

## 限制 / 備註

- 61.4× 為 1024 生成長度 + Fast-dLLM 疊加的極端設定；DPad 單獨在 256–512 長度約 1.2–4.2×。
- 高斯距離衰減與視窗長度為手調超參數；對 Dream MBPP 的單獨加速僅 1.19×。
- 「只實作幾行程式」但需修改 RoPE 處理非連續位置，與現有 FlashAttention kernel 的整合需自行處理。

## 與其他論文的關係

- 明確設計為 **Fast-dLLM** 的正交補充（prefix cache + parallel decoding 處理 prefix / 步數，DPad 處理 suffix）。
- **Streaming-dLLM** 進一步把 suffix pruning 與動態解碼 / early exit 結合，宣稱最高 68.2×；**Elastic-Cache** 則以「視窗外 MASK 快取」而非丟棄處理同一冗餘。
- **Sparse-dLLM** 用注意力分數淘汰 suffix，DPad 用位置距離，兩者是「學習到的顯著性 vs. 先驗結構」的對照。
- 「suffix 冗餘、遠端 mask 不重要」也被 **Focus-dLLM**、**d²Cache** 的分析所支持。
