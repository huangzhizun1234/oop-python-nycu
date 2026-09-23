# Dream 7B: Diffusion Large Language Models

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2508.15487`（模型與 blog 2025-04 先釋出，論文 2025-08） |
| 作者 / 單位 | Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, Lingpeng Kong（香港大學 HKU NLP、華為諾亞方舟實驗室） |
| 日期 | 2025-08 |
| 類別 | 模型 |
| 連結 | [arXiv](https://arxiv.org/abs/2508.15487) · [GitHub](https://github.com/DreamLM/Dream) · [Dream-Coder GitHub](https://github.com/DreamLM/Dream-Coder) · [Dream-Coder arXiv](https://arxiv.org/abs/2509.01142) |

## 一句話總結

以 Qwen2.5-7B 的 AR 權重初始化、只用 580B tokens 續訓出的全序列 masked diffusion 模型，在通用/數學/程式任務追平 Qwen2.5 7B、全面超越用 2.3T tokens 從零訓練的 LLaDA 8B，並在 Countdown / Sudoku 等規劃任務大幅領先 AR；推論可用「步數」調節速度-品質。

## 要解決的問題

- LLaDA 證明 dLLM 可 scale，但從零訓練要 2.3T tokens，資料效率遠低於 AR；需要一條「借用 AR 預訓練成果」的路徑。
- 既有 AR→diffusion 轉換（DiffuLLaMA）在 7B 仍與 AR 有差距；問題在於 (1) 如何把因果權重對齊到雙向去噪的架構、(2) masked diffusion 的序列級噪聲排程對「靠近乾淨上下文的 token」給了錯誤的訓練訊號。

## 核心方法與特色

- **AR 權重初始化 + shift 對齊**：架構與 Qwen2.5-7B 完全相同（7B、全注意力），把因果 mask 拿掉改為雙向；模型以「shifted」方式預測遮罩 token（預測位置對應 AR 的 next-token head），使 AR 的輸出頭與位置關係最大程度對齊、權重可直接沿用。論文消融顯示 AR 初始化遠優於從零訓練。
- **Context-adaptive token-level noise rescheduling (CART)**：把序列級的噪聲時間 t 改成 token 級：每個遮罩 token 的有效噪聲依它與最近乾淨 token 的距離（幾何分布 Geo(p)）重新加權，靠近乾淨上下文的 token 視為「噪聲較低」，訓練訊號更接近推論時的實際狀況。
- **580B tokens 續訓 + 1.8M SFT**：資料為 Dolma v1.7、OpenCoder、DCLM-Baseline；SFT 用 Tulu 3 + SmolLM2 的 1.8M 對，3 epochs。訓練成本約為 LLaDA 的 1/4 資料量。
- **推論彈性**：全序列去噪（無 block、無 KV cache），提供 entropy / maskgit_plus / topk_margin 等 remasking 策略；步數可從等於長度降到 5–20 步，在 Countdown 上 5–20 步即同時比 Qwen2.5 7B 更快且更準。支援任意順序生成與 infilling。
- **Dream-Coder 7B（2509.01142）**：同配方、純開源資料（adaptation → SFT → RL）做的程式 dLLM，LiveCodeBench (2410–2505) pass@1 21.4%，並出現依任務自適應的「sketch-first / 左到右 / 迭代」生成模式；DreamOn 解決變長 infilling。
- **代價**：仍是全序列雙向模型，無法直接用 KV cache，長輸出速度慢；MATH 等難數學任務與 AR 差距仍大。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| Dream-v0-Base-7B | 7B（dense, 全注意力） | Qwen2.5-7B 初始化 + 580B tokens |
| Dream-v0-Instruct-7B | 7B | +1.8M SFT |
| Dream-Coder-7B / -Instruct | 7B | 程式版（備註） |
| DreamOn-7B | 7B | 變長 infilling 版（備註） |
| Qwen2.5-7B、LLaMA3-8B、LLaDA-8B | 7–8B | 對照 |

## PPA / 效能數據

> 軟體論文：speedup、throughput (tok/s)、latency、記憶體、benchmark 準確度變化。

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| 續訓 tokens | 580B | LLaDA 8B 2.3T（≈1/4）；Qwen2.5 7B 預訓練 18T | 論文未揭露 GPU 數與時數（未查到） |
| Base MMLU / BBH | 69.5 / 57.9 | Qwen2.5 7B 71.9 / 63.9；LLaDA 8B 65.9 / 47.4 | — |
| Base GSM8K / MATH | 77.2 / 39.6 | Qwen2.5 7B 78.9 / 41.1；LLaDA 8B 70.9 / 30.7；LLaMA3 8B 55.3 / 18.0 | — |
| Base HumanEval / MBPP | 57.9 / 56.2 | Qwen2.5 7B 56.7 / 63.6；LLaDA 8B 32.9 / 39.0 | — |
| Instruct MMLU / GSM8K / IFEval | 67.0 / 81.0 / 62.5 | Qwen2.5-7B-Inst 76.6 / 91.6 / 74.7；LLaMA3-8B-Inst 68.4 / 78.3 / 49.7 | — |
| Instruct MATH | 39.2 | Qwen2.5-7B-Inst 75.5 | 差距最大處 |
| Countdown / Sudoku / Trip planning（Base） | 16.0 / 81.0 / 17.8 | Qwen2.5 7B 6.2 / 21.0 / 3.6；LLaMA3 8B 3.7 / 0.0 / 8.7 | 規劃任務 |
| 速度-品質 | 5–20 diffusion steps 時速度與準確率同時優於 Qwen2.5 7B | Qwen2.5 7B AR | Countdown 任務；論文未給 tok/s 絕對值（未查到） |
| Dream-Coder LiveCodeBench | 21.4% pass@1 | 開源同級 | 2410–2505 題集 |
| 記憶體需求 | ≥20 GB GPU | — | repo 建議 |

## 限制 / 備註

- 論文未提供絕對 throughput（tok/s）與 GPU 時數；後續 Efficient-DLM 量測 Dream 7B 作為 baseline，其 8B 模型比 Dream 7B 快 4.5×。
- 全序列雙向、無 KV cache：Fast-dLLM v2 指出把同樣的 Qwen2.5 轉成 block diffusion 只需 ~1B tokens（比 Dream 少 500×），且能用 KV cache。
- 不同論文重跑 Dream 的數字差異大（例如 GSM8K 77.2 vs 77.79、HumanEval 57.9 vs 57.92），與 LLaDA 一樣需固定評測協定。
- 後續作品：Dream-Coder（程式）、DreamOn（infilling）、Dream-VL / Dream-VLA（2512.22615）。

## 與其他論文的關係

- 建立在 DiffuGPT / DiffuLLaMA（AR→diffusion adaptation, 2410.17891）與 LLaDA 之上，是「AR 初始化 dLLM」路線的代表；LLaDA-MoE、Efficient-DLM、Fast-dLLM v2、SDAR、LLaDA2.0 都以 Dream 7B 為主要對照。
- 與 LLaDA 8B 是最常被同時使用的兩個開源 dLLM baseline（Fast-dLLM、dKV-Cache、DLLMQuant、Prophet、d1 等加速 / RL 論文都同時評測兩者）。
- Efficient-DLM 與 Fast-dLLM v2 直接批評 Dream 的「全雙向 + 大量 tokens」策略，改用 block-wise attention 以保留 AR 權重分布並啟用 KV cache。
- dLLM 框架（2602.22661）把 Dream 列為可重現、微調、評測的核心模型之一。
