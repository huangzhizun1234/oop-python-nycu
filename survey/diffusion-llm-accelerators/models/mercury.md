# Mercury: Ultra-Fast Language Models Based on Diffusion

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2506.17298`（技術報告；產品 2025-02 發布） |
| 作者 / 單位 | Inception Labs：Samar Khanna, Siddhant Kharbanda, Shufan Li, Harshit Varma, Eric Wang, Sawyer Birnbaum, Ziyang Luo, Yanis Miraoui, Akash Palrecha, Stefano Ermon, Aditya Grover, Volodymyr Kuleshov |
| 日期 | 2025-06 |
| 類別 | 模型 |
| 連結 | [arXiv](https://arxiv.org/abs/2506.17298) · [Blog](https://www.inceptionlabs.ai/blog/introducing-mercury) · [API](https://platform.inceptionlabs.ai) |

## 一句話總結

第一個商用規模的擴散 LLM：Mercury Coder Mini / Small 在 NVIDIA H100 上由第三方 Artificial Analysis 實測達 1,109 / 737 tok/s，比 GPT-4o Mini、Claude 3.5 Haiku 等速度優化型 AR 模型快 5–10×，程式 benchmark 品質相當。

## 要解決的問題

- AR 模型的速度有硬性下限：每個 token 一次 forward，200 token 的回答就是 200 次序列計算；即使在 H100 上，速度優化型商用模型也只有 ~200 tok/s，互動式程式補全（IDE）體驗受限。
- 學術 dLLM（LLaDA、Dream）雖能平行生成，但實際 tok/s 反而低於 AR；Mercury 要證明擴散模型在「真實 serving 系統 + 商用品質」下能兌現速度優勢。

## 核心方法與特色

- **Transformer 參數化的離散擴散**：以 Transformer 為骨幹，訓練目標是從噪聲（遮罩）狀態同時預測多個 token；推論是「粗到細」的迭代去噪，每步平行修正整段輸出，因此每次 forward 產生多個 token。報告未揭露參數量、資料、步數等細節。
- **專有推論引擎**：Inception 宣稱建立了為擴散取樣特化的 serving 系統（動態批次、平行去噪的 kernel），這是 1,000+ tok/s 的主要來源；相同 dLLM 架構在開源引擎上（如 LLaDA 原始碼）無法達到此速度。
- **面向程式的兩個尺寸**：Mercury Coder Mini（速度優先）與 Small（品質優先），支援 32K context、fill-in-the-middle（FIM），提供 OpenAI 相容 API 與 playground。
- **獨立評測**：速度由 Artificial Analysis 第三方量測；品質在 HumanEval、MBPP、EvalPlus、MultiPL-E、FIM、LiveCodeBench、BigCodeBench 上與 Claude 3.5 Haiku、Gemini 2.0 Flash-Lite、GPT-4o Mini、Codestral、Qwen2.5-Coder 7B 比較；並以 Copilot Arena 的真實開發者投票驗證。
- **代價**：完全閉源，無法分析其加速設計（是否 block、KV cache、步數）；基準對手為 2024 年的速度型模型，且速度並非在同品質下比較。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| Mercury Coder Mini | 未揭露 | 1,109 tok/s on H100 |
| Mercury Coder Small | 未揭露 | 737 tok/s on H100 |
| 對照：GPT-4o Mini、Claude 3.5 Haiku、Gemini 2.0 Flash-Lite、Codestral 2501、Qwen2.5-Coder 7B、DeepSeek Coder V2 Lite | AR | 速度型商用 / 開源程式模型 |

## PPA / 效能數據

> 軟體論文：speedup、throughput (tok/s)、latency、記憶體、benchmark 準確度變化。

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| Throughput（Mini） | **1,109 tok/s** | 速度優化型 AR 約 200 tok/s 級 → 最高 10×（平均 5–10×） | NVIDIA H100，Artificial Analysis 獨立量測 |
| Throughput（Small） | 737 tok/s | 同上 | 同上 |
| HumanEval（Small） | 90.0 | 與 Claude 3.5 Haiku / Gemini 2.0 Flash-Lite 同級 | — |
| MultiPL-E（Small） | 76.2（C++ 82.0 / JS 83.9 / TS 82.6） | — | 多語言 |
| FIM 平均（Small） | 84.8 | Codestral 2501 82.5 | fill-in-the-middle |
| MBPP / EvalPlus / LiveCodeBench / BigCodeBench | 報告有評測，數值未查到 | — | — |
| Copilot Arena | 品質第 2 名、延遲第 1 名；Mini 回應延遲 25 ms，約為 GPT-4o Mini 的 4× 快 | GPT-4o Mini 等 | 真實開發者投票 |
| API 價格 | $0.25 / M input, $0.75 / M output | — | 2025 定價（第三方筆記） |
| 後續 Mercury 2（2026） | 第三方實測 711–1,196 tok/s，132 個模型中速度第一 | — | 第三方彙整，待確認 |

## 限制 / 備註

- 無架構、參數量、訓練資料、去噪步數等任何可重現資訊；速度是「擴散架構 + 專有 serving 工程」的綜合結果，不能歸因於架構本身。
- 品質比較對象是速度型（小型）商用模型，非前沿模型；社群質疑其對 chain-of-thought / 推理任務的支援。
- 報告的 throughput 為單請求輸出速率，未揭露 batch size 與延遲分佈。
- 對本 survey 的意義：它是「dLLM 能在真實硬體上快 5–10×」的第一個商用證據，也是 Gemini Diffusion、Seed Diffusion 用來對比的速度標竿。

## 與其他論文的關係

- 作者群（Kuleshov、Ermon、Grover）與 MDLM、BD3-LM、Set Diffusion 為同一學術脈絡，Mercury 可視為這些方法的商用化（細節未公開）。
- 與 Gemini Diffusion（1,479 tok/s）、Seed Diffusion（2,146 tok/s on H20）構成閉源 dLLM 三強，三者硬體與量測條件不同，不可直接換算。
- 開源陣營（LLaDA2.0-flash 535 tok/s、SDAR、Fast-dLLM v2、Nemotron-Labs-Diffusion）以其為速度目標；學術加速論文（Fast-dLLM、dInfer）則試圖在開源模型上重現此級別的速度。
