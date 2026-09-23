# Efficient-DLM: From Autoregressive to Diffusion Language Models, and Beyond in Speed

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2512.14067` |
| 作者 / 單位 | Yonggan Fu, Lexington Whalen, Zhifan Ye, Xin Dong, Shizhe Diao, … , Song Han, Yingyan (Celine) Lin, Pavlo Molchanov（NVIDIA；合作 Georgia Tech、MIT） |
| 日期 | 2025-12 |
| 類別 | 模型 |
| 連結 | [arXiv](https://arxiv.org/abs/2512.14067) · HF: nvidia/Efficient-DLM-4B、-8B（2026-05 釋出） |

## 一句話總結

系統性研究 AR→dLLM 轉換的注意力模式與遮罩策略，發現「保留 AR 預訓練權重分布」是關鍵：採 block 間因果、block 內雙向的 block-wise attention 續訓 Qwen3，加上位置相依（越靠後遮罩機率越高）的遮罩排程，得到 Efficient-DLM 8B，比 Dream 7B 準確率高 5.4% 且吞吐高 4.5×、比 Qwen3 4B 高 2.7% 且快 2.7×。

## 要解決的問題

- dLLM 從零訓練的學習效率落後 AR；既有 AR→dLLM 轉換（Dream 的全雙向、Fast-dLLM v2 的 block）缺乏系統性比較，不清楚哪種注意力模式、哪種遮罩分布最能保留 AR 能力。
- 訓練-測試落差：訓練時遮罩均勻分布在整段，測試時（尤其 block 解碼）遮罩集中在右側、生成高度左到右，導致模型在推論分布下表現不佳、需要更多步。

## 核心方法與特色

- **注意力模式的受控比較**：比較全雙向、block-wise 等多種模式後發現，全雙向會大幅改變預訓練 AR 權重的分布（需大量資料修復），而 block-wise attention（block 間因果 + block 內雙向、每個 block 以乾淨上下文為條件）最能保留權重分布，同時天然支援 KV cache，「準確率與效率雙贏」。
- **Block-wise continuous pretraining**：以 Qwen2.5-1.5B（block 16）、Qwen3-4B / 8B（block 64）為起點續訓 300B（1.5B / 4B）或 500B（8B）tokens，128×H100；移除 AR 的 token-shift（不同於 Fast-dLLM v2）。
- **位置相依遮罩（position-dependent masking）**：訓練時對 block 內較後面的 token 給更高遮罩機率（half-life 比例 λ = 0.1 的衰減），模擬推論時「前面已解碼、後面仍遮罩」的分布，縮小 train-test gap，讓每步能自信接受更多 token。
- **設計空間研究**：論文報告了 dLLM 的注意力模式、訓練動態、block 大小、資料量等消融，作為可擴展 AR→dLLM 轉換的「原則」而非單一模型。
- **代價**：需 300–500B tokens 續訓（比 SDAR 50B、Fast-dLLM v2 1B 多），block 64 較大使短輸出的 KV cache 效益有限；「4.5× vs Dream」是與慢的全序列 dLLM 比，對 AR（Qwen3 4B）是 2.7×。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| Efficient-DLM 1.5B | 1.5B | Qwen2.5-1.5B 續訓 300B tokens，block 16 |
| Efficient-DLM 4B | 4B | Qwen3-4B 續訓 300B tokens，block 64 |
| Efficient-DLM 8B | 8B | Qwen3-8B 續訓 500B tokens，block 64 |
| Qwen3 4B / 8B | AR 對照 | — |
| Dream 7B | 全序列 dLLM 對照 | — |

## PPA / 效能數據

> 軟體論文：speedup、throughput (tok/s)、latency、記憶體、benchmark 準確度變化。

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| 8B vs Dream 7B | 準確率 **+5.4%**，吞吐 **4.5×** | Dream 7B | 論文摘要 |
| 8B vs Qwen3 4B | 準確率 +2.7%，吞吐 2.7× | Qwen3 4B（AR） | 論文摘要 |
| 8B vs Qwen3 8B | 準確率相當（略高） | Qwen3 8B | 論文摘要 |
| 8B 平均準確率 | 71.62（TPF = 1.0 設定） | — | 第三方筆記 |
| 8B 各項 | GSM8K 69.22；HumanEval 67.36；MMLU 77.22（Minerva Math 亦記為 77.22，疑為筆記重複，待確認） | — | 第三方筆記 |
| 8B 吞吐 | 39.99 tok/s | — | batch = 1（第三方筆記；GPU 未註明） |
| 續訓資料 | 300B / 500B tokens | Dream 580B；SDAR 50B；Fast-dLLM v2 ~1B | 128×H100 |
| 遮罩 half-life | λ = 0.1 | 均勻遮罩 | 消融 |

## 限制 / 備註

- 論文本身的 tok/s 與 GPU 條件未能從可取得來源確認；第三方整理的 39.99 tok/s @ bs=1 相對於 Dream 的原始速度合理，但絕對值遠低於 Mercury / LLaDA2.0 級別，說明 4.5× 是相對慢基準。
- 續訓成本高（500B tokens），不像 Fast-dLLM v2 那樣「幾乎免費」；適合有大算力的機構。
- 模型至 2026-05 才在 HF 開源（nvidia/Efficient-DLM-4B/8B）。
- 同團隊後續 Nemotron-Labs-Diffusion（2607.05722）把本文的 block-wise 目標與 AR 目標聯合訓練，並加入 self-speculation。

## 與其他論文的關係

- 同 NVIDIA 團隊（Fu、Diao、Han、Molchanov）在 Fast-dLLM v2 之後的「原則化」研究：v2 證明可行，本文回答「為什麼 block-wise 比全雙向好」。
- 與 Dream（全雙向轉換）是直接對照與批評對象；與 SDAR、LLaDA2.0 同屬 AR→block diffusion 轉換路線，差異在資料量與 block 大小（64）。
- 位置相依遮罩與 LLaDA2.0 的 CAP、Seed Diffusion 的 on-policy 軌跡優化同屬「在訓練期縮小 train-test 解碼落差」的方法族。
- 是 Nemotron-Labs-Diffusion 與 TiDAR（2511.08923，think in diffusion / talk in AR）的前置工作。
