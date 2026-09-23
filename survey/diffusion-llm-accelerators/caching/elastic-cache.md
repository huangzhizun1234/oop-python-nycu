# Attention Is All You Need for KV Cache in Diffusion LLMs（Elastic-Cache）

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2510.14973`（ICLR 2026） |
| 作者 / 單位 | Quan Nguyen-Tri, Mukul Ranjan, Zhiqiang Shen（VILA Lab, MBZUAI） |
| 日期 | 2025-10 |
| 類別 | KV cache 與稀疏 |
| 連結 | [arXiv](https://arxiv.org/abs/2510.14973) · [GitHub](https://github.com/VILA-Lab/Elastic-Cache) · [Project page](https://vila-lab.github.io/elastic-cache-webpage/) |

## 一句話總結

以「最被關注 token 的注意力漂移」決定何時刷新 KV cache、以「深層先變」決定從哪一層開始刷新，配合滑動視窗與視窗外 MASK cache，在 LLaDA 系列上達到 GSM8K 8.7×、長序列最高 45.1× 的加速且準確度持平或提升。

## 要解決的問題

- 既有 dLLM cache（Fast-dLLM、dKV-Cache、dLLM-Cache）都用固定週期（每 block、每 k 步）刷新，與實際 KV 漂移無關：刷新太頻繁浪費、太少則品質下降。
- 刷新時一律重算所有層，但實證上淺層 KV 幾乎不變，只有深層明顯漂移。
- 遠端的 MASK token 幾乎不被注意，卻每步都參與計算。

## 核心方法與特色

- **三個實證觀察**：(1) 遠離當前解碼視窗的 MASK token 得到的注意力極低；(2) KV 狀態的漂移隨層深增加，淺層幾乎穩定；(3) 「最被注意的 token」（most-attended token）的注意力變化是整體 KV 漂移的好指標且變化平穩。
- **When：attention-aware drift test**：每步追蹤 `track_num` 個最被注意 token，若它們的注意力分布相對快取時的變化超過閾值（threshold / gamma），才觸發刷新；否則整步沿用快取。這讓刷新頻率隨內容自適應，簡單任務可以連續數十步不刷新。
- **Where：depth-aware selective refresh**：刷新時不重算全部層，而是從某個選定的層開始向深層重算，淺層 cache 直接重用，節省大量 FLOPs。
- **Sliding-window decoding + off-window MASK cache**：只對當前視窗（window_size）內的 token 做完整注意力，視窗外的 MASK token 的 KV 直接快取重用（因其幾乎不被注意），可與 block caching 合併。
- **Training-free、架構無關**：不改權重、不需微調，同一套超參數可套在 LLaDA-Instruct、LLaDA-1.5、LLaDA-V（多模態）與 Dream。
- **代價**：需每步計算並比較注意力統計（額外少量開銷）；閾值 / gamma 決定加速—品質權衡，需要調參；作者事後更正評測，指出原始 GSM8K / MATH 吞吐略被高估。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| LLaDA-8B-Instruct | 8B | 主要實驗 |
| LLaDA-1.5 | 8B | 加速最大（45.1×） |
| LLaDA-V | 8B | 多模態 dLLM |
| Dream-7B | 7B | README 表格包含 |

## PPA / 效能數據

> 軟體論文：speedup、throughput (tok/s)、latency、記憶體、benchmark 準確度變化。GPU 型號未查到。

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| GSM8K 加速 | 8.7× | 原始 LLaDA | LLaDA-Instruct, gen 256 |
| 長序列最高加速 | 45.1× | 原始 LLaDA-1.5 | LLaDA-1.5, GSM8K, gen 512 |
| HumanEval 加速 | 4.8×（README 稱最高約 5×） | 原始 LLaDA | 程式碼生成 |
| LLaDA-Instruct GSM8K 512 | 90.1 tok/s（25.2×），77.71% | Fast-dLLM 44.0 tok/s @ 74.83% | arXiv v1 數字 |
| GSM8K 準確度（修正後） | 82.79% vs 81.35% baseline | 原始 LLaDA | README 更正後：準確度更高、吞吐略低於原報告 |
| 修正後 GSM8K 512 加速 | 最高 16.0× | 原始 LLaDA | README |
| 加速範圍（各設定） | 約 1.5×–16× | 原始模型 | 256 / 512 tokens、GSM8K / MATH / HumanEval |

## 限制 / 備註

- README 明確聲明原論文 GSM8K / MATH 的吞吐數字略被高估（評測小錯誤），修正後準確度更高但吞吐較低；引用時建議以修正後（最高 16×、45.1× 為 arXiv v1 數字）為準並註明。
- 加速倍率高度依賴 baseline 的低效實作（無 cache 的 LLaDA）；相對 Fast-dLLM 的實際增益約 2×。
- 需調整 window_size / threshold / gamma / track_num。

## 與其他論文的關係

- 直接以 **Fast-dLLM**（block-wise 固定刷新）與 **dKV-Cache / dLLM-Cache**（固定週期 / 固定比例）為對照，主張「用注意力判斷何時刷新」優於固定週期。
- 「視窗外 MASK 幾乎不被注意」的觀察與 **DPad**、**Streaming-dLLM** 的 suffix dropout / pruning 一致，Elastic-Cache 選擇快取而非丟棄。
- 「深層先變、淺層重用」與 **ES-dLLM**（early layers 跳過不重要 token）互為對偶：一個從深層刷新、一個在淺層跳過。
- **DyLLM**、**d²Cache** 同屬自適應 token 級刷新；後續 Elastic-dLLM（同實驗室）延伸至上下文壓縮。
