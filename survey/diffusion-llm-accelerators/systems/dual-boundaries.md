# Orchestrating Dual-Boundaries: An Arithmetic Intensity Inspired Acceleration Framework for Diffusion Language Models (ODB-dLLM)

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2511.21759` |
| 作者 / 單位 | 未查到第一作者全名（PKU-SEC-Lab，北京大學；聯絡人 lywei25@stu.pku.edu.cn） |
| 日期 | 2025-11 |
| 類別 | 系統與 serving（roofline 分析驅動的加速框架） |
| 連結 | [arXiv](https://arxiv.org/abs/2511.21759) · [GitHub](https://github.com/PKU-SEC-Lab/ODB-dLLM) |

## 一句話總結

ODB-dLLM 用 roofline / arithmetic intensity 分析指出 dLLM 推論同時卡在兩個邊界——prefill（含 cache refresh）是 compute-bound、block 解碼是 memory-bound——於是對前者用「EOS 信心驅動的自適應長度預測」砍掉冗餘序列，對後者用「jump-share 投機解碼」在頻寬受限的迭代裡塞進更多有用計算，在 A100 上對 LLaDA 達到 46–182× 加速、比 Fast-dLLM 快 2.6–7.2×。

## 要解決的問題

- **雙向注意力迫使 prefill 與 decoding 交錯**：dLLM 使用 KV cache 時，每個 block 結束都要 refresh 整段 cache（類 prefill，compute-bound），而 block 內的逐步 denoising 只算少量 token（memory-bound）；兩個 phase 的 arithmetic intensity 差異巨大，任何單一優化都只能碰到其中一個邊界。
- **固定回應長度造成大量冗餘計算**：dLLM 需要預先指定生成長度（例如 1024），但多數任務實際輸出遠短於此；多出來的 [MASK] 位置在每次 refresh 與每步 decoding 都要被算，白白抬高 prefill 成本。
- **既有平行解碼在 memory-bound 迭代裡沒有把算力用滿**：threshold 解碼每步只接受高信心 token，低於門檻但其實正確的候選被丟掉，迭代數仍多；而 GPU 在這些迭代中大部分時間在等記憶體。

## 核心方法與特色

- **Arithmetic intensity / roofline 建模**：在 NVIDIA A100 roofline 上量化 dLLM 各 phase 的 FLOPs/Byte；prefill+refresh 落在 compute-bound 區、block decoding 落在 memory-bound 區。框架設計原則：compute-bound 的部分「減少總工作量」，memory-bound 的部分「在不增加記憶體流量的前提下增加有用計算」。
- **Adaptive length prediction（自適應長度預測）**：利用模型自身對 [EOS] 的信心，在生成過程中逐步縮短「有效序列長度」——一旦後段位置持續高信心預測為 EOS，就把之後的 [MASK] 從計算圖中剪掉，後續 refresh 與 decoding 只處理縮短後的序列。實驗中有效序列長度減少 40–60%，直接降低 prefill 開銷。不需訓練。
- **Jump-share speculative decoding（跳躍共享投機解碼）**：把「低於 threshold 但信心仍高」的候選 token 當作投機分支，一次 forward 同時驗證多個分支（inter-block jump verification），並讓各分支共享已解碼 token 的 KV pairs，避免重複計算。因為 decoding 本來就是 memory-bound，多加的分支計算幾乎不增加時間，卻提高每次迭代接受的 token 數，減少總迭代數。dLLM 自己同時當 drafter 與 verifier，不需額外模型。
- **建立在 Fast-dLLM 之上**：沿用其 KV cache 與 threshold 解碼實作，兩個新機制以插件形式加入；程式碼開源（MIT）。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| LLaDA-8B-Instruct | 8B | 46–162× vs vanilla |
| LLaDA-1.5 | 8B | 50–182× vs vanilla |

## PPA / 效能數據

> 軟體論文：speedup、throughput (tok/s)、latency、記憶體、benchmark 準確度變化。

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| Speedup（LLaDA-8B-Instruct） | 46×–162× | vanilla LLaDA（無 cache、逐步解碼） | A100 80GB；GSM8K / MATH（Minerva）/ BBH / HumanEval / MBPP |
| Speedup（LLaDA-1.5） | 50×–182× | vanilla LLaDA-1.5 | 同上 |
| Speedup vs Fast-dLLM（LLaDA-8B-Instruct） | 2.63×–6.30× | Fast-dLLM | 同上 |
| Speedup vs Fast-dLLM（LLaDA-1.5） | 2.60×–7.22× | Fast-dLLM | 同上 |
| 有效序列長度縮減 | 40–60% | 固定生成長度 | adaptive length prediction |
| 絕對 tok/s | README 以圖片呈現，數字未擷取到 | — | — |
| 準確度變化 | 論文宣稱維持精度；逐項數字未查到 | — | — |

## 限制 / 備註

- 46–182× 的巨大倍數是相對「完全未優化的 vanilla LLaDA」（無 KV cache、每步一 token）；對實務更有意義的是 vs Fast-dLLM 的 2.6–7.2×。
- 自適應長度預測依賴模型 EOS 信心是否可靠；對開放式長文本或訓練時 EOS 訊號弱的模型可能提前截斷。
- 投機分支數與 threshold 的選擇影響接受率；batch size 較大時 decoding 不再純 memory-bound，jump-share 額外計算就不再免費。
- 只評測 A100 單卡與 8B 級 LLaDA，未涵蓋 Dream、MoE 或 block-diffusion 模型。

## 與其他論文的關係

- 直接建立在 **Fast-dLLM** 的 KV cache + threshold 平行解碼之上，並以其為主要對照。
- 自適應長度預測與 *Diffusion LLM with Native Variable Generation Lengths: Let [EOS] Lead the Way*（arXiv 2510.24605）、**Streaming-dLLM**（suffix pruning）屬同一路線：利用 EOS/後綴訊號剪掉冗餘 [MASK]。
- jump-share 投機解碼與 **Flash-dLLM** 的 cache-driven draft-and-verify（`flash-dllm.md`）、**LoPA**（lookahead parallel decoding）、**CreditDecoding** 同屬「dLLM 自我投機」家族。
- 其 roofline 觀點與 **dLLM-Serve** 的 Refresh（compute-bound）/ Reuse（bandwidth-bound）兩相分析互相印證（`dllm-serve.md`）。
