# Accelerating Diffusion LLMs via Adaptive Parallel Decoding (APD)

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2506.00413`（NeurIPS 2025, Spotlight） |
| 作者 / 單位 | Daniel Israel, Guy Van den Broeck, Aditya Grover（UCLA） |
| 日期 | 2025-05 |
| 類別 | 平行與投機解碼 |
| 連結 | [arXiv](https://arxiv.org/abs/2506.00413) · [GitHub](https://github.com/danielmisrael/apd) |

## 一句話總結

用一個小型 AR 模型（Qwen2.5-0.5B）的 joint 機率去「驗證」dLLM 一次平行提出的多個 token，動態決定每步接受幾個，並加上 KV cache 與限制 masked 輸入長度，讓 Dream-7B 的 throughput 首次超過同尺寸 AR 模型。

## 要解決的問題

- dLLM 理論上可平行生成，但實務上（LLaDA、Dream）每步只解一個 token 才能維持品質；一次平行採樣多個 token 時，各位置的 marginal 彼此獨立，會產生不連貫的組合（joint 與 marginal 乘積不一致）。
- dLLM 沒有 KV cache，每步都對整段 masked 序列做 full forward，長度越長越慢。
- 既有的固定 k-token-per-step 或固定 threshold 策略無法依內容難易度自適應。

## 核心方法與特色

- **Multiplicative mixture 驗證準則**：定義目標分佈為 dLLM marginal 機率（各位置獨立）與小型 AR 模型 joint 機率的乘積混合，權重 R（`apd_mixture_weight`）。從 dLLM 一次平行採樣出候選序列後，由左到右用 AR 模型逐 token 檢查是否落在混合分佈下可接受，接受最長的前綴。這等於「反轉」speculative decoding：大模型（dLLM）當 drafter、小 AR 模型當 verifier。
- **自適應平行度**：接受的 token 數由驗證結果決定，簡單片段（模板、常見片語）一次接受很多，困難片段自動退回接近逐 token；GSM8K 上平均每步超過 5 個 token。
- **KV cache 啟用**：因為採用由左到右的接受方式，已接受的 prefix 不再改變，可對 dLLM 做 prefix KV cache，避免每步重算整段。
- **限制 masked 輸入長度（max_lookahead M）**：只把接下來 M 個 mask 位置餵進 dLLM，而非整個 gen_length，減少每步的計算量；另有 `kv_window` W 控制 cache 視窗。
- **三個旋鈕（R, M, W）**構成 throughput–quality 的 Pareto 前緣；代價是需要額外載入一個小 AR 模型並多一次 verifier forward，且 R 調高時品質會下降。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| Dream-7B（Dream-v0-Instruct） | 7B | 主要 dLLM，從 Qwen2.5-7B 蒸餾而來 |
| Qwen2.5-0.5B | 0.5B | 輔助 AR verifier |
| LLaDA-8B | 8B | 作為 baseline 比較（repo 內含 llada 實作） |
| Qwen2.5-7B（AR） | 7B | AR throughput 基準 |

## PPA / 效能數據

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| Throughput | 59 tok/s | AR Qwen2.5-7B 37 tok/s（≈1.6x） | Dream-7B + APD，NVIDIA A5000 |
| 每步平均接受 token 數 | > 5 tokens/iteration | 逐 token 解碼 = 1 | GSM8K，維持約 80% 準確度 |
| GSM8K 準確度 | ≈ 80% | Dream-7B 原始解碼（數字未查到） | 高 throughput 設定 |
| 其他 benchmark | GPQA、MATH、HumanEval 有評測 | — | 逐項數字未查到 |

## 限制 / 備註

- 需要一個與 dLLM 詞表相容的小 AR 模型（Dream 與 Qwen2.5 共用 tokenizer 才可行；LLaDA 沒有現成的小 AR 模型）。
- 由左到右接受 prefix 的設計使 dLLM 退化為近似 AR 的生成順序，放棄了部分「任意順序」的彈性。
- 品質數字（如 80% GSM8K）已低於 Dream-7B 原始逐 token 解碼的水準，論文將其視為可調的 trade-off。

## 與其他論文的關係

- 是最早把「speculative-decoding 式驗證」引入 dLLM 的工作之一；**Spiffy**、**FreeDave**、**PSD** 後續改為「自我投機」（不需外部 AR 模型）。
- 與 **Fast-dLLM**（KV cache + confidence threshold）同期，兩者都指出 prefix cache 對 dLLM throughput 的關鍵性。
- **DARTree / DFlash / Domino** 走相反方向：用 diffusion 模型當 drafter、AR 模型當 target，APD 則是 dLLM 當 drafter、小 AR 當 verifier。
- **Jacobi Forcing** 引用了同樣的「AR 與 diffusion 混合」動機，但選擇改造 AR 模型而非改造 dLLM 解碼。
