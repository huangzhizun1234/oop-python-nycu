# Swordsman: Entropy-Driven Adaptive Block Partition for Efficient Diffusion Language Models

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2602.04399` |
| 作者 / 單位 | 未查到（arXiv 被擋、搜尋摘要未列出作者） |
| 日期 | 2026-02 |
| 類別 | 平行與投機解碼（自適應 block 分割） |
| 連結 | [arXiv](https://arxiv.org/abs/2602.04399) · GitHub：未查到 |

## 一句話總結

block-wise dLLM 解碼不再每 32 個 token 硬切一刀，而是依相鄰 token 間的 entropy 變化找出語意 / 句法成分的邊界來動態切 block，讓 LLaDA-8B-Instruct 在 GSM8K 達 81.50%（比 Fast-dLLM 高 6.29 點）且維持 8.79x 加速。

## 要解決的問題

- Semi-AR / block-wise 解碼（LLaDA 預設、Fast-dLLM 的 KV cache 都依賴它）用固定長度 block（例如 32），這種切法會把完整的語意或句法單位（一個子句、一段程式敘述）切成兩半，模型在 block 邊界失去必要的局部上下文，預測品質下降。
- 固定 block 也讓 block 內平行度與難度不匹配：簡單區段本可用大 block，複雜區段需要小 block。

## 核心方法與特色

- **Entropy Reduction Hypothesis (ERH) 的借用**：語言學上，成分邊界處是不確定性大幅下降（或重置）的地方；作者觀察到 dLLM 對相鄰位置的預測 entropy 在成分邊界會出現明顯跳變，因此 entropy 剖面可以當作免費的邊界偵測器。
- **Entropy-driven 邊界偵測**：在每個 block 開始前，用模型當前對後續 mask 位置的 entropy 分佈，找出相鄰 token 間 entropy 差異超過門檻的位置作為候選邊界，並以最小 / 最大 block 長度做約束，得到本次 block 的實際長度。
- **自適應 block 解碼**：block 邊界對齊成分邊界後，block 內的雙向注意力剛好覆蓋一個完整語意單位，減少邊界截斷造成的錯誤；同時 KV cache（prefix）機制沿用 Fast-dLLM，不損失速度。
- **Training-free、只改分割策略**：模型與解碼規則（threshold 平行解碼、cache）不動，因此能與各種既有加速直接組合；代價是每個 block 開始前需要一次額外的 entropy 評估（可與正常 forward 共用）。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| LLaDA-8B-Instruct | 8B | GSM8K 等 |
| LLaDA-1.5 | 8B | HumanEval 等 |
| Dream-7B | 7B | 依搜尋摘要推測有評測，待確認 |

## PPA / 效能數據

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| GSM8K 準確度 | 81.50% | Fast-dLLM（低 6.29 點，即約 75.2%） | LLaDA-8B-Instruct |
| HumanEval 準確度 | 43.90% | Fast-dLLM（低 8.31 點，即約 35.6%） | LLaDA-1.5 |
| 加速 | 8.79x | vanilla LLaDA | GSM8K；「與 Fast-dLLM 相當或更快」 |
| GPU / tok/s | 未查到 | — | — |

## 限制 / 備註

- 作者、單位、程式碼與完整結果表皆未查到；上述數字來自搜尋摘要。
- 準確度提升幅度（+6–8 點）相當大，可能部分來自 baseline Fast-dLLM 的設定（threshold / block 32），需查原文確認公平性。
- Entropy 門檻與最小 / 最大 block 長度為新增超參。

## 與其他論文的關係

- 建立在 **Fast-dLLM**（block-wise + KV cache + threshold 平行解碼）之上，只替換 block 分割策略。
- 與 **BlockBatch**（多 block size 共識）、**SemBlock**（語意邊界動態 block, 2606.04964）同屬「block 怎麼切」的研究線；SemBlock 可視為直接後續。
- 與 **DAWN** 互補：Swordsman 對齊 block 與語意單位，DAWN 在 block 內避免同步解耦合 token。
- 對 **CDLM / Fast-dLLM v2 / SDAR** 這類訓練時就固定 block 大小的模型不直接適用（它們的 block 是架構性的）。
