# Free Draft-and-Verification: Toward Lossless Parallel Decoding for Diffusion Large Language Models (FreeDave)

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2510.00294`（ID 已查證；v 最新更新 2026-02） |
| 作者 / 單位 | Shutong Wu, Jiawei Zhang（依記憶：UC Santa Barbara，待確認） |
| 日期 | 2025-09（arXiv 首次公開 2025-09-30） |
| 類別 | 平行與投機解碼（lossless、training-free） |
| 連結 | [arXiv](https://arxiv.org/abs/2510.00294) · [GitHub](https://github.com/cychomatica/FreeDave) |

## 一句話總結

把「上一步平行解出的多個 token」當作 draft，讓模型在下一步（本來就要做的 forward）順便驗證，完全不增加 forward 次數、不需任何額外模組，理論上以最少 forward 次數重現逐 token 解碼的結果，實測最高 2.83x。

## 要解決的問題

- 逐 token（one-token-per-step）解碼品質最好但最慢；threshold 平行解碼快但會掉分（尤其在溫度 > 0 時不穩）。
- 現有 speculative 方法要嘛需要額外 draft model / verifier（APD），要嘛需要複雜的 draft 結構（Spiffy）。
- 想要一個「免費」的 draft-and-verify：不多跑模型、不改模型，卻能保證輸出與逐 token 解碼一致。

## 核心方法與特色

- **Draft 來自平行解碼的副產品**：第 t 步除了取最高信心的一個 token（逐 token 解碼會 commit 的那個），也把其他高信心位置的預測留下來當候選 draft。
- **驗證由下一步的 forward 順便完成**：第 t+1 步本來就要 forward；把「只 commit 一個 token」與「同時 commit draft 候選」的幾種序列做成一個 batch 一起 forward，比較各分支的預測是否與逐 token 路徑一致，一致的就整段接受。因此 forward 次數不增加，只有 batch 維度變大。
- **理論保證**：證明在 greedy / 固定隨機種子下，此流程用最少的模型呼叫次數重現 one-token-per-step 的輸出序列（lossless）。
- **兩種 draft 實作**：`batch_expanding`（複製 cache 與當前 block 到 batch 維，記憶體多但計算少）與 `tree_attention`（只複製當前 block、用 block tree attention mask 共享 cache，記憶體開銷約 20MB，實測反而略快）。
- **也可疊在 threshold 平行解碼上**：以 subset-based 驗證規則，在 threshold 解碼的品質下再進一步提升 TPF；代價是 batch forward 的記憶體與算力開銷，接受率低時加速有限。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| LLaDA-8B-Instruct | 8B | full-attention dLLM |
| Dream-7B-Instruct | 7B | full-attention dLLM |
| TraDo-4B-Instruct / TraDo-8B-Instruct | 4B / 8B | block-attention dLLM（RL 訓練版） |
| SDAR | 未載明尺寸 | block-attention dLLM，repo 支援 |

## PPA / 效能數據

（下列 TPF / TPS 由官方 README 結果圖讀取，為近似值；GPU 未載明）

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| 最大加速 | 2.83x | 逐 token 解碼 | 數學與程式 benchmark，準確度不變（摘要） |
| LLaDA-8B TPF（static） | ≈1.0 → ≈2.5 | static 逐 token | MATH500 / GSM8K / MBPP / HumanEval |
| LLaDA-8B TPF（threshold） | ≈2.6–3.3 → ≈4.4–5.3 | threshold 平行解碼 | 同上 |
| LLaDA-8B TPS（HumanEval, threshold） | ≈120 → ≈165 tok/s | threshold 解碼 | 0-shot |
| Dream-7B TPS（static） | ≈60 → ≈110–130 tok/s | static 逐 token | 四個 benchmark |
| TraDo-8B TPF（static） | ≈0.8 → ≈1.5–1.6 | static | block-attention 模型加速較小 |
| 準確度 | 各模型 / benchmark 與對應 baseline 幾乎重合 | — | lossless（static）/ 近似（threshold） |

## 限制 / 備註

- TraDo / SDAR 這類 block-attention 模型 TPF 基準低於 1（因 block 內每步不一定 commit），加速倍率也較小。
- README 註明早期版本沿用 Trace-RL 的溫度 0.1 設定導致 threshold 解碼掉分，2026-04 後改為溫度 0；引用舊版數字需注意。
- 依賴 batch forward，在記憶體受限或 batch 已飽和的 serving 場景下收益下降。

## 與其他論文的關係

- 與 **Spiffy** 為同期「self-speculative lossless」雙雄；FreeDave 更輕量（零額外 forward、零校準）。
- 建立在 **Fast-dLLM** 的 threshold 平行解碼與 KV cache 之上（repo 引用），可視為給它加上 lossless 驗證。
- **PSD** 與 **SimSD** 後續進一步探討多深度 draft 與跨模型 draft。
- 與 **Block Verification**（2606.13426）在理論上都關心「接受規則」，但後者是連續 diffusion。
