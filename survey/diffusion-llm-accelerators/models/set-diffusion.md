# Set Diffusion: Interpolating Token Orderings Between Autoregression and Diffusion for Fast and Flexible Decoding

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2607.01775`（ICML 2026） |
| 作者 / 單位 | Marianne Arriola, Volodymyr Kuleshov（Cornell Tech / Cornell University） |
| 日期 | 2026-07 |
| 類別 | 模型 |
| 連結 | [arXiv](https://arxiv.org/abs/2607.01775) · [GitHub](https://github.com/kuleshov-group/setdlms) · [Project](https://m-arriola.com/setdlms/) |

## 一句話總結

BD3-LM 作者的續作：把「固定大小、左到右的 block」推廣為「位置任意、大小可變的 token 集合（set）」，以 set-factorized 似然與 set-causal 注意力架構，讓模型每一步解碼一個任意順序的 token 集合並在每步後更新 KV cache，同時取得比 block diffusion 更高的平行度與任意順序（含 infilling、滑動視窗）解碼彈性。

## 要解決的問題

- Block diffusion（BD3-LM、LLaDA2.0、SDAR 等）雖能 KV cache，但 block 是固定大小且嚴格左到右：(1) 每個 block 內部要多步去噪，跨 block 不能平行，平行度受 block 大小限制；(2) 失去了全序列 dLLM 的任意順序生成與 infilling 能力。
- 全序列 dLLM 則反過來：彈性高、但無法 KV cache、步數多。作者想找一個「順序」層面的插值：AR 是一次一個 token 的固定順序，diffusion 是一次全部的任意順序，中間應存在「一次一個任意集合」的族。

## 核心方法與特色

- **Set-factorized 似然**：把序列分解為一串 token 集合 S_1, S_2, …，每個集合的位置可任意、大小可變（上限 s_max），p(x) = ∏_k p(x_{S_k} | x_{S_<k})；集合內以 masked diffusion 平行預測，集合間自迴歸。s_max = 1 對應 AR、單一集合對應全序列 diffusion、連續固定集合對應 BD3-LM。
- **Set-causal 架構**：注意力 mask 只允許看「已解碼集合」的 token，而不要求它們在位置上位於左側；因此每完成一個集合，其 KV 就是確定的、可寫入 cache，做到「每一步推論後都能更新 KV cache」，這是全序列 dLLM 做不到、block diffusion 只能在 block 結束時做到的。
- **Token ordering interpolation**：訓練時對集合的位置與大小做隨機化（含 block 集合與 sliding-window 集合），使同一模型在推論時可選擇左到右 block、滑動視窗、或任意順序（例如先填高信心位置、或做中間 infilling）而不需重訓。
- **加噪 / 去噪過程**：repo 描述噪聲是「由右到左一次遮罩一個 token」再迭代去噪，對應集合序列的反向過程。
- **代價**：任意順序需要位置感知的 set mask，實作與 kernel 複雜度高於 block diffusion；實驗規模為 OWT / LM1B 級小模型；相較 AR 的品質差距在論文中如何，本文未能取得數字。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| SetDLM-smax8 / -smax16 / -smax32 | 未查到（推測與 BD3-LM 同為 110M–級，待確認） | s_max = 集合大小上限；OWT / LM1B 訓練 |
| AR、MDLM、BD3LM | 同架構對照 | — |

## PPA / 效能數據

> 軟體論文：speedup、throughput (tok/s)、latency、記憶體、benchmark 準確度變化。

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| Perplexity（OWT / LM1B） | 論文有報告，數值未查到 | AR、MDLM、BD3LM | — |
| 吞吐 / 加速 | repo 註明在 H100 上量測 GSM8K、CNN/DailyMail 摘要、ROCStories infilling 三任務的吞吐並與 block diffusion 比較；倍率未查到 | BD3LM、AR | H100 |
| KV cache 更新頻率 | 每一步推論後 | BD3LM 每 block 結束；MDLM 無 | 架構性質 |
| 支援的解碼順序 | 左到右 block、sliding window、任意順序 / infilling | BD3LM 僅左到右 | — |
| 釋出 checkpoint | smax 8 / 16 / 32 三個 | — | HF |

## 限制 / 備註

- arXiv、專案頁、themoonlight 等來源皆被封鎖，本文的數值資訊僅來自 GitHub README 與摘要，perplexity / 加速倍率均「未查到」。
- 目前為小規模（OWT / LM1B）驗證，尚未在 7B+ 模型上證明；若要用於實際 serving，需要如 Fast-dLLM v2 般的 AR→SetDLM 轉換配方。
- 社群筆記（unturtle audit）指出截至 2026-09 HF 上尚無以此方法轉換的大型模型。

## 與其他論文的關係

- 直接延續 BD3-LM（同作者），把 block 推廣為 set；理論上包含 AR、MDLM、BD3-LM 為特例，並吸收 Eso-LM（any-order diffusion）的任意順序思想。
- 與 Nemotron-Labs-Diffusion 的 speed-of-light 分析互補：後者指出擴散取樣器離理論 TPF 上限還差 76.5%，Set Diffusion 提供更高平行度、且可 KV cache 的架構作為逼近上限的路徑。
- 與「The Flexibility Trap」（2601.15165，質疑任意順序對推理的價值）形成辯論：Set Diffusion 主張彈性順序可與 KV cache 效率並存。
- 對本 survey 中的 KV cache 類論文（dKV-Cache、dLLM-Cache、Fast-dLLM）而言，它代表「從架構層面讓 KV cache 精確成立」的方向，而非推論期近似。
