# Accelerating Speculative Diffusions via Block Verification

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2606.13426` |
| 作者 / 單位 | Alexander Soen, Hisham Husain, Valentin De Bortoli, Arnaud Doucet（依記憶：RIKEN AIP / Google DeepMind 等，待確認） |
| 日期 | 2026-06 |
| 類別 | 平行與投機解碼（**連續 diffusion** 的 speculative sampling 理論） |
| 連結 | [arXiv](https://arxiv.org/abs/2606.13426) · GitHub：未查到 |

## 一句話總結

把 LLM 的 speculative sampling（含 block verification）嚴格搬到「連續空間」的 diffusion 模型：提出能高效從連續 residual 分佈採樣的機制，使 block verification 可證明提高接受率，並提出免訓練的自我投機 Free Drafter，比既有 speculative diffusion 方法再快最多 6.3%。

## 要解決的問題

- Speculative sampling 的核心是「拒絕後從 residual 分佈 (p − q)₊ 採樣」以保證輸出分佈不變；在離散 token 空間這很容易，但在連續空間（diffusion 的高斯狀態）residual 分佈沒有封閉形式，難以高效採樣。
- 既有 speculative diffusion 方法因此要嘛用昂貴的採樣技巧，要嘛改用替代（非嚴格）的接受方案，無法直接享受 LLM 端已知的改進（如 block verification）。
- **注意**：本文對象是連續（影像 / 一般）diffusion 的 sampler 步驟，不是 masked dLLM；收錄作為「投機解碼理論在 diffusion 上的一般化」對照。

## 核心方法與特色

- **連續空間的精確 speculative sampling**：設計一個能高效實作原始 speculative sampling 機制（含 residual 採樣）的方案，使 draft 步驟被拒絕時仍能從正確的修正分佈重新採樣，保持 target sampler 的分佈。
- **Block verification 移植**：LLM 端的 block verification（Sun et al. 2024）一次對整個 draft block 做聯合接受判定而非逐 token 獨立判定，可證明接受率不低於逐 token；本文證明在連續 diffusion 上同樣成立。
- **Free Drafter**：一種不需訓練的自我投機 drafter——用便宜的方式（例如重用前一步的預測 / 粗略 ODE 步）產生下一步或數步的 draft，再由 target sampler 的平行驗證 pass 決定接受；overhead 幾乎只有既有的平行驗證計算。
- **理論分析**：形式化 Free Drafter 的接受率並分析其與步長 / 噪音水準的關係。
- **代價**：加速幅度有限（6.3%），主要貢獻是理論正確性與可組合性。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| 連續 diffusion 模型（具體模型未查到） | — | 非 dLLM；未評測 LLaDA / Dream |

## PPA / 效能數據

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| 加速 | up to 6.3% | 既有 speculative diffusion 方法 | Free Drafter + block verification，無額外訓練 |
| 接受率 | 可證明提升 | 逐步驗證 | 理論結果 |
| 輸出分佈 | 與 target sampler 一致 | — | 精確 speculative sampling |
| GPU / 具體任務 | 未查到 | — | — |

## 限制 / 備註

- 非 dLLM 論文；對 masked dLLM 的直接意義在於：(1) block verification 的接受率優勢在離散 token 空間本就成立，可直接用於 Spiffy / FreeDave / PSD 類方法的驗證規則；(2) 提醒「lossless」需要正確的 residual 採樣，僅比對 argmax 的方法只在 greedy 下無損。
- 6.3% 為相對既有 speculative diffusion 的增量，非相對原始 sampler。

## 與其他論文的關係

- 理論根源為 LLM 的 **Block Verification Accelerates Speculative Decoding**（2403.10444）。
- 與 **Spiffy / FreeDave / PSD** 的關係：這些 dLLM 方法多以「與 greedy 一致」定義 lossless；本文提供的是分佈層級的保證，可作為它們在 temperature > 0 時的修正依據。
- 與 **Accelerating Diffusion Sampling via Speculative Draft Trees**（2609.17691）為同一路線的後續。
- 與 **DARTree / SimSD** 的 block / tree 驗證在形式上相通，但對象分別是 AR target 與 dLLM target。
