# Spiffy: Multiplying Diffusion LLM Acceleration via Lossless Speculative Decoding

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2509.18085`（ICML 2025 Workshop；後續版本改題為 "Structuring The Future: Diffusion LLM Speculative Decoding via Calibrated Draft Graphs"） |
| 作者 / 單位 | Sudhanshu Agrawal, Risheek Garrepalli, Raghavv Goel, Mingu Lee, Christopher Lott, Fatih Porikli（Qualcomm AI Research） |
| 日期 | 2025-09 |
| 類別 | 平行與投機解碼（lossless speculative decoding） |
| 連結 | [arXiv](https://arxiv.org/abs/2509.18085) · GitHub：未查到官方 repo |

## 一句話總結

把 speculative decoding 搬到 dLLM：用 dLLM 自己的分佈做 auto-speculative drafting，把多個候選 draft state 組織成「有向 draft graph」讓一次 forward 就能驗證多步，在保證輸出分佈不變的前提下加速 2.8–3.1x，與 KV cache、multi-token unmasking 疊加後總加速最高 7.9x。

## 要解決的問題

- 開源 dLLM 每個 denoising step 通常只 unmask 一個 token，實際速率遠低於「平行生成」的理論優勢。
- AR 的 speculative decoding 依賴左到右的 draft chain 與 causal 驗證，dLLM 是雙向、block-wise、任意順序 unmask，draft 的結構與驗證方式都要重新設計。
- 訓練獨立的 draft model 成本高，且對 dLLM 沒有現成小模型。

## 核心方法與特色

- **Auto-speculation（自我投機）**：不訓練 draft model，直接從 dLLM 當前 step 的預測分佈中抽出多個候選 unmask 組合作為 draft states，成本只是一次 forward 的副產品。
- **Directed draft graph**：由於 dLLM 一步可以 unmask 任意位置，draft 不是鏈而是圖：節點為「部分 unmask 的序列狀態」，邊為一次 unmask 動作；不同路徑可以匯合到同一狀態（比 tree 更省節點）。整張圖以 batch 方式一次餵進 dLLM 做平行驗證，接受最深的合法路徑。
- **離線校準（calibrated）+ 推論時動態剪枝**：先用少量資料統計哪些圖結構（分支寬度、深度）接受率最高，把圖形狀固定下來；推論時再依當步信心動態砍掉低機率分支，減少驗證 batch 大小。
- **Lossless 保證**：驗證規則設計成與原本逐 token（static unmasking, one token per step）解碼的輸出分佈一致（provably preserves output distribution），因此準確度理論上不變。
- **與其他加速正交且可乘**：與 prefix KV cache、multi-token unmasking（confidence threshold）疊加後，總加速可達 7.9x；代價是驗證 batch 增加了每步 FLOPs，在算力已飽和（大 batch serving）時效益下降。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| LLaDA-8B-Instruct | 8B | GSM8K / MATH500 / MBPP / HumanEval，block size 512 |
| Dream-7B-Instruct | 7B | 同上 |
| SDAR-8B-Chat-b32 | 8B | block diffusion 模型（block 32） |

## PPA / 效能數據

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| Spiffy 單獨加速 | 2.8x–3.1x | static unmasking（每步 1 token）+ prefix KV cache | LLaDA / Dream / SDAR，數學與程式任務 |
| 疊加 KV cache + multi-token unmasking 的總加速 | up to 7.9x | 原始逐 token 解碼 | 論文摘要（v1） |
| 準確度變化 | 0（lossless，分佈保證） | — | 理論保證 |
| GPU / 絕對 tok/s | 未查到 | — | — |

## 限制 / 備註

- 「lossless」是相對於 static 逐 token 解碼；若疊在 multi-token unmasking 上，整體品質取決於後者。
- 摘要中的 7.9x 為 v1 數字；後續改題版本的數字可能更新，未查到。
- 未查到官方程式碼，第三方復現有限。

## 與其他論文的關係

- 與 **FreeDave**（2510.00294）幾乎同時、同為「self-speculative + lossless」，FreeDave 主打零額外 forward、Spiffy 主打 draft graph 結構與校準。
- **PSD**（2605.15609）明確引用並延伸：在 Spiffy 的「時間軸投機」之上再乘上「空間軸平行 unmask」。
- **SimSD**（2606.02544）改走「小 dLLM draft + 大 dLLM 驗證」的傳統 SD 架構，與 Spiffy 的 auto-speculation 互為對照。
- 與 **APD** 相反：APD 用外部 AR 模型驗證 dLLM 的 draft，Spiffy 完全在 dLLM 內部完成。
