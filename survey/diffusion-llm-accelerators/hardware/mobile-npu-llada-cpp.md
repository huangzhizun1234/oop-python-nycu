# Efficient On-Device Diffusion LLM Inference with Mobile NPU（llada.cpp）

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2606.13740` |
| 作者 / 單位 | Tuowei Wang et al.（單位未在搜尋結果確認；Tuowei Wang 過去隸屬 Tsinghua University（依記憶，待確認）） |
| 日期 | 2026-06 |
| 類別 | 硬體加速器（商用手機 NPU 上的系統實作） |
| 連結 | [arXiv](https://arxiv.org/abs/2606.13740) · GitHub：論文稱基於 llama.cpp 擴充 12K+ 行程式，公開 repo 未查到 |

## 一句話總結

llada.cpp 是第一個 NPU-aware 的手機 dLLM 推論框架：把 block-wise 的 LLaDA 解碼「靜態化、形狀穩定化」以符合 Qualcomm Hexagon NPU 的執行特性（多 block 投機解碼填滿工作量、CPU 側雙路徑漸進修訂、swap 最佳化記憶體 runtime），在 Snapdragon 8 Elite 手機上把 LLaDA-8B 的生成延遲較 CPU baseline 降低 17x~42x 且品質不掉。

## 要解決的問題

- 手機 NPU（Hexagon HMX/HVX）偏好靜態、規則、形狀固定的計算圖；但 block-wise dLLM 推論本質上是動態的：
  1. **Token commitment 讓每 block 有效工作量越來越小**：一個 block 內越後面的去噪步，未定的 token 越少，NPU 的密集矩陣單元吃不飽。
  2. **Token revision 讓 KV cache 難以重用**：dLLM 允許已定 token 被改寫，AR 式的 append-only KV cache 假設失效。
  3. **NPU 可見位址空間有限**：大模型 (8B) 的權重/KV 需要不斷重新映射 (VA remapping) 與搬資料，overhead 很大。
- 既有的手機 NPU LLM 框架（llm.npu、llama.cpp-npu）都是為 AR 模型設計，沒有處理上述 dLLM 特有的動態性。

## 核心方法與特色

- **Multi-Block Speculative Decoding**：當目前 block 的後期去噪步工作量縮小時，把「未來 block」的 token 當作投機候選一起送進 NPU，把批次填回 NPU 喜歡的固定形狀；等於用未來 block 的預先計算換取單位時間的有效吞吐。代價：未來 block 在上下文尚未穩定時就被計算，單獨使用會掉精度。
- **Dual-Path Progressive Revision**：已 commit 的 token 在「穩定」之前仍保持可修訂；不穩定的 token / prefix KV 走 CPU 側路徑刷新，NPU 側繼續執行密集、不被打斷的主路徑。這把 dLLM 的 token revision 從 NPU 圖上移走，同時把投機解碼掉的品質拉回接近純 CPU 路徑。
- **Swap-Optimized Memory Runtime**：依計算圖的算子執行順序與張量的 producer-consumer 關係排列 buffer 映射，讓 NPU 可見位址空間內的 swap 次數最小化，並把資料 staging 與 NPU 計算重疊。
- **Hexagon 算子庫（12K+ LoC 擴充 llama.cpp）**：用 Hexagon SDK 編成 DSP shared object，鎖定 HMX（矩陣）與 HVX（向量）引擎，實作 FP16 matmul、Q4_0 4-bit 權重 matmul、FP16 KV 的 FlashAttention、RMSNorm、elementwise 算子、共享記憶體映射與 device-side worker pool。
- **支援 prefix KV cache 重用與較長上下文**：報告的 17x~42x 是在啟用 prefix KV cache 重用下量到的；記憶體 runtime 讓手機上能跑較長的 dLLM 生成。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| LLaDA-8B-Instruct | 8B | 主要工作負載；4-bit (Q4_0) 權重 + FP16 活化/KV |
| Llama（AR 參考） | 未查到具體版本 | 用作四個任務上的精度參考，證明 LLaDA 在手機上「可用」 |

## PPA / 效能數據

> 軟體/系統論文（跑在商用 NPU 上）：speedup、latency、能耗、精度變化。

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| 生成延遲降低 | 17x ~ 42x | 同手機上的 CPU baseline（llama.cpp 式 CPU 路徑） | LLaDA-8B，啟用 prefix KV cache 重用；OnePlus Ace5 Pro (Snapdragon 8 Elite, SM8750) |
| 精度 | 「與 CPU 路徑接近」；NPU 化本身不造成系統性掉分 | CPU 路徑 / Llama AR 參考 | 四個任務（名稱未查到，推測含 GSM8K/HumanEval 類） |
| 只用 Multi-Block Speculative Decoding | 會掉精度 | 同上 | Dual-Path Progressive Revision 把品質拉回 |
| 功耗與每請求能耗 | 皆下降（數值未查到） | CPU baseline | NPU 有足夠平行 token 工作量時才省能 |
| 跨裝置 | 亦在 OnePlus 12 (Snapdragon 8 Gen3, SM8650)、OnePlus 15 (Snapdragon 8 Elite Gen5, SM8850) 驗證 | — | 數值未查到 |
| tok/s 絕對值 | 未查到 | — | — |

## 限制 / 備註

- 這是「在商用手機 NPU 上的系統/軟體實作」而非新硬體；所有 speedup 都是相對於同機 CPU baseline，不是相對於手機 GPU 或桌面 GPU。
- 只針對 Qualcomm Hexagon（因其 SDK 相對開放），Apple ANE / 聯發科 APU 未涵蓋。
- 論文的關鍵洞見之一：「把 LLM 丟到 NPU 上」本身不保證省能，必須讓 NPU 有足夠的平行 token 工作——這正是 dLLM 相對 AR 模型在手機 NPU 上的結構性優勢。
- 具體 tok/s、能耗數字、四個 benchmark 名稱與精度表格本次未能取得（arXiv 被擋）。

## 與其他論文的關係

- 建立在 **llama.cpp** 與 AR 手機 NPU 系統（**llm.npu / mllm-NPU**（ASPLOS'25）、**llama.cpp-npu**）之上，把它們的 INT8/Q4 kernel、chunk-sharing、位址映射技巧延伸到 dLLM。
- 與 **Fast-dLLM**（block-wise KV cache + confidence-aware parallel decoding）互補：llada.cpp 的 block-wise 解碼與 prefix KV 重用都以 Fast-dLLM 類演算法為前提，再解決 NPU 上的動態形狀問題。
- 與 **DART (npu-dllm-sampling.md)** 是「軟體對齊現有 NPU」vs「重新設計 NPU」的兩端；DART 指出的取樣瓶頸在 llada.cpp 中由 CPU 側處理。
- 與 **DiPe (dipe-edge.md)** 同為邊緣 dLLM 部署，但 DiPe 在 Jetson GPU 上做量化 + 跨步聚合，llada.cpp 在手機 NPU 上做圖形靜態化。
