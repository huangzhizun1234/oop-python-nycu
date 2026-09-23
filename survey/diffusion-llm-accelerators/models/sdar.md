# SDAR: A Synergistic Diffusion–AutoRegression Paradigm for Scalable Sequence Generation

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2510.06303`（v3 2025-10-18） |
| 作者 / 單位 | Shuang Cheng, Yihan Bian, Dawei Liu, Yuhua Jiang, Yihao Liu, Linfeng Zhang, Wenhai Wang, Qipeng Guo, Kai Chen, Biqing Qi, Bowen Zhou（上海人工智能實驗室 Shanghai AI Lab；GitHub 組織 JetAstra） |
| 日期 | 2025-10 |
| 類別 | 模型 |
| 連結 | [arXiv](https://arxiv.org/abs/2510.06303) · [GitHub](https://github.com/JetAstra/SDAR) |

## 一句話總結

用僅 50B tokens 的輕量續訓 + 4B tokens SFT，把 Qwen3 系列（1.7B / 4B / 8B / 30B-A3B）轉換成 block diffusion 模型：block 間自迴歸、block 內平行去噪，在通用任務與 AR 對照持平、科學推理（GPQA、ChemBench）反超，推論比靜態解碼快 >2×，越大的模型對 block 大小與閾值越穩健。

## 要解決的問題

- 論文先做了受控實驗，證明在同資料同算力下 masked diffusion 從零訓練的計算效率明顯低於 AR，因此「從零訓練 dLLM」不是 scaling 的好路徑；應以訓練好的 AR 模型為基礎做「範式轉換」。
- 全序列 dLLM（LLaDA、Dream）無法 KV cache、長輸出慢；需要一種保留 AR 的全域一致性、又能在局部平行生成的推論方式，且轉換成本要低到能套用在 30B MoE 這種規模。

## 核心方法與特色

- **輕量 AR→block diffusion 轉換**：以 Qwen3-Base 為起點，續訓 50B tokens 開源資料（約為原預訓練量的 0.14%），再做 4B tokens SFT 得到 Chat 版；訓練時 block 內使用雙向注意力與遮罩去噪目標，block 間維持因果。轉換成本比 Dream（580B）低一個數量級。
- **Blockwise 推論（semi-AR）**：預設 block_length = 4、denoising_steps = 4，block 間自迴歸、block 內平行去噪，已完成的 block 直接 KV cache。可在訓練後再擴展到 block 8 / 16 / 32 / 64（block scaling 變體）。
- **動態信心閾值解碼**：threshold = 0.9 時，block 內信心高於閾值的 token 一次全部接受，實際步數常少於 4；論文報告此「動態」推論比「靜態」（固定步數）快 >2× 且準確率幾乎不變，並發現模型越大對 block 大小與閾值越不敏感，因此加速比隨規模提高。
- **JetEngine 推論引擎**：基於 nano-vllm 的專用引擎（另整合 LMDeploy），在 H200 上 SDAR-4B 達 3,700+ tok/s（FlashAttention-2，應為批次吞吐）。
- **SDAR-Sci**：在 30B-A3B 上先做 500B tokens 科學領域續訓 + 500B 退火，再 50B SDAR 轉換 + 推理 SFT；在 GPQA、ChemBench、Physics 上超越相同流程的 AR 版本，且在 majority voting / pass@k 測試時擴展下增益更大。
- **代價**：block 4 很小，平行度有限，加速主要靠動態閾值；轉換後在部分通用任務略低於 AR；benchmark 數字以圖片形式發布，不易引用。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| SDAR-1.7B / 4B / 8B-Chat | 1.7B / 4B / 8B（dense） | Qwen3-Base 轉換 |
| SDAR-30B-A3B-Chat | 30B 總 / 3B 啟用（MoE） | Qwen3-30B-A3B-Base 轉換 |
| SDAR-30B-A3B-Sci | 30B / 3B | 科學推理版 |
| Qwen3-1.7B / 30B-AR-SFT | 同尺寸 AR | 相同 SFT 資料的公平對照（greedy decoding） |
| AR-30B-A3B-Sci | 30B / 3B | SDAR-Sci 的 AR 對照 |

## PPA / 效能數據

> 軟體論文：speedup、throughput (tok/s)、latency、記憶體、benchmark 準確度變化。

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| 轉換訓練量 | 50B tokens 續訓（≈0.14% 原預訓練）+ 4B SFT | Dream 580B；LLaDA 2.3T | — |
| 通用任務（MMLU、GSM8K、MATH500、HumanEval、MBPP、IFEval 等） | 「與 AR-SFT 對照 on-par」 | Qwen3-1.7B-AR-SFT、Qwen3-30B-AR-SFT | 具體數字在 README 圖片中，未查到 |
| 科學推理（GPQA、ChemBench、Physics） | SDAR-30B-A3B-Sci 明顯高於 AR-30B-A3B-Sci，「接近或超越閉源系統」 | 相同流程 AR | GPQA 8 次平均；AIME24/25、LiveMathBench 32 次平均；數字未查到 |
| 動態 vs 靜態推論加速 | **>2×**，準確率損失可忽略 | 靜態固定步數 | block 4、steps 4、threshold 0.9 |
| 規模效應 | 越大模型對 block 大小與閾值越穩健 → 更高加速且不掉分 | — | 1.7B → 30B |
| 引擎吞吐 | SDAR-4B **3,700+ tok/s** | — | H200，JetEngine + FlashAttention-2，batch 未註明（應為批次吞吐） |
| 測試時擴展 | majority voting / pass@k 增益大於 AR | AR 對照 | 30B-A3B |

## 限制 / 備註

- 主要 benchmark 表格以圖片發布，本文無法逐項引用；「與 AR 持平」為作者敘述。
- 3,700 tok/s 未註明 batch size 與序列長度，不能與 Mercury / Gemini 的單請求速度直接比較。
- block_length = 4 意味每步最多 4 個 token，單請求延遲改善有限；加速依賴動態閾值與批次。
- 後續：SDAR 被 TraceRL / TraDo、DARE、I-DLM 等 dLLM RL 與加速工作用作基礎模型。

## 與其他論文的關係

- 方法直接來自 BD3-LM 的 block diffusion，並與 Fast-dLLM v2（Qwen2.5，~1B tokens）、Efficient-DLM（Qwen3，300–500B tokens）、LLaDA2.0（Ling MoE，>100B tokens）同屬 AR→block diffusion 轉換；SDAR 的訓練量（50B）介於 Fast-dLLM v2 與 Efficient-DLM 之間，規模（30B MoE）僅次於 LLaDA2.0。
- 「AR 比 MDM 計算效率高、故應轉換」的實證結論與 Efficient-DLM、LLaDA2.0 的動機一致，並直接反駁從零訓練路線（LLaDA-MoE）。
- 信心閾值動態解碼承自 Fast-dLLM；JetEngine 與 dInfer、Fast-dLLM 同為 dLLM 專用推論引擎。
- 是後續 dLLM 強化學習（TraceRL、d-TreeRPO）與 I-DLM（introspective strided decoding）的常用基礎模型。
