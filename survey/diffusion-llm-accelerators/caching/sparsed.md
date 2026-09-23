# SparseD: Sparse Attention for Diffusion Language Models

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2509.24014`（ICLR 2026） |
| 作者 / 單位 | Zeqing Wang, Gongfan Fang, Xinyin Ma, Xingyi Yang, Xinchao Wang（xML Lab, National University of Singapore；香港理工大學） |
| 日期 | 2025-09 |
| 類別 | KV cache 與稀疏（稀疏注意力） |
| 連結 | [arXiv](https://arxiv.org/abs/2509.24014) · [GitHub](https://github.com/INV-WZQ/SparseD) |

## 一句話總結

觀察到 dLLM 的注意力 pattern「每個 head 不同、但跨 denoising step 幾乎不變、且早期 step 最關鍵」，因此在早期用全注意力並算出每個 head 的稀疏 pattern，之後所有 step 重用該 pattern 做稀疏注意力，在 64k 上下文達到相對 FlashAttention 1.50× 的無損加速。

## 要解決的問題

- 長上下文 dLLM 推論的主要成本是每步的雙向全注意力 O(L²)，而且要重複 T 個 denoising step。
- AR 模型的稀疏注意力（如 sliding window、StreamingLLM 式 sink）假設因果與逐 token 查詢，直接套用到雙向、全序列同時查詢的 dLLM 會嚴重掉分。
- 若每個 step 都重新估計稀疏 pattern，估計本身的開銷會吃掉加速。

## 核心方法與特色

- **三個實證觀察**：(1) 不同 attention head 的稀疏 pattern 差異很大（head-specific）；(2) 同一 head 的 pattern 在相鄰甚至整個 denoising 過程中高度相似；(3) 早期 denoising step 對生成品質影響最大，後期 step 較能容忍近似。
- **一次計算、跨步重用的 head-specific 稀疏 pattern**：在前 `skip` 比例（例如 20%）的 step 用全注意力，同時以 block 為單位（block_size 32–128）統計每個 head 的重要 query-key block，選出 `select` 比例（0.3–0.5）的 block 作為該 head 的稀疏 mask；之後所有 step 直接重用，不再重新估計。
- **早期全注意力保品質**：因為早期 step 決定了全局結構，這段不做近似，之後才切換稀疏；作者稱之為「skip sparse attention in early steps」。
- **FlexAttention 實作**：稀疏 mask 以 block-sparse 形式交給 PyTorch FlexAttention 執行，因此加速是實際 wall-clock 而非僅 FLOPs。
- **Training-free、與 KV cache 正交**：不改變 cache 機制，可和 dKV-Cache 等疊加。
- **代價**：加速幅度受限於注意力在總時間中的占比；在短序列（4k 以下）幾乎無收益，需長上下文才顯著；pattern 固定後若後期注意力真的改變則無法適應。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| Dream-v0-Instruct-7B | 7B | 主要實驗 |
| LLaDA-1.5 | 8B | — |

## PPA / 效能數據

> 軟體論文：speedup、throughput (tok/s)、latency、記憶體、benchmark 準確度變化。GPU 型號未查到。

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| 最高加速 | 1.50× | FlashAttention（全注意力） | 64k 上下文、1,024 denoising steps |
| 品質 | 「lossless」，LongBench 上與原模型相當 | 原始 Dream / LLaDA | 4k–64k 上下文 |
| 測試上下文長度 | 4k / 8k / 16k / 32k / 64k | — | — |
| 典型超參數 | skip=0.2、select=0.3–0.5、block_size=32–128 | — | repo 範例 |

## 限制 / 備註

- 1.5× 是相對於已經很快的 FlashAttention kernel，而非相對原始 PyTorch 實作，因此數字看起來比其他 cache 論文保守，但是「真實 kernel 級」加速。
- 逐 benchmark 的準確度數字在 README 以圖呈現，未擷取到。
- 對短生成 / 短 prompt 任務（GSM8K 等）幾乎沒有收益，定位是長上下文。

## 與其他論文的關係

- 同實驗室（NUS xML）的 **dKV-Cache** 處理「時間軸」上的重複計算，SparseD 處理「空間軸」上的注意力冗餘，兩者互補可疊加。
- 與 **Sparse-dLLM**（動態淘汰 KV）、**Focus-dLLM**（信心引導預測 unmask 區域 + sink-aware pruning）同為 dLLM 稀疏注意力路線；Focus-dLLM 在 32K 上宣稱 29.6× 加速並以 SparseD 為比較對象之一。
- **Prefilling-dLLM** 以 chunk 級 top-K 選擇處理長 prompt，是更粗粒度的稀疏化，同樣鎖定長上下文。
- 後續 PulseCol（週期性刷新的 column-sparse attention）、FlashBlock 等工作延續「pattern 跨步重用」思路。
