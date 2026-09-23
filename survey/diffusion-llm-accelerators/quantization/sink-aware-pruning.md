# Sink-Aware Pruning for Diffusion Language Models

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2602.17664` |
| 作者 / 單位 | Aidar Myrzakhan, Tianyi Li, Bowei Guo, Shengkun Tang, Zhiqiang Shen（VILA Lab, MBZUAI，依記憶，待確認） |
| 日期 | 2026-02 |
| 類別 | 量化 / 壓縮（剪枝） |
| 連結 | [arXiv](https://arxiv.org/abs/2602.17664) · [GitHub](https://github.com/VILA-Lab/Sink-Aware-Pruning) |

## 一句話總結

指出 dLLM 的 attention sink 是「隨 denoising step 漂移的暫態現象」而非 AR LLM 那種固定的全域錨點，因此 AR 剪枝「永遠保留 sink」的啟發式在 dLLM 上反而有害；提出把不穩定 sink 的激活貢獻歸零後再套 Wanda / SparseGPT，在 LLaDA-8B 50%–70% 稀疏度下一致優於原方法。

## 要解決的問題

- dLLM 每個 token 要跑多次 denoising，推論成本高，需要剪枝來減少權重與計算；但 dLLM 專用的剪枝研究幾乎空白，只能直接套 AR 的 Wanda / SparseGPT。
- AR 剪枝方法（含其 outlier / sink 保護機制）隱含假設 attention sink 固定在 prefix token、跨時間穩定；作者量測發現 dLLM 的 sink 位置在整個生成軌跡上變異數很高、空間上分散、且隨 denoising 推進逐步移動——在高噪聲階段形成全域結構時重要，之後就消退。
- 若照 AR 的作法保留 sink 對應的權重，等於把稀疏度預算花在「未來不會持續存在」的位置上，壓縮後模型品質反而下降。

## 核心方法與特色

- **Sink 動態量測**：在完整 denoising 軌跡上逐步計算每個 token 收到的 attention mass，找出 sink token，並統計其位置變異數；變異數大者判定為「不穩定 / 暫態 sink」。
- **Sink-aware 激活修正**：依 attention mass 推導每個 token 的 down-weighting 係數，把不穩定 sink token 對應的激活列（activation rows）歸零（或壓低），得到「去除暫態 sink 影響」的校準激活。
- **無縫接到既有剪枝準則**：把修正後的激活餵給 Wanda（|W|·‖X‖ 重要度）或 SparseGPT（Hessian-based OBS），由它們決定最終剪掉哪些權重；因此方法是一個通用的「重要度校正」而非新剪枝準則，支援 unstructured 與 structured 兩種模式。
- **無需重訓**：純 post-training、一次性（one-shot）剪枝，只需要少量校準資料跑 denoising 軌跡。
- **代價**：校準需模擬多步 denoising 以取得 sink 統計（比 AR 剪枝的單次 forward 貴）；unstructured 稀疏在 GPU 上不直接轉為速度，structured 版本才有實際加速，但掉分較大。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| LLaDA-8B-Base | 8B | 主要報告對象 |
| LLaDA-1.5 | 8B | |
| Dream-7B | 7B | |
| MMaDA | 8B | 多模態 dLLM |

## PPA / 效能數據

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| 平均準確度（6 任務：MMLU/ARC-C/PIQA/WinoG/GSM8K/HellaSwag） | 53.18（Wanda 骨幹）；52.36（SparseGPT 骨幹） | Wanda 52.70；SparseGPT 52.34；Dense 57.93 | LLaDA-8B，50% unstructured |
| MMLU | 62.16 | Wanda 61.43 | LLaDA-8B，50% unstructured |
| ARC-C | 41.38 | Wanda 39.08 | 同上 |
| GSM8K | 55.88（Wanda 骨幹）/ 52.11（SparseGPT 骨幹） | Wanda 57.01 / SparseGPT 53.53 | 同上（GSM8K 略輸 Wanda，是少數例外） |
| Structured pruning 30% | PIQA 0.6955 / WinoG 0.6740 / ARC-E 0.7175 / ARC-C 0.3820 | Baseline 0.6834 / 0.6630 / 0.6907 / 0.3780 | LLaDA-8B，structured |
| Structured pruning 50% | PIQA 0.6037 / WinoG 0.5724 / ARC-E 0.5279 / ARC-C 0.2362 | Baseline 0.5898 / 0.5572 / 0.4853 / 0.2039 | LLaDA-8B，structured |
| 高稀疏度趨勢 | 增益在 50%–75% 稀疏度最明顯 | Wanda / SparseGPT | 8 個 benchmark（全表見論文） |
| 推論 speedup | 論文未提供 / 未查到 | — | — |

## 限制 / 備註

- 增益幅度整體不大（50% unstructured 平均 +0.5 左右），主要價值在於指出「AR sink 假設不適用 dLLM」的觀察。
- 50% 稀疏後 GSM8K 仍從 69.29 掉到 ~56，說明 dLLM 剪枝對推理任務仍相當傷。
- 未報告實際 wall-clock 加速或 2:4 半結構化稀疏的結果。

## 與其他論文的關係

- 建立在 AR 剪枝方法 **Wanda** 與 **SparseGPT** 之上，作為其激活修正前處理。
- 與 **Quant-dLLM (2510.03274)**、**DLLMQuant (2508.14090)** 同屬 dLLM 權重壓縮，量化與剪枝可疊加。
- 與關注 dLLM attention 稀疏的 **SparseD (2509.24014)**、**Sparse-dLLM (2508.02558)** 互補：後者剪的是 runtime attention / KV，本篇剪的是靜態權重。
- 「sink 隨 timestep 漂移」的觀察也與 STaR-Quant / FAIR-Calib 的「dLLM 狀態隨時間變化、AR 假設不成立」論點一致。
