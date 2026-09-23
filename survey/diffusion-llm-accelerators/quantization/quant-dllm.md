# Quant-dLLM: Post-Training Extreme Low-Bit Quantization for Diffusion Large Language Models

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2510.03274`（ICLR 2026 poster） |
| 作者 / 單位 | Tianao Zhang, Zhiteng Li, Xianglong Yan, Haotong Qin, Yong Guo, Yulun Zhang（上海交通大學為主，Haotong Qin 為 ETH Zürich，依記憶，待確認） |
| 日期 | 2025-10 |
| 類別 | 量化 |
| 連結 | [arXiv](https://arxiv.org/abs/2510.03274) · [GitHub](https://github.com/ZTA2785/Quant-dLLM) · [OpenReview](https://openreview.net/forum?id=HD7tuVakmR) |

## 一句話總結

第一個把 dLLM 推到 **2-bit weight-only** 的 PTQ 框架：用 MCS 模擬 masked-denoising 的校準分佈、用 DAQ 以多個二值矩陣疊加逼近權重、用 ABMP 在嚴格 2-bit 平均預算下做 1/2/3-bit block 混合精度，把 LLaDA-8B 從 16.09 GB 壓到 3.69 GB 且準確度遠高於 GPTQ / GPTAQ / Slim-LLM。

## 要解決的問題

- dLLM 與 AR LLM 一樣持續變大，邊緣部署需要極低 bit 的 weight 壓縮；但既有 AR-transfer PTQ（GPTQ、GPTAQ、Slim-LLM）在 dLLM 上降到 2-bit 時準確度崩潰。
- 兩個根本原因：(1) 標準 PTQ 校準假設輸入是「完整可見」的文字，而 dLLM 推論時輸入是「部分 mask、依 timestep 變化」的序列，校準統計（Hessian 等）與實際推論分佈不匹配；(2) 2-bit 的均勻量化格點太少，需要更有表達力的量化器與精度分配。

## 核心方法與特色

- **MCS（Masked Calibration Simulation）**：在收集校準激活 / Hessian 時，不餵乾淨文字，而是依 dLLM 的 timestep-dependent masking 機率對校準序列做遮罩，模擬真實 denoising 時模型看到的輸入分佈，使得後續 Hessian-based 誤差補償真正對齊推論狀態。這是「幾乎零成本但影響最大」的改動。
- **DAQ（Data-aware Any-order Quantizer）**：把每個權重矩陣表示成多個二值矩陣（±1）的加權疊加（builds on BiLLM / ARB-LLM 的 alternating refined binarization），並配合 row-column rescaling，用 MCS 校準資料引導的迭代優化（交替更新二值矩陣與尺度）逼近原權重；「any-order」指可用任意個二值分量對應 1/2/3-bit 等不同精度。
- **ABMP（Adaptive Blockwise Mixed Precision）**：以 Hessian 敏感度為依據，在 channel group / block 層級把 bit 寬度在 1、2、3-bit 之間重新分配，並保證平均恰好落在 2-bit 預算內（repo 預設 ABMP ratio：LLaDA 5%、Dream 10%），把有限 bit 花在最敏感的 block。
- **Training-free、weight-only**：整條流程無需微調；激活維持高精度，因此不受 dLLM 激活 outlier 問題影響，適合以記憶體為瓶頸的部署場景。
- **代價**：校準需要約 80 GB GPU（8B 模型、C4 128 samples、seq len 4096、group/block size 128）；多二值矩陣疊加的解量化 kernel 比標準 INT2 GEMM 複雜，論文未主打 latency 加速。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| LLaDA-8B-Base | 8B | 主要報告對象 |
| LLaDA-8B-Instruct | 8B | |
| LLaDA-1.5 | 8B | |
| Dream-v0-Base-7B | 7B | |
| Dream-v0-Instruct-7B | 7B | |

## PPA / 效能數據

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| 記憶體 | 16.09 GB → 3.69 GB（約 4.4×） | FP16 | LLaDA-8B-Base，2-bit weight-only |
| 7 項一般知識任務平均準確度 | 51.3% | Slim-LLM 40.9% / GPTQ 36.5% / GPTAQ 35.6% | 2-bit（模型為 LLaDA 系列之一，具體型號未查到） |
| 平均分數提升 | 42.39 → 54.06（相對基線 +27%） | 最佳基線 | LLaDA-8B-Base，2-bit |
| 相對 FP 保留率 | 平均達 FP16 的 87.72% | FP16 | LLaDA-8B-Base，2-bit |
| 評估任務 | PIQA、ARC-E、ARC-C、HellaSwag（zero-shot）；MMLU、WinoGrande（5-shot）；BBH；GSM8K 等數理與程式碼任務 | — | 依 repo eval 腳本 |
| 推論 latency / speedup | 論文未提供 / 未查到 | — | — |

## 限制 / 備註

- 只做 weight-only；激活仍為高精度，因此不解決 W4A4 類的激活 outlier 問題，與 DLLMQuant / STaR-Quant 的問題設定不同。
- 2-bit 下數學與程式碼生成任務仍有明顯掉分（具體數字未查到），一般知識任務保留率較好。
- 二值矩陣疊加格式需要專用 kernel 才能兌現記憶體節省為速度。

## 與其他論文的關係

- 建立在 **BiLLM / ARB-LLM**（AR LLM 二值化）之上，程式碼直接衍生自 ARB-LLM。
- 與 **DLLMQuant (2508.14090)** 的 TMAS 想法相通（校準資料須模擬 timestep-dependent masking），但本篇聚焦 2-bit weight-only。
- 引用 **Quantization Meets dLLMs (2508.14896)** 的結論作為動機（AR PTQ 直接遷移到 dLLM 效果差）。
- 與 **Sink-Aware Pruning (2602.17664)** 同屬 dLLM 的 weight 壓縮路線（量化 vs. 剪枝），可互補疊加。
