# Quantization Meets dLLMs: A Systematic Study of Post-training Quantization for Diffusion LLMs

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2508.14896`（期刊版：Machine Intelligence Research, Springer, 2025） |
| 作者 / 單位 | Haokun Lin et al.（中科院自動化所 / 香港城市大學 / 港中文 / 南洋理工等，依論文作者列表：Haokun Lin, Haobo Xu, Yichen Wu, Ziyu Guo, Renrui Zhang, Zhichao Lu, Ying Wei, Qingfu Zhang, Zhenan Sun） |
| 日期 | 2025-08 |
| 類別 | 量化 |
| 連結 | [arXiv](https://arxiv.org/abs/2508.14896) · [GitHub](https://github.com/FelixMessi/QDLM)（撰寫時回傳 404，可能已改名或設為私有） |

## 一句話總結

第一篇系統性地把 AR LLM 的 PTQ 方法（GPTQ、AWQ、SmoothQuant、QuaRot、DuQuant）搬到 dLLM（LLaDA、Dream）上做 benchmark 的研究，指出 dLLM 也有 activation outlier（特別是 massive outlier）問題，並給出「weight-only 用 GPTQ、W4A4 必須用 rotation 類方法」的實務建議。

## 要解決的問題

- dLLM（LLaDA-8B、Dream-7B）參數量與 AR LLM 相當，但每個 token 要經過多次 denoising forward，記憶體與算力需求更高，邊緣部署困難。
- PTQ 在 AR LLM 已很成熟，但 dLLM 的 full attention、masked-denoising 解碼、多 timestep 的激活分佈是否適用既有 PTQ 完全未知；沒有任何 benchmark 告訴使用者「哪個方法、哪個 bit-width 在 dLLM 上安全」。

## 核心方法與特色

- **首個 dLLM PTQ benchmark 框架**：實作 GPTQ、AWQ（weight-only）與 SmoothQuant、QuaRot、DuQuant（weight-activation）五種 SOTA PTQ，沿四個維度系統評估：bit-width、量化方法、任務類別（常識/知識、數學、程式碼）、模型型態（Base vs. Instruct、LLaDA vs. Dream）。
- **發現 dLLM 的 activation outlier 結構**：把離群值分為 *Normal Outliers*（在多數 token 上都偏大的 channel）與 *Massive Outliers*（只出現在極少數 token、但數值極端大）。Massive outlier 在 LLaDA-Base、LLaDA-Instruct、Dream 的多層與多種輸入激活中都觀察到，說明這是 dLLM 的共通現象，而非單一模型的特例；它會把動態範圍撐大，使低 bit 激活量化失真。
- **Weight-only 結論：GPTQ 優於 AWQ**：在 LLaDA-8B / LLaDA-8B-Instruct 的 3-bit 與 4-bit weight-only 設定下，GPTQ 平均準確度都高於 AWQ。作者推測原因是 LLaDA 系列的 (normal) activation outlier 不如傳統 AR LLM 明顯，AWQ 依賴「用激活尺度保護顯著 weight channel」的假設在 outlier 結構弱時效果打折。
- **Weight-activation 結論：rotation 類方法完勝 SmoothQuant**：QuaRot 與 DuQuant 在所有任務與量化設定都優於 SmoothQuant；差距在 W4A4 時最明顯，SmoothQuant 在程式碼與數學任務幾近全面崩潰，而 DuQuant 透過 rotation + permutation 進一步平滑 massive outlier，表現最穩。
- **任務敏感度**：生成型推理任務（GSM8K、HumanEval 等）對量化最敏感，知識/常識 QA 相對穩健；低 bit 激活量化會被多步 denoising 放大誤差，這也是後續 DLLMQuant、STaR-Quant 等工作的出發點。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| LLaDA-8B-Base | 8B | 主要分析對象 |
| LLaDA-8B-Instruct | 8B | 比較 Base vs. Instruct 對量化的敏感度 |
| Dream-7B（Base / Instruct） | 7B | 驗證 massive outlier 為跨模型共通現象 |

## PPA / 效能數據

> 軟體論文：本篇為 benchmark 研究，重點在各方法/各 bit-width 的準確度變化；未主打 speedup。

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| Weight-only 3/4-bit 平均準確度 | GPTQ > AWQ（具體差距論文未於摘要提供 / 未查到） | AWQ | LLaDA-8B-Base 與 Instruct |
| W4A4 平均準確度 | QuaRot / DuQuant ≫ SmoothQuant；SmoothQuant 在 code / math 幾近崩潰 | SmoothQuant | LLaDA / Dream，per-channel/per-token 4-bit |
| W8A8 / W4A16 | 接近無損（依記憶，待確認） | FP16 | LLaDA-8B |
| 各 benchmark 具體分數 | 論文未於可存取來源提供 / 未查到 | — | — |

## 限制 / 備註

- 純 benchmark 研究，未提出新量化演算法；也沒有針對 dLLM 的 timestep / mask 特性設計校準資料（這正是 DLLMQuant、Quant-dLLM 後續改進點）。
- 未報告實際 kernel 加速或記憶體數字。
- 期刊版標題與 arXiv 相同，內容可能較 arXiv v1 有更新（v3 存在）。

## 與其他論文的關係

- **與 DLLMQuant (2508.14090) 同月發表、互補**：本篇是「系統評估」，DLLMQuant 是「針對 dLLM 設計的新 PTQ」；兩者共同確立「AR PTQ 直接搬到 dLLM 在 W4A4 會嚴重掉分」的共識。
- **Quant-dLLM (2510.03274)** 引用本篇的 outlier 分析，並把 bit-width 推進到 2-bit weight-only。
- **STaR-Quant (2606.04945)、FAIR-Calib (2606.06547)** 進一步把本篇觀察到的「多步誤差累積」形式化為 temporal error accumulation / frontier decision flip。
- 沿用 AR LLM 的 QuaRot、DuQuant、SmoothQuant、GPTQ、AWQ 等方法作為基線。
