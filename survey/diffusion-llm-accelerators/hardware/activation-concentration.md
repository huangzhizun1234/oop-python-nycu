# Activation Concentration: Characterizing Column-Level Output Sparsity Across Diffusion Model Architectures

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2606.00567`（cs.PF / cs.AR） |
| 作者 / 單位 | Dazhi Yang, Shafayat Mowla Anik, Byeong Kil Lee, Jeeho Ryoo（Fairleigh Dickinson University；University of Colorado at Colorado Springs） |
| 日期 | 2026-06 |
| 類別 | 硬體特性分析（活化稀疏度） |
| 連結 | [arXiv](https://arxiv.org/abs/2606.00567) · GitHub：未查到 |

> **重要備註：本文只涵蓋影像 / 音訊 / 影片 / 動作 (motion) 的連續 diffusion 模型，七個工作負載中沒有任何 diffusion 語言模型。** 收錄原因：它揭示「element-level 活化稀疏 ≠ systolic array 可利用的 column-level 稀疏」，這個方法論與結論對 dLLM 加速器（同樣以 GELU/SiLU FFN + systolic array 為主）設計直接適用，見「對 dLLM 加速器設計的啟示」。

## 一句話總結

第一個跨七個 diffusion 工作負載（三類架構、四種模態）的 column-level 活化稀疏特性研究：既有 diffusion 加速器宣稱跳過近零 GELU 輸出可得 52~85% element-level 稀疏，但 systolic array 是以「整欄 (column)」為單位計算，只要欄內有一個非零就整欄要算——真正可利用的稀疏度被高估最多 78 個百分點；UNet+Transformer 類展現「activation concentration」（非零集中在少數熱欄）可減 cycle 最多 30.6%，純 Transformer DiT 稀疏分散只有 12.4%，動作/舞蹈 Transformer 從小幅到 MLD 的 50.8%。

## 要解決的問題

- 近期 diffusion 加速器（跳過 GELU 近零輸出的 sparsity-aware 設計）用 element-level 稀疏度（52~85%）估算收益，但 weight-stationary / output-stationary systolic array 的資料流是整欄輸入一起推進，無法逐元素跳過；硬體實際能省的是「整欄全零」的比例。
- 沒有人系統性地量測不同 diffusion 架構（UNet+Transformer、純 DiT、動作 Transformer）在 column 粒度的稀疏度、其隨去噪迭代的變化、以及對 systolic array cycle 的真實影響。

## 核心方法與特色

- **Column-level 稀疏度定義與量測**：對每個 FFN/GELU 輸出矩陣，以 systolic array 的欄粒度統計「整欄為零（或全部低於門檻）」的比例，並與 element-level 稀疏度並列，得到「高估幅度」（最多 78 個百分點）。
- **七個工作負載、四種模態**：UNet+Transformer 類（Stable Diffusion 影像、Make-an-Audio 音訊、VideoCrafter2 影片）、純 Transformer 類（DiT 影像）、動作/舞蹈 Transformer 類（MDM、EDGE、MLD）。
- **三分類 taxonomy**：(1) **Concentration**（UNet+Transformer）：非零元素集中在少數「熱欄」，且熱欄集合在去噪早期就大致固定，模型從稠密迅速跳到稀疏區間 → 適合靜態 hot/cold 欄位佈局；(2) **Dispersion**（DiT）：迭代 0 有 ~42.6% column 稀疏，之後隨迭代逐步下降、非零分散到更多欄 → 靜態佈局無效；(3) 動作/舞蹈 Transformer：差異大，MLD 因 token 維度極大且 FFN 展開比高而得到最大收益。
- **Cycle-level 硬體模型**：用 systolic array 的 cycle 模型換算成實際 cycle 減少，而不是只報告稀疏比例；並提出「靜態 hot-cold column layout」（先算熱欄、跳過冷欄）作為 concentration 型工作負載的低成本利用方式。
- **跨迭代動態分析**：追蹤 column 稀疏度隨去噪步的曲線，指出稀疏結構在時間上是否穩定決定了硬體能否用靜態排程利用它。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| Stable Diffusion（UNet） | ~0.86B UNet | 影像；UNet+Transformer 類 |
| Make-an-Audio | 未查到 | 音訊；UNet+Transformer 類 |
| VideoCrafter2 | 未查到 | 影片；UNet+Transformer 類 |
| DiT | 未查到（推測 DiT-XL/2, ~675M） | 影像；純 Transformer 類 |
| MDM / EDGE / MLD | 未查到 | 動作 / 舞蹈生成 Transformer |
| （無 dLLM） | — | **未評估任何 diffusion 語言模型** |

## PPA / 效能數據

> 特性分析論文：稀疏度與模擬 cycle 減少；無新硬體 PPA。

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| 既有加速器宣稱的 element-level 稀疏 | 52% ~ 85% | — | GELU 近零輸出 |
| Element vs column 稀疏高估 | 最多 78 個百分點 | column-level 實測 | 跨七個工作負載 |
| Cycle 減少（UNet+Transformer） | 最多 30.6% | 稠密 systolic array 執行 | 依工作負載而異 |
| Cycle 減少（DiT） | 12.4% | 同上 | 稀疏分散 |
| Cycle 減少（動作/舞蹈） | 小幅 ~ 50.8%（MLD） | 同上 | MLD 因 token 維度與展開比極端 |
| DiT 初始 column 稀疏 | ~42.6%（迭代 0），之後遞減 | — | — |
| Systolic array 尺寸、製程 | 未查到 | — | cycle 模型 |

## 限制 / 備註

- **不含 dLLM**：七個工作負載都是連續 diffusion（影像/音訊/影片/動作）。dLLM（LLaDA/Dream）用的是 SwiGLU FFN 而非 GELU，activation 稀疏度的基礎統計可能更低（SwiGLU 的門控輸出不像 GELU 那樣有大量近零區），本文結論不能直接套用數字，只能套用方法論。
- 只分析 FFN/GELU 輸出稀疏，未涵蓋 attention 稀疏或 KV cache。
- 沒有實作硬體，cycle 減少來自 systolic array 分析模型。

## 對 dLLM 加速器設計的啟示

- **稀疏度必須以硬體粒度量測**：DART、block-diffusion-edge-hw 等 dLLM 加速器若要利用「未更新 token 的活化稀疏」或 FFN 稀疏，應以 systolic array 的 column/tile 粒度評估，而非 element-level；本文顯示高估可達 78 個百分點。
- **dLLM 的 token 級稀疏比元素級更「結構化」**：dLLM 每步只有部分 token 改變，這是「整欄/整列（token 維度）」粒度的稀疏，天生比 GELU 元素稀疏更適合 systolic array——這正是 block-diffusion-edge-hw 的 DAT-FFN（依 token 飄移決定是否重算）與 MASQ 的 mask-aware 精度所利用的結構。
- **時間穩定性決定靜態排程可行性**：本文的「concentration（熱欄早期固定）vs dispersion」分析方法，可用來檢驗 dLLM 中「已確定 token 的 KV/FFN 輸出是否在後續步驟保持穩定」，這正是 cross-step cache（dLLM-Cache、DiPe 的跨步聚合）的硬體前提。
- 純 Transformer（DiT）稀疏分散的結論提醒：dLLM 全是純 Transformer，若期待 GELU/SiLU 元素稀疏帶來收益，很可能只有 ~10% 級別，重點應放在 token 級稀疏與 KV/權重流量。

## 與其他論文的關係

- 直接回應宣稱 52~85% 稀疏的影像 diffusion 加速器（SQ-DM 的時間稀疏、DiSC、RT-Lynx 等 sparsity-aware 設計），指出其收益估計方法的問題。
- 與 **masq.md** 同為影像 diffusion 硬體研究：MASQ 利用 mask 區域的精度冗餘，本文分析活化稀疏冗餘；兩者都是「非 GEMM 全稠密」的利用方式。
- 對 **DART (npu-dllm-sampling.md)** 與 **block-diffusion-edge-hw.md** 提供方法論：任何宣稱利用 dLLM 稀疏的硬體都應報告 column/tile 粒度的真實可跳過比例。
