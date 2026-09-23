# STaR-Quant: State-Time Consistent Post-Training Quantization for Diffusion Large Language Models

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2606.04945` |
| 作者 / 單位 | Xin Yan, Aqiang Wang, Zhenglin Wan, Xingrui Yu, Ivor Tsang（北京師範大學人工智慧學院 / 新加坡國立大學 / A*STAR CFAR） |
| 日期 | 2026-06 |
| 類別 | 量化 |
| 連結 | [arXiv](https://arxiv.org/abs/2606.04945) · GitHub：未查到 |

## 一句話總結

針對 dLLM 低 bit（W4A4）量化提出「狀態一致 + 時間一致」的 PTQ：SGAT 把 masked / unmasked token 送進不同的激活變換空間（weight 側共用一個靜態變換），TAC 用輕量 block-diagonal affine 映射補償跨 denoising step 累積的 attention 誤差，在 LLaDA / LLaDA-1.5 / Dream 上優於 RTN、AWQ、QuaRot、DLLMQuant。

## 要解決的問題

- **State-dependent activation disparity**：同一個 denoising step 內，masked token 與已 unmask 的 token 激活分佈明顯不同（mask token 的 embedding 相同、且沒有內容資訊），用同一組 smoothing / rotation 參數量化兩類 token 必然有一方失真。
- **Temporal error accumulation**：dLLM 需要多步迭代，第 t 步的量化誤差會透過 attention 傳到 t+1 步，逐步累積；AR PTQ 只針對單次 forward 最小化誤差，沒有機制抑制跨步累積。
- 既有 dLLM PTQ（DLLMQuant）雖考慮 timestep 校準，但沒有把 mask 狀態的差異在「變換空間」層級分開處理，也沒有顯式補償 attention 誤差。

## 核心方法與特色

- **SGAT（State-Guided Activation Transformation）**：依 token 是 masked 還是 unmasked，將其激活分派到兩個不同的變換空間（各自的 scaling / rotation），但在 weight 側只用一個統一的 static transformation，因此權重仍可離線量化、不需要兩份權重；推論時只需依 mask 狀態選擇對應的激活變換，開銷極小。效果是讓兩類 token 在各自空間內分佈更集中，4-bit 激活格點利用率提高。
- **TAC（Temporal Attention Compensation）**：在 attention 輸出上加一個輕量的 block-diagonal affine 映射（少量可學參數），以校準資料上「量化模型 vs. FP 模型的 attention 表徵差距」為目標離線擬合，用來校正量化後的 attention 表徵，使誤差不會在下一步 denoising 中被放大。block-diagonal 結構保證額外 FLOPs 與參數量都很小。
- **純 PTQ、不需微調**：兩個模組都只用校準資料離線求解，不需 end-to-end 訓練。
- **Ablation 顯示兩模組互補**：LLaDA-8B W4A4 下，完整方法平均 57.07；拿掉 TAC 掉到 55.43，拿掉 SGAT 掉到 56.39，表示時間誤差累積（TAC）對 4-bit 激活的影響略大於狀態差異（SGAT），但兩者疊加最好。
- **代價**：需要為 masked / unmasked 各維護一套激活變換參數；TAC 的 affine 映射是額外算子，實際 kernel 加速數字論文未提供。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| LLaDA-8B | 8B | 主要 ablation 模型 |
| LLaDA-1.5-8B | 8B | |
| Dream-7B | 7B | |

## PPA / 效能數據

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| 9 項任務平均（量化後） | 57.07 | w/o TAC 55.43；w/o SGAT 56.39 | LLaDA-8B，W4A4，ablation |
| vs. 既有 PTQ | 在 W4A4 下一致優於 RTN、AWQ、QuaRot、DLLMQuant（具體差距未查到） | RTN / AWQ / QuaRot / DLLMQuant | LLaDA-8B、LLaDA-1.5-8B、Dream-7B，9 個任務（知識、推理、程式碼） |
| FP16 平均分數 | 論文未於可存取來源提供 / 未查到 | — | — |
| speedup / 記憶體 | 論文未提供 / 未查到 | — | — |

## 限制 / 備註

- 2026-06 的新作，目前僅 arXiv 版本，未查到程式碼與詳細數表。
- 只評估 W4A4；W2 極低 bit weight（Quant-dLLM 的設定）與 W8A8 未涵蓋。
- TAC 的 affine 映射在 block-diffusion / KV-cache 加速（如 Fast-dLLM）框架下如何配合尚未討論。

## 與其他論文的關係

- 直接把 **DLLMQuant (2508.14090)** 當基線並超越之；兩者都關注 timestep 與 mask，但 STaR-Quant 在「變換空間」與「attention 補償」層級做區分。
- 沿用 **QuaRot / DuQuant** 的 rotation 思想（unified weight-side transform），並延伸為 state-guided 的雙激活空間。
- 與 **FAIR-Calib (2606.06547)** 同月發表、切入角度互補：FAIR-Calib 從「校準損失加權」保護 write frontier 的脆弱決策，STaR-Quant 從「激活變換 + attention 補償」處理狀態/時間不一致。
- 觀察到的 temporal error accumulation 與 **Quantization Meets dLLMs (2508.14896)** 指出「生成型任務對量化最敏感」一致。
