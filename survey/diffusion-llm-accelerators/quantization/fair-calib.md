# FAIR-Calib: Frontier-Aware Instability-Reweighted Calibration for Post-Training Quantization of Diffusion Large Language Models

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2606.06547`（ICML 2026 poster） |
| 作者 / 單位 | 未查到（arXiv 摘要頁被 proxy 阻擋） |
| 日期 | 2026-06 |
| 類別 | 量化 |
| 連結 | [arXiv](https://arxiv.org/abs/2606.06547) · [ICML 2026](https://icml.cc/virtual/2026/poster/62977) · GitHub：未查到 |

## 一句話總結

觀察到 dLLM「先寫入、後才穩定」的 stability lag：量化誤差最容易在 write frontier 翻轉邊界決策並被永久鎖定，因此提出兩階段 PTQ 校準——先用 FP teacher 估計每個位置的脆弱度先驗，再以此重加權 layer-wise hidden-state MSE，在 LLaDA / Dream W4A4 上優於 SOTA 基線且只需較短的校準序列。

## 要解決的問題

- dLLM 以迭代方式精煉 token，但一旦某個位置被 unmask（commit）就不可逆；作者發現剛被寫入的決策在後續幾步內仍然「脆弱」（stability lag），此時若量化誤差把 logits 邊界翻轉，錯誤 token 會被鎖定並在之後放大。
- 既有 PTQ 校準（含 DLLMQuant 的 timestep 抽樣）對所有位置的誤差一視同仁，沒有針對這些「write frontier 上的邊界決策」加強保護；而直接用 end-to-end 多步 diffusion rollout 做校準又太貴。

## 核心方法與特色

- **Stage I：Teacher probing 估計位置先驗**：用 FP 模型跑生成，統計每個位置 (a) 成為 write frontier 的次數（frontier hits）與 (b) 在 masked 階段預測的可靠度，兩者合成一個 position prior，標記出「最容易被量化誤差翻轉」的脆弱狀態。
- **Stage II：加權 layer-wise 校準**：以 off-policy 方式（不需重新 rollout 量化模型）逐層最小化「重加權的 hidden-state MSE」，權重即 Stage I 的先驗——脆弱 frontier 位置的重建誤差被放大，穩定位置則降權；因此校準資源集中在真正決定輸出的地方。
- **理論保證**：證明此加權目標是輸出 KL divergence 的 surrogate（上界），把「保護 frontier」與「維持輸出分佈」連結起來。
- **校準成本低**：預設校準序列長度僅 1024（短於多數基線），且不需要昂貴的 end-to-end diffusion rollout。
- **代價 / 限制**：Stage I 需要額外跑 FP teacher 的生成軌跡；先驗與生成設定（block size、steps、remasking 策略）綁定，換解碼設定可能需重估。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| LLaDA-8B | 8B | W4A4 |
| Dream-7B | 7B | W4A4 |

## PPA / 效能數據

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| W4A4 各 benchmark 準確度 | 一致優於 SOTA PTQ 基線（具體數字論文未於可存取來源提供 / 未查到） | 既有 dLLM PTQ 基線 | LLaDA-8B、Dream-7B，W4A4 |
| Frontier decision flips | 顯著減少 | 基線 PTQ | 同上 |
| Post-commit mismatch | 顯著抑制 | 基線 PTQ | 同上 |
| 校準序列長度 | 1024（預設） | 基線通常較長 | — |
| speedup / 記憶體 | 論文未提供 / 未查到 | — | — |

## 限制 / 備註

- 2026-06 新作，可存取來源僅有摘要層級的描述，缺乏具體數表與作者資訊。
- 只評估 W4A4；是否適用於 weight-only 2-bit 未知。
- 先驗依賴 FP teacher 的生成行為，若部署時解碼策略改變（例如改用 Fast-dLLM 的 parallel decoding），frontier 分佈可能改變。

## 與其他論文的關係

- 與 **STaR-Quant (2606.04945)** 同月發表、互補：STaR-Quant 改「變換空間與 attention 補償」，FAIR-Calib 改「校準損失的加權」；兩者可疊加。
- 延伸 **DLLMQuant (2508.14090)** 的 CGQ（用 mask 狀態與置信度加權誤差補償）思路，但把加權依據換成「frontier 脆弱度」並給出 KL surrogate 的理論證明。
- 與 **Quantization Meets dLLMs (2508.14896)** 觀察到的「生成型任務對量化最敏感」一致，並給出機制解釋（frontier flip 被鎖定放大）。
