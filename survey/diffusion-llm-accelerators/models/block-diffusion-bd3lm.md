# Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models (BD3-LM)

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2503.09573`（ICLR 2025 Oral） |
| 作者 / 單位 | Marianne Arriola, Aaron Gokaslan, Justin T. Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham S. Sahoo, Volodymyr Kuleshov（Cornell Tech；合作 Stanford、Cohere） |
| 日期 | 2025-03 |
| 類別 | 模型 |
| 連結 | [arXiv](https://arxiv.org/abs/2503.09573) · [GitHub](https://github.com/kuleshov-group/bd3lms) · [Project](https://m-arriola.com/bd3lms/) |

## 一句話總結

提出 Block Discrete Denoising Diffusion LM：序列切成固定大小的 block，block 之間自迴歸、block 內部做離散擴散，從而同時獲得 AR 的 KV cache 與任意長度生成、以及擴散的平行取樣；並以 clipped 噪聲排程降低梯度變異，在 LM1B / OWT 上取得擴散模型中最佳 perplexity（比 MDLM 好 13%）。

## 要解決的問題

- 全序列離散擴散 LM（SEDD、MDLM）有三個結構性缺陷：(1) 只能生成訓練長度的固定序列，無法變長；(2) 每步全序列重算，無法用 KV cache；(3) 似然（perplexity）仍明顯落後 AR。
- 作者發現擴散 LM 與 AR 的 perplexity 差距很大一部分來自「訓練梯度變異過高」（隨機遮罩比例造成損失估計噪聲），而非模型容量。

## 核心方法與特色

- **Block-autoregressive 似然分解**：p(x) = ∏_b p(x^b | x^{<b})，每個 block（大小 L'）的條件分布用 masked diffusion 建模。L'=1 退化為 AR，L'=序列長度退化為 MDLM，因此是兩者的插值；可任意接續生成新 block 達成變長輸出。
- **單次 forward 的向量化訓練**：設計特殊注意力 mask，把「噪聲序列」與「乾淨序列」串接（2L×2L mask）在一次 forward 中同時計算所有 block 的去噪損失，避免逐 block 前向的訓練開銷；乾淨前綴的 KV 可直接快取。
- **KV cache + 平行取樣**：推論時已完成的 block 其 KV 固定不變，可像 AR 一樣快取；當前 block 內以擴散步數 T 平行去噪，NFE ≈ (序列長度 / L') × T，比 MDLM 少且可變長。
- **Clipped（資料驅動）噪聲排程**：理論分析顯示遮罩比例極端（接近 0 或 1）時梯度變異大；改為從 U[β, ω]（如 [0.45, 0.95]）取樣遮罩率並依 block 大小自適應搜尋，L'=4 時 PPL 由 30.18 降到 29.21、梯度變異由 23.45 降到 6.24。
- **代價**：block 之間仍是序列化，失去全域雙向規劃；小 block 品質接近 AR 但平行度低，大 block 平行度高但 perplexity 退化；實驗僅到 110M 規模。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| BD3-LM (L'=4 / 8 / 16) | 110M（12 層、768 hidden、12 heads，GPT-2 small 級） | OWT 1M steps（850K 預訓 + 150K 微調）；LM1B 亦有 |
| AR baseline | 110M | 同架構 |
| MDLM、SEDD | 110M | 全序列擴散對照 |
| SSD-LM | — | semi-AR 連續擴散對照（L'=25） |

## PPA / 效能數據

> 軟體論文：speedup、throughput (tok/s)、latency、記憶體、benchmark 準確度變化。

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| LM1B perplexity | L'=4 ≤28.23；L'=8 ≤29.83；L'=16 ≤30.60 | AR 22.83；MDLM ≤31.78；SEDD ≤32.68 → 比 MDLM 好 ~13% | 110M，65B tokens |
| OWT perplexity | L'=4 ≤20.73；L'=8 ≤21.68；L'=16 ≤22.27 | AR 17.54；MDLM ≤22.98；SEDD ≤24.10 | 110M，524B tokens |
| L'=1 與 AR 的差距 | ≈2 PPL | AR | LM1B，顯示訓練變異是主因 |
| 生成 perplexity（變長 2048 tokens） | L'=4：23.6；L'=8：28.2 | AR 13.2；SSD-LM (L'=25) 35.3 | NFE：BD3-LM 2K vs SSD-LM 80K |
| 噪聲排程消融（L'=4） | Clipped U[0.45,0.95]：PPL 29.21 / 梯度變異 6.24 | 線性 U[0,1]：30.18 / 23.45 | LM1B |
| KV cache | 支援（block 完成後固定） | MDLM 不支援 | — |
| 絕對 tok/s | 論文未提供 | — | — |

## 限制 / 備註

- 規模僅 110M（GPT-2 small 級），未驗證於 1B+；「block diffusion 在大模型上可行」是由後續 LLaDA2.0（100B）、SDAR（30B）、Fast-dLLM v2、Stable-DiffCoder 證明。
- 生成 perplexity 仍明顯高於 AR（23.6 vs 13.2），且 block 越大越差；後續研究（Diffusion-in-Diffusion 2601.13599）指出 semi-AR block 生成有全域一致性缺口。
- 序列長於訓練長度（>1024）需去掉 BOS/EOS 重訓。
- 無 tok/s 量測；「速度優勢」在本文停留在 NFE 與 KV cache 可行性層面。

## 與其他論文的關係

- 建立在 MDLM（同團隊 NeurIPS 2024）與 SEDD 的離散擴散框架，以及 SSD-LM 的 semi-AR 想法之上；是 LLaDA 「semi-autoregressive remasking」的原則化版本。
- 幾乎所有 2025 下半年後的實用 dLLM 都採用其 block diffusion 形式：LLaDA2.0（WSD 課程放大再縮回 block）、SDAR（Qwen3 轉換、block 4）、Fast-dLLM v2（block 32 / sub-block 8 + 階層式 cache）、Efficient-DLM（block-wise attention）、Nemotron-Labs-Diffusion（32-token block）、Stable-DiffCoder（block-wise clipped noise schedule 直接沿用本文）。
- 同團隊後續 Set Diffusion（ICML 2026）把「固定順序、固定大小的 block」推廣為「任意順序、可變大小的 token 集合」以取回解碼彈性與更高平行度；Eso-LM 則把 AR 與 MDM 融合為另一種插值。
- dLLM 框架（2602.22661）內建 BD3LM 訓練演算法。
