# Fast and Accurate Causal Parallel Decoding using Jacobi Forcing

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2512.14681`（第三方摘要提及 ICML 2026，待確認） |
| 作者 / 單位 | Lanxiang Hu, Siqi Kou, Yichao Fu, Samyam Rajbhandari, Tajana Rosing, Yuxiong He, Zhijie Deng, Hao Zhang（UC San Diego Hao AI Lab；依記憶另含 Snowflake、上海交通大學，待確認） |
| 日期 | 2025-12 |
| 類別 | 平行與投機解碼（**AR 模型**的因果平行解碼；作為 dLLM 對照） |
| 連結 | [arXiv](https://arxiv.org/abs/2512.14681) · [GitHub](https://github.com/hao-ai-lab/JacobiForcing) · [Blog](https://hao-ai-lab.github.io/blogs/jacobi-forcing/) |

## 一句話總結

不把 AR 模型改成 diffusion，而是保留 causal backbone，用「漸進式 consistency 蒸餾」訓練模型沿著自己的 Jacobi 解碼軌跡處理帶噪的未來 block，得到一個仍由左到右、但每次 forward 可解 4 個以上 token 的 AR 模型：HumanEval 4.0x、GSM8K 3.7x wall-clock 加速，品質接近 AR，且 KV cache 完全可用。

## 要解決的問題

- 把 AR 預訓練模型改造成 dLLM（Dream、Fast-dLLM v2 等）存在 pretrain-to-posttrain mismatch：masked 訓練分佈與 causal 預訓練分佈不同、雙向 attention 與 causal attention 衝突，導致品質下降（同表中 dLLM 方法 HumanEval 僅 53–54%）。
- dLLM 的雙向注意力破壞標準 KV cache，實際 serving 不友善。
- 既有 Jacobi / consistency 類 AR 平行解碼（CLLM）加速有限（2–2.5x），因為訓練時沒有讓模型學會在「未來 block 很吵」的情況下也給出好草稿。

## 核心方法與特色

- **Jacobi 解碼作為骨架**：一次把一個 n-token block 全填成猜測，做 forward 得到每個位置在給定前面猜測下的預測，迭代到固定點；因為 attention 仍是 causal，已確定的 prefix 可以 KV cache。
- **Jacobi Forcing 訓練（progressive distillation）**：先從基座 AR 模型收集每個 block 的 Jacobi 軌跡（中間狀態 + 固定點），再用一個從乾淨到全噪的漸進式 noise schedule 把「帶噪未來 block」拼進訓練序列；損失 = progressive consistency loss（帶噪 block 的預測要對齊固定點）+ AR loss，並用特製 attention mask 讓乾淨 block 與噪音 block 在一次 forward 內同時算 logits。
- **Multiblock decoding + rejection recycling**：同時讓 K 個 block「在飛」，早期迭代中出現的高品質 n-gram 被回收當作後續 draft（類似 lookahead），提高 GPU 利用率；建議 n=64, K=2, pool_size=4, r=0.85。
- **與 dLLM 的對照意義**：作者在同一張表上直接比較 dLLM（D2F、Fast-dLLM、dParallel）與 SD（EAGLE-3、HASS）；結論是 dLLM 路線在把 AR 模型轉成平行解碼器時付出的品質代價太大，而保持 causal 的 Jacobi Forcing 能以小規模微調得到近 AR 品質與 3–4x 加速。
- **代價**：需要蒸餾訓練（自身軌跡）；平行度來自 block 內收斂速度，上限低於激進的 dLLM（TPF 約 4 vs DMax 的 6+）。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| Qwen2.5-Coder-7B-Instruct（AR 基座） | 7B | → JacobiForcing_Coder_7B（OpenCodeInstruct 資料） |
| Qwen2.5-Math-7B-Instruct（AR 基座） | 7B | → JacobiForcing_Math_7B（OpenThoughts2 math split） |
| 對照 dLLM：D2F、Fast-dLLM、dParallel | 7–8B | 作者以第三方身份量測 |

## PPA / 效能數據

（官方 README 表；GPU 未載明主表型號，另提及 B200）

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| HumanEval 加速 / TPS | 4.0x，163.9 TPS，TPF 4.1 | AR 41.3 TPS（87.8%） | Jacobi Forcing (MR)，準確度 83.5% |
| GSM8K 加速 / TPS | 3.7x，154.9 TPS，TPF 4.0 | AR 41.8 TPS（92.4%） | Jacobi Forcing (MR)，準確度 91.4% |
| 對照 dLLM（HumanEval） | D2F 1.8x / 54.3%；Fast-dLLM 1.5x / 53.0%；dParallel 2.1x / 54.3% | AR | 同表 |
| 對照 dLLM（GSM8K） | D2F 2.2x / 77.6%；Fast-dLLM 1.2x / 75.0%；dParallel 3.1x / 82.9% | AR | 同表 |
| 對照 SD | EAGLE-3 2.9–3.3x；HASS 3.1–3.4x | AR | 作者註明 SD 的 TPF→TPS 轉換率較差 |
| 最高 TPF | up to 4.5x | AR | 摘要 |
| 單卡引擎 throughput | 800–1000 tok/s | — | 自製 nano-vLLM 式引擎（paged KV、CUDA graph） |
| B200 | ≈330 tok/s vs AR ≈80 tok/s | AR | multiblock + rejection recycling |
| 對話 demo | 181.8 vs 39.81 TPS（>4x） | AR Qwen2.5-Coder-7B | 程式對話 |

## 限制 / 備註

- **這是 AR 模型的方法，不是 dLLM 加速**；收錄理由是它提供了對 dLLM 路線最直接的批評與同表對照（品質—速度），並展示「block-causal + 平行填充」的另一種實現。
- HumanEval 準確度從 87.8% 降到 83.5%（−4.3 點），並非完全無損。
- 表中 dLLM 對照數字由 Jacobi Forcing 作者量測，設定可能與各原論文不同。

## 與其他論文的關係

- 對 **dParallel、Fast-dLLM、D2F** 做了直接對比，是評估「AR→dLLM 改造是否值得」的重要參考。
- 與 **CDLM / SDAR / Fast-dLLM v2** 都收斂到 block-causal 平行解碼，但 Jacobi Forcing 從 AR 端出發、attention 完全 causal。
- 與 **DARTree / DFlash**（diffusion drafter 加速 AR）目標相同（無損或近無損加速 AR），但 Jacobi Forcing 是單模型、無 draft–verifier。
- 前身為 **CLLM**（Consistency LLM）與 lookahead decoding（同實驗室）。
