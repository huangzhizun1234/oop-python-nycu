# d²Cache: Accelerating Diffusion-Based LLMs via Dual Adaptive Caching

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2509.23094`（ICLR 2026） |
| 作者 / 單位 | Yuchu Jiang, Yue Cai, Xiangzhong Luo, Jiale Fu, Jiarui Wang, Chonghan Liu, Xu Yang（東南大學等；單位依記憶，待確認） |
| 日期 | 2025-09 |
| 類別 | KV cache 與稀疏 |
| 連結 | [arXiv](https://arxiv.org/abs/2509.23094) · [GitHub](https://github.com/Kamichanw/d2Cache) |

## 一句話總結

以「兩階段細粒度 token 選擇」決定每步只重算哪些 token 的 KV（mask token 用確定性先驗、非 mask token 用注意力分數），其餘沿用快取，並順帶得到近似由左至右的穩定解碼，平均 3.5×、最高 4.7× 加速且品質持平或提升。

## 要解決的問題

- dLLM 的雙向注意力使標準 KV cache 不可用；既有近似 cache（Fast-dLLM 的 block cache、dLLM-Cache 的固定比例更新）粒度粗，要嘛整個 block 全算、要嘛用固定比例挑 token，無法針對「真正在變」的 token 做自適應更新。
- dLLM 以信心值做 remask 時，序列末端的 token 常出現「過早過度自信」（premature overconfidence），導致品質不穩。

## 核心方法與特色

- **兩階段 token 選擇（dual adaptive）**：每個 decoding step 先做 (1) **deterministic-prior-guided masked token selection**：對 mask token 依「確定性先驗」（certainty prior，反映該位置在當前步驟被解碼的可能性）挑出需要更新 KV 的少數 mask token；再做 (2) **attention-aware non-masked token selection**：對已解碼 / prompt token 依注意力貢獻挑出仍需刷新的 token。兩組之外的 token KV 全部沿用快取。
- **自適應更新而非固定週期**：更新集合的大小隨 step 動態變化，早期 step 更新多、後期收斂後更新少，避免 dKV-Cache / Fast-dLLM 固定間隔的浪費或不足。
- **Certainty-prior 解碼（quasi left-to-right）**：因為 mask token 的選擇本身依 certainty prior，順帶得到一種「依先驗而非預測信心」的解碼順序，自然趨向由左至右，抑制序列尾端 token 的過早過度自信，因此在多個 benchmark 上準確度反而提升（例如 Dream-Inst MBPP 52.0→58.0）。
- **Training-free、與其他方法相容**：不改模型權重，可套用在 LLaDA、Dream、SDAR；repo 同時實作了多種 caching 與 decoding 策略供比較。
- **代價**：每步需計算 certainty prior 與注意力統計以做選擇，帶來額外開銷；在極短生成長度上加速有限。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| LLaDA-8B-Instruct（LLaDA-Inst） | 8B | 主要實驗 |
| Dream-v0-Instruct-7B（Dream-Inst） | 7B | 主要實驗 |
| LLaDA-1.5 | 8B | repo 支援 |
| SDAR-8B | 8B | repo 支援 |

## PPA / 效能數據

> 軟體論文：speedup、throughput (tok/s)、latency、記憶體、benchmark 準確度變化。GPU 型號未查到。

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| 平均加速 | 3.5× | Vanilla（原始實作） | LLaDA-Inst + Dream-Inst，GSM8K / MBPP / HumanEval / MATH-500 等 |
| GSM8K 吞吐 | 2.62 → 12.25 tok/s（4.7×） | Vanilla | 搜尋來源對模型標示不一（Dream-Inst 或 LLaDA-Inst），待確認 |
| Dream-Inst MBPP | 分數 52.0 → 58.0，4.6× 加速 | Vanilla | — |
| 品質 | 六個資料集平均分數持平或優於 Vanilla | Vanilla / dLLM-Cache / Fast-dLLM | — |
| 與 baseline 比較 | 平均吞吐與延遲皆優於 dLLM-Cache 與 Fast-dLLM | — | 兩模型、全部 benchmark |

## 限制 / 備註

- 具體逐 benchmark 表格與 GPU 型號未從 README 取得（README 以連結指向論文）。
- 加速倍率（3.5–4.7×）低於激進的 suffix pruning 類方法（DPad、Streaming-dLLM），但特點是幾乎無損甚至提升品質。
- Certainty prior 解碼改變了原始的信心排序，屬於「解碼策略」與「cache」耦合設計，與其他純 cache 方法組合時需注意衝突。

## 與其他論文的關係

- 直接以 **dLLM-Cache** 與 **Fast-dLLM** 為 baseline，主張細粒度自適應選擇優於固定比例 / 固定 block。
- 與 **Elastic-Cache**（注意力漂移決定刷新時機）、**DyLLM**（saliency 決定重算 token）屬同一類「自適應選 token 重算」的思路，三者幾乎同期（2025-09 至 2026-03）。
- 「quasi left-to-right」與 **DPad / Streaming-dLLM** 對 suffix 的觀察一致：遠端 suffix token 貢獻小、不應過早決定。
- repo 內含 SDAR 支援，與 block-diffusion 類模型（SDAR、Fast-dLLM v2）相容。
