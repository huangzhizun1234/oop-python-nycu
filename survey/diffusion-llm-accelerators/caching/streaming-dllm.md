# Streaming-dLLM: Accelerating Diffusion LLMs via Suffix Pruning and Dynamic Decoding

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2601.17917` |
| 作者 / 單位 | Zhongyu Xiao, Zhiwei Hao, Jianyuan Guo, Yong Luo, Jia Liu, Jie Xu, Han Hu（單位未查到；依作者背景推測含北京理工大學 / 武漢大學，待確認） |
| 日期 | 2026-01 |
| 類別 | KV cache 與稀疏（suffix 剪枝 + 動態解碼） |
| 連結 | [arXiv](https://arxiv.org/abs/2601.17917) · [GitHub](https://github.com/xiaoshideta/Streaming-dLLM) |

## 一句話總結

從空間（剪掉遠端 suffix mask token，只保留當前 block 附近與尾端的近似 suffix）與時間（依信心動態決定每步解碼數並對已收斂 block 提早退出）兩個維度精簡 dLLM 推論，在 LLaDA-1.5 MBPP 上達到 68.2× 加速且品質持平或略優。

## 要解決的問題

- **空間冗餘**：dLLM 對所有 suffix mask token 一視同仁地建模，但遠端 suffix 對當前 block 的資訊貢獻隨距離衰減，卻佔用大量注意力計算。
- **時間低效**：固定的 denoising 步數表對所有 block 一樣，即使某個 block 的 token 已經全部收斂仍要跑完剩餘步數。
- 既有 KV cache 方法只處理 prefix，沒有處理 suffix 與步數兩個維度。

## 核心方法與特色

- **Attenuation-guided suffix modeling（近似 suffix 剪枝）**：分析注意力衰減發現 suffix 的影響集中在「緊接當前 block 的鄰近區域」與「序列尾端位置」，因此只保留這兩段 mask token 拼接成近似 suffix，其餘 mask token 直接不參與 forward；每步序列長度從 L 縮到接近 block + 小視窗，且與生成長度無關。
- **Dynamic confidence-aware parallel decoding**：不使用固定步數 / 固定每步 token 數，而是依信心分布動態決定本步解幾個 token（承襲 Fast-dLLM 閾值策略並做調整），信心高時大量解碼。
- **Early exit for block diffusion**：當前 block 內所有 token 皆已達收斂條件時，跳過該 block 剩餘的 denoising 迭代直接進入下一 block，減少總步數。
- **Training-free、與 prefix cache 相容**：可疊在 Fast-dLLM 式 KV cache 上，三個元件相乘後在 LLaDA-1.5 上得到數十倍加速。
- **代價**：近似 suffix 使模型看不到中段 suffix，對需要全局規劃的長文本可能有影響；early exit 與動態解碼閾值需調參；加速倍率在 Dream 與 Open Pangu 上遠低於 LLaDA-1.5（1.4–13×），顯示效果依模型而異。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| LLaDA-8B-Instruct | 8B | 論文評測 |
| LLaDA-1.5 | 8B | 加速最大（68.2×） |
| Dream-7B | 7B | 3.7–13.3× |
| openPangu（Embedded 7B diffusion） | 7B | 1.4–1.6× |

## PPA / 效能數據

> 軟體論文：speedup、throughput (tok/s)、latency、記憶體、benchmark 準確度變化。GPU 型號未查到。加速倍率相對各模型原始（vanilla）實作。

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| 最高加速 | 68.2×（61.4 TPS，38.4%） | vanilla LLaDA-1.5 | MBPP, gen 512 |
| LLaDA-1.5 GSM8K | 28.0×（69.8 TPS，81.2%） | vanilla LLaDA-1.5 | gen 512 |
| LLaDA-1.5 MATH | 8.5×（66.2 TPS，33.7%） | vanilla | gen 256 |
| Dream HumanEval | 3.7×（74.7 TPS，54.3%）；5.3×（72.3 TPS，54.6%） | vanilla Dream | gen 256 / 512 |
| Dream GSM8K-CoT | 13.3×（94.1 TPS，74.7%） | vanilla Dream | gen 512 |
| Dream MBPP | 10.6×（92.4 TPS，55.8%） | vanilla Dream | gen 512 |
| Dream MATH | 11.2×（96.0 TPS，39.4%） | vanilla Dream | gen 512 |
| openPangu GSM8K / HumanEval / MATH | 1.6×（18.3 TPS，75.82%）/ 1.4×（14.6 TPS，48.17%）/ 1.4×（13.1 TPS，41.46%） | vanilla | — |
| 品質 | 與 baseline 相當或略優 | vanilla / 其他加速方法 | — |

## 限制 / 備註

- 68.2× 出現在 LLaDA-1.5 MBPP（原始 LLaDA-1.5 實作極慢），跨模型比較時 Dream 僅 3.7–13.3×、openPangu 1.4–1.6×。
- GPU 型號未查到；TPS 數字跨論文比較時需注意硬體差異。
- 加速由三個元件相乘，論文有 ablation，但各元件單獨貢獻的數字未從可用來源取得。

## 與其他論文的關係

- 與 **DPad**（suffix dropout：sliding window + 距離衰減）幾乎同一觀察與做法，Streaming-dLLM 額外保留尾端位置並加上時間維度的動態解碼與 early exit，因此宣稱的最高加速（68.2×）略高於 DPad（61.4×）。
- 動態解碼與 early exit 承襲 **Fast-dLLM** 的信心閾值平行解碼，並與 **R²-dLLM**（定稿已穩定 token）、Local Determinism Propagation 等步數削減工作思路相近。
- 「suffix 資訊集中在近端」也支持 **Elastic-Cache** 的視窗外 MASK 快取與 **Sparse-dLLM** 的 suffix 淘汰。
- 可與 **dLLM-Cache / Fast-dLLM** 的 prefix cache 疊加，形成 prefix cache + suffix pruning + 動態步數的完整 training-free 組合。
