# dParallel: Learnable Parallel Decoding for dLLMs

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2509.26488`（ICLR 2026） |
| 作者 / 單位 | Zigeng Chen, Gongfan Fang, Xinyin Ma, Ruonan Yu, Xinchao Wang（xML Lab, National University of Singapore） |
| 日期 | 2025-09 |
| 類別 | 平行與投機解碼（訓練式步數壓縮） |
| 連結 | [arXiv](https://arxiv.org/abs/2509.26488) · [GitHub](https://github.com/czg1225/dParallel) · [HF 模型](https://huggingface.co/Zigeng/dParallel-LLaDA-8B-instruct) |

## 一句話總結

發現 dLLM 平行解碼的瓶頸是「mask token 的信心是循序收斂的」，提出 certainty-forcing distillation：讓模型沿著自己原本的採樣軌跡蒸餾，但強迫它對 mask token 更早、更同時地達到高信心，LLaDA-8B 在 GSM8K 的步數從 256 降到 30（8.5x）且不掉分。

## 要解決的問題

- 即使 dLLM 每步能預測全部位置，實際上只有少數位置的信心夠高，其餘位置要等鄰居被 unmask 後信心才逐個「傳遞」上來，形成事實上的序列依賴 → 步數無法真正壓低。
- 直接減少步數（few-step decoding）或 consistency distillation 會嚴重掉分（GSM8K 75.7% → 68–70%）。
- 需要一種便宜的訓練，改變模型的「信心收斂速度」而非改變它的輸出分佈。

## 核心方法與特色

- **診斷：sequential certainty convergence**：量測每步各位置的 entropy，顯示高信心位置像波前一樣由已知 token 往外擴散，是平行解碼的根本限制。
- **Certainty-forcing distillation**：先用原模型（teacher = 自己）以標準逐 token 解碼產生軌跡與最終輸出當作目標；訓練時對中間狀態做 semi-AR masking（LLaDA block 32、Dream block 256，mask 比例 50%），除了一般的蒸餾損失（跟隨原軌跡的預測），再加一個 certainty loss（溫度 0.5），把 mask 位置的預測分佈往 low-entropy 推。這讓模型學會「在同樣的上下文下就敢確定」，但不改變它會生成什麼。
- **只用 LoRA、資料由自己生成**：不需外部教師或人工標註；24GB GPU 即可訓練（官方腳本 8 GPU DeepSpeed），釋出 LLaDA 與 Dream 兩份蒸餾資料集。
- **推論：entropy-threshold 平行解碼**：推論時用 entropy 門檻（0.45 或 0.5）決定每步 unmask 哪些 token；因為蒸餾後大多數位置很快就低於門檻，步數自然大減。報告的結果**未使用 KV cache 或稀疏注意力**，疊加後可更快。
- **代價**：需要一次微調（模型變成專用版本）；在 MATH 這種高難度任務步數壓縮較少（256→46，5.7x）且有約 2 個百分點的損失。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| LLaDA-8B-Instruct | 8B | 蒸餾產出 dParallel-LLaDA-8B-instruct |
| Dream-7B-Instruct | 7B | 蒸餾產出 dParallel-Dream-7B-instruct（表格標題寫 Dream-8B，實為 7B） |

## PPA / 效能數據

（seq length 256、block 32、semi-AR remasking；無 cache；GPU 未載明）

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| GSM8K-CoT 步數 / 延遲 | 256 → 30 步，18.6s → 2.2s，**8.5x** | LLaDA-8B-Instruct | 準確度 75.7% → 76.1% |
| MBPP 步數 / 延遲 | 256 → 24 步，50.1s → 4.8s，**10.5x** | LLaDA-8B-Instruct | 42.4% → 40.8% |
| HumanEval | 256 → 33 步，8.2x | LLaDA-8B-Instruct | 38.4% → 40.2% |
| MATH (4-shot) | 256 → 46 步，5.7x | LLaDA-8B-Instruct | 33.5% → 31.5% |
| 對照：Few-step / Consistency Distillation 64 步 | 4.0x | LLaDA | GSM8K 68.6% / 69.9%（明顯掉分） |
| Dream GSM8K | 256 → 39 步，6.9x | Dream-7B-Instruct | 82.9% → 82.1% |
| Dream MBPP-Instruct | 256 → 29 步，8.8x | Dream-7B-Instruct | 58.8% → 56.2% |
| Dream HumanEval-Instruct | 256 → 37 步，6.9x | Dream-7B-Instruct | 52.4% → 54.3% |
| 第三方量測（Jacobi Forcing 論文） | HumanEval 88.5 TPS、GSM8K 128 TPS | AR Qwen2.5-7B 41 TPS | 準確度 54.3% / 82.9% |

## 限制 / 備註

- 蒸餾資料來自模型自身的逐 token 解碼，因此品質上限即原模型；模型若在某任務本來就弱，蒸餾後也不會變強。
- 官方數字未疊 KV cache；同團隊後續的 **DMax** 已把此路線推到 LLaDA-2.0-mini 並整合 dInfer 引擎。
- 訓練資料領域（數學 / 程式）以外的泛化能力論文著墨較少。

## 與其他論文的關係

- 與 **Learn2PD**（2509.25188）同名不同路：Learn2PD 外掛 filter、本體不動；dParallel 微調本體。
- 直接對比並勝過 **consistency distillation**（CDLM 路線的前身）與 few-step decoding；**CDLM** 之後結合 block-causal mask 補上 KV cache。
- 同團隊 **DMax**（2604.08302）是續作：從「加快信心收斂」進一步到「允許自我修正」，TPF 再翻倍。
- 推論端與 **Fast-dLLM** 的 threshold 平行解碼相同，可視為「訓練讓 threshold 解碼變得更有效」。
