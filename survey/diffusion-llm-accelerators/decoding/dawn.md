# DAWN: Dependency-Aware Fast Inference for Diffusion LLMs

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2602.06953` |
| 作者 / 單位 | Lizhuo Luo, Zhuoran Shi, Jiajun Luo, Zhi Wang, Shen Ren, Wenya Wang, Tianwei Zhang（依記憶：Nanyang Technological University 為主，待確認） |
| 日期 | 2026-02 |
| 類別 | 平行與投機解碼（依賴感知的 unmask 排程） |
| 連結 | [arXiv](https://arxiv.org/abs/2602.06953) · [GitHub](https://github.com/lizhuo-luo/DAWN) |

## 一句話總結

從 attention map 抽出一張稀疏的 token 依賴圖，避免在同一步同時 unmask 彼此強耦合的位置（那正是平行解碼產生不連貫的來源），並用已 commit 的高信心 token 當錨點放寬其依賴位置的門檻，training-free 地在 LLaDA / Dream 上加速 1.80–8.06x。

## 要解決的問題

- 平行解碼把各位置視為條件獨立來採樣，但實際上強相關的位置（例如一個詞組的兩個 token）若同時獨立決定，容易產生「各自合理、合起來矛盾」的輸出。
- 信心門檻只看單一位置，看不到位置之間的耦合；LocalLeap 等只用局部性啟發式，仍可能同時解相鄰的耦合 token。

## 核心方法與特色

- **Dependency Graph Construction**：從模型當步的 attention map 取得位置間的注意力強度，先過濾掉 attention-sink（異常高入度）位置，再只保留高分連結，建成稀疏有向依賴圖；成本只是讀取既有 attention，不需額外 forward。
- **Anchor-Guided Decoding**：先用高門檻選出「一定安全」的位置平行 unmask；接著把已 commit 的高信心 token 當作錨點，對圖上依賴這些錨點的位置放寬信心要求（因為它們的條件已經穩定），擴大安全平行度。
- **Conflict-Based Scheduling**：對剩下較低信心的候選，以依賴圖定義「衝突」（有邊相連的兩位置不能同步 commit），貪婪地選出一個最大獨立集一次 unmask；這保證同一步 commit 的 token 彼此近似獨立，直接消除 joint/marginal 不一致。
- **三個門檻層級**（安全 / 錨點放寬 / 低門檻獨立集）讓平行度隨內容自適應；代價是每步需要處理 attention 矩陣（在長序列時是 O(L²) 的讀取）。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| LLaDA-8B-Instruct | 8B | — |
| LLaDA-1.5 | 8B | — |
| Dream-v0-Base-7B | 7B | — |
| Dream-v0-Instruct-7B | 7B | — |

## PPA / 效能數據

（官方 README 主結果表，NVIDIA H100 80GB；TPS = tokens/s；對比 Original 逐 token 解碼）

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| 加速範圍 | 1.80x–8.06x | Original | 四模型 × 四 benchmark |
| LLaDA-8B-Instruct GSM8K | 77.94%，44.72 TPS，4.33x | Original 77.94%，10.32 TPS | 準確度完全不變 |
| LLaDA-8B-Instruct MBPP | 30.80%，45.17 TPS，4.77x | Original 29.60%，9.46 TPS | — |
| LLaDA-8B-Instruct HumanEval | 40.24%，108.99 TPS，4.12x | Original 40.24%，26.46 TPS | LocalLeap 109.80 TPS 但 39.63% |
| LLaDA-1.5 MBPP | 37.60%，27.80 TPS，**8.06x** | Original 38.80%，3.45 TPS | 最高加速 |
| LLaDA-1.5 GSM8K | 80.82%，43.24 TPS，4.48x | Original 81.12%，9.65 TPS | — |
| Dream-v0-Instruct-7B GSM8K | 73.16%，32.99 TPS，4.52x | Original 76.35%，7.30 TPS | Dream 上準確度下降約 3 點 |
| Dream-v0-Instruct-7B MBPP | 55.80%，44.03 TPS，5.26x | Original 54.20%，8.37 TPS | — |
| Dream-v0-Base-7B GSM8K | 73.54%，25.88 TPS，1.80x | Original 76.42%，14.37 TPS | 最低加速 |
| 相對 LocalLeap | +0.05–5.17 TPS，準確度最多 +3.04% | LocalLeap | 多數 benchmark |

## 限制 / 備註

- Dream 系列在 GSM8K 上有 2–3 個百分點的準確度損失（Confidence baseline 亦然），LLaDA 系列幾乎無損。
- 依賴 attention map，與使用 sparse / linear attention 或不回傳 attention 的推論核心整合較麻煩。
- arXiv 頁面標記「Paper Coming Soon」時期較長，正式版本細節以最新版為準。

## 與其他論文的關係

- 直接對比並勝過 **LocalLeap**、**KLASS** 與 confidence-threshold（**Fast-dLLM**）三類 training-free 平行解碼。
- 與 **DAPD**（Dependency-Aware Parallel Decoding via Attention, 2603.12996）為同主題的後續工作。
- 與 **APD** 都在解決「平行採樣的 joint 不一致」：APD 用外部 AR 模型驗證，DAWN 用依賴圖避免同步解耦合位置。
- 可與 **STaRR / TACG** 的動態門檻疊加（它們決定門檻高低，DAWN 決定哪些位置可同步）。
