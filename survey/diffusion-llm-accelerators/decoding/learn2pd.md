# Learning to Parallel: Accelerating Diffusion Large Language Models via Learnable Parallel Decoding (Learn2PD)

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2509.25188`（ICLR 2026；注意：題目中指定的 2509.14488 為誤植，經查證正確 ID 為 2509.25188，v1 副標題為 "via Adaptive Parallel Decoding"） |
| 作者 / 單位 | Wenrui Bao, Zhiben Chen, Dan Xu, Yuzhang Shang（依記憶：HKUST / University of Central Florida，待確認） |
| 日期 | 2025-09 |
| 類別 | 平行與投機解碼（可學習的 unmask 策略） |
| 連結 | [arXiv](https://arxiv.org/abs/2509.25188) · [GitHub](https://github.com/ims-kdks/Learning-to-Parallel-Decoding) · [Project](https://ims-kdks.github.io/learning-to-parallel/) |

## 一句話總結

訓練一個極輕量的 filter 模型，逐位置預測「這個 token 現在的預測是否已經等於最終答案」，用它取代固定 confidence 門檻來決定要不要 unmask，再加上 EoT 預測提早截斷 padding，在 LLaDA 上達到最高 22.58x（配 KV cache 57.51x）加速且不掉分。

## 要解決的問題

- 既有平行解碼（Fast-dLLM 等）用固定、與輸入無關的 confidence threshold 決定 unmask，對不同任務 / 不同難度的 token 都用同一條線，speed–quality 折衷不理想。
- 理想的「oracle」策略是：只要預測已與最終輸出相同就立刻 unmask（Extremely Greedy Parallel）；但推論時沒有答案可比對。
- 指定很長的 gen_length（如 1024）時，模型在 EoT 之後仍持續對 padding 位置做去噪，浪費大量算力。

## 核心方法與特色

- **Oracle 策略的定義與模擬**：離線先跑完整解碼取得最終序列，回頭標記每一步每個位置「當下預測是否 == 最終 token」，這就是 filter 的訓練標籤，讓 filter 學會近似 oracle。
- **輕量 filter 模型 f_θ**：以每個位置的信心等特徵（logit / 機率統計）為輸入，輸出「可 unmask」的二元決策；模型極小，post-training 只需分鐘級 GPU 時間（用 FLAN 資料產生訓練樣本，`training.ipynb` 即可訓練），基座 dLLM 完全不動。
- **輸入自適應**：因為 filter 是逐 token 判斷，簡單、格式化的 token 會很早被放行，困難 token 會被保留到上下文充分，速度依樣本自動調整。
- **End-of-Text Prediction (EoTP)**：一旦偵測到 [EoT] token，下一步就把其後所有位置丟掉，動態縮短輸入長度；對 gen_length=1024 的設定效益最大。
- **與 KV cache 疊加**：filter 只影響 unmask 決策，可直接疊在 Fast-dLLM 式 KV cache 上，得到 57.51x 的組合加速；代價是 filter 為近似 oracle，過度放行時會有小幅準確度損失。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| LLaDA-8B-Instruct | 8B | 主要實驗；gen_length 256 與 1024；GSM8K / MATH / HumanEval / MBPP |
| Dream | 7B | README 表示整合「coming soon」，論文主結果以 LLaDA 為主 |

## PPA / 效能數據

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| 最大加速（無 cache） | 22.58x | LLaDA vanilla 逐 token 解碼 | GSM8K，Learn2PD + EoTP，準確度不變 |
| 最大加速（含 KV cache） | 57.51x（16.37 tok/s） | LLaDA vanilla | GSM8K，僅極小準確度損失 |
| filter 訓練成本 | 分鐘級 GPU 時間 | — | post-training，基座不動 |
| 其他 benchmark（MATH / HumanEval / MBPP） | 加速數字以圖表呈現，未查到逐項數值 | — | 兩種生成長度 |
| GPU | 未查到 | — | — |

## 限制 / 備註

- 57.51x 對應的絕對 throughput 只有 16.37 tok/s，反映 vanilla LLaDA baseline 非常慢（gen_length 1024、無 cache），引用時需注意口徑。
- filter 依賴離線完整解碼取得標籤，換模型 / 換領域需要重新生成訓練資料（雖然便宜）。
- 主結果只在 LLaDA 上驗證。

## 與其他論文的關係

- 與 **dParallel**（2509.26488）同週發表、同名「Learnable Parallel Decoding」：dParallel 是微調 dLLM 本體讓信心更快收斂，Learn2PD 是不動本體、外掛一個 filter；兩者可疊加。
- 直接改良 **Fast-dLLM** 的 confidence-threshold 規則（同一個 GitHub 生態），是「學習取代手調門檻」的代表。
- **STaRR / TACG** 也在做動態門檻，但走 training-free 的統計訊號路線。
- **APD** 用外部 AR 模型驗證，Learn2PD 用內部 filter 判斷，兩者都在回答「每步該接受哪些 token」。
