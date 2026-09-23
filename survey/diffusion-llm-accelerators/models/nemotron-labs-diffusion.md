# Nemotron-Labs-Diffusion: A Tri-Mode Language Model Unifying Autoregressive, Diffusion, and Self-Speculation Decoding

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2607.05722`（模型 2026-05 於 HF 釋出，論文 2026-07） |
| 作者 / 單位 | Yonggan Fu, Lexington Whalen, Abhinav Garg, Chengyue Wu, Maksim Khadkevich, Nicolai Oswald, Enze Xie, … , Song Han, Jan Kautz, Pavlo Molchanov（NVIDIA） |
| 日期 | 2026-07 |
| 類別 | 模型 |
| 連結 | [arXiv](https://arxiv.org/abs/2607.05722) · [HF collection](https://huggingface.co/collections/nvidia/nemotron-labs-diffusion) · [NVIDIA Research](https://research.nvidia.com/publication/2026-05_nemotron-labs-diffusion-tri-mode-language-model-unifying-autoregressive) |

## 一句話總結

以聯合 AR + diffusion 目標訓練的一組權重，推論時只需切換注意力 mask 即可在三種模式間切換：純 AR、32-token block 擴散平行解碼、以及「擴散起草 + AR 驗證」的 self-speculation；8B 版每次 forward 解碼的 token 數是 Qwen3-8B 的 6×，在 GB200 + SGLang 上吞吐 4×（約 850 tok/s，客製 CUDA kernel 達 1,015 tok/s），且 self-speculation 模式在溫度 0 下與 AR 無損等價。

## 要解決的問題

- 部署場景差異大：低併發時擴散平行解碼最省延遲，高併發時 AR 的 batch 效率更好，而純 dLLM 在需要精確品質時無法回退；單一模型難以在所有 concurrency 下維持最高吞吐。
- 傳統投機解碼需要額外 draft 模型或 MTP head，接受率有限；dLLM 的多 token 草稿若能由同一模型的 AR 路徑驗證，就能得到無損且高接受率的加速。
- 論文也想量化：在理想取樣器下，擴散模式的「每 forward token 數」上限究竟比 self-speculation 高多少（speed-of-light 分析）。

## 核心方法與特色

- **三模式共享權重**：模型以聯合 AR-diffusion 目標訓練（先 1T tokens 純 AR 續訓，再 300B tokens 聯合目標；256×H100），同一權重依注意力 mask 決定行為：因果 mask → AR；block 內雙向 → diffusion（每次以 32-token block 迭代去噪，信心閾值決定每步 commit 哪些 token）；兩者串接 → self-speculation。
- **AR 與 diffusion 目標互補**：論文發現 diffusion 目標提升「前瞻規劃」能力，AR 目標提供左到右的語言先驗；聯合訓練後三種模式的準確率都高於單一目標訓練。
- **Self-speculation（擴散起草、AR 驗證）**：擴散路徑一次產生整個 block 的草稿，AR 路徑以因果 mask 在同一 forward 中驗證，接受最長匹配前綴；線性 / 二次（quadratic）兩種變體 TPF 分別 5.99× / 6.38×，平均接受長度 8.7 tokens，在接受率與實機效率上均優於 MTP / EAGLE 類方法，且溫度 0 時輸出與 AR 完全相同（無損）。
- **Speed-of-light 分析**：在最佳取樣器假設下，純擴散模式的 TPF 上限比 self-speculation 再高 76.5%，指出目前擴散取樣器（信心閾值）仍遠未達理論上限，是未來加速空間。
- **多尺寸與多模態**：3B / 8B / 14B，各有 Base、Instruct 與 Vision-Language 版；已整合 SGLang、mlx-vlm 等推論框架。
- **代價**：需 1.3T tokens 續訓；擴散模式在高併發時 batch 效率不如 AR；block 32 的擴散模式品質仍略低於 AR，需靠 self-speculation 取回無損。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| Nemotron-Labs-Diffusion-3B / -3B-Base | ≈3–4B | 第三方筆記記為 ~4B |
| Nemotron-Labs-Diffusion-8B / -Base / -Chat | 8B | 主要評測對象 |
| Nemotron-Labs-Diffusion-14B / -Base | 14B | — |
| VL 版本 | 3B / 8B / 14B | 視覺語言 |
| Qwen3-8B | AR 對照 | — |
| 開源 dLLM（LLaDA、Dream、SDAR、Efficient-DLM 等） | 對照 | 「準確率 +9%–22.4%」 |

## PPA / 效能數據

> 軟體論文：speedup、throughput (tok/s)、latency、記憶體、benchmark 準確度變化。

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| Tokens per forward（8B） | **6×**（線性 self-spec 5.99×、二次 6.38×；擴散模式約 2.6×） | Qwen3-8B = 1× | 論文摘要 / 第三方筆記 |
| 吞吐（8B, SPEED-Bench） | **4×** | Qwen3-8B | SGLang，GB200 |
| 絕對速度（8B, self-spec） | 850 tok/s；客製 CUDA kernel 1,015 tok/s | AR 253 tok/s（3.3×–4×） | GB200，單併發 |
| DGX Spark（FP8） | 112 tok/s | AR 41.8 tok/s（2.7×；另一筆記 3.14×） | 消費級桌上機 |
| RTX 6000 Pro（FP8） | 3.4× | AR | 第三方筆記 |
| 準確率（8B, quadratic self-spec） | 64.04% | Qwen3-8B 62.75% | 平均分（筆記） |
| 相對開源 dLLM | +9% ~ +22.4% | LLaDA / Dream 等 | 筆記 |
| 平均接受長度 | 8.7 tokens | MTP 類方法更低 | 線性 self-spec |
| Speed-of-light | 擴散模式理論 TPF 可再高 76.5% | self-speculation | 最佳取樣器假設 |
| 訓練量 | 1T AR + 300B 聯合 | — | 256×H100 |

## 限制 / 備註

- 高速數字（850–1,015 tok/s）為 GB200 單併發，與 Mercury（H100）、Seed（H20）不同硬體；4× 是 SPEED-Bench 上與 Qwen3-8B 的同框架比較，較有可比性。
- 純擴散模式品質仍低於 AR，無損加速只在 self-speculation 模式；因此其「dLLM」身分更接近「自帶 draft 的投機解碼模型」。
- 基礎模型來源（第三方筆記記為 Ministral3）與各尺寸精確參數量待論文確認。
- 同期 NVIDIA 相關：Nemotron-Labs-Diffusion-Image（2606.29814）、Nemotron-Labs-TwoTower（2606.26493）。

## 與其他論文的關係

- 是 NVIDIA 同團隊 Fast-dLLM v2 → Efficient-DLM → TiDAR 路線的集大成：block-wise diffusion 目標來自 Efficient-DLM，AR/diffusion 雙目標與「think in diffusion, talk in AR」來自 TiDAR（2511.08923）。
- Self-speculation 與 DFlash、DEER（draft with diffusion, verify with AR, 2512.15176）、Jacobi forcing 等「擴散做草稿」的投機解碼研究直接競爭。
- 與 LLaDA2.0（100B block diffusion + CAP）同為 block 32 的產品級 dLLM，但 Nemotron 以 AR 驗證保證無損，LLaDA2.0 以 CAP 提升 TPF。
- Speed-of-light 分析與 Set Diffusion「以更彈性的 token 集合提高每步平行度」的研究方向呼應。
