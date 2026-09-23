# SimSD: Simple Speculative Decoding in Diffusion Language Models

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2606.02544` |
| 作者 / 單位 | Junxia Cui, Haotian Ye, Runchu Tian 等共 12 位作者（單位未查到） |
| 日期 | 2026-06 |
| 類別 | 平行與投機解碼（draft–target 兩模型 speculative decoding） |
| 連結 | [arXiv](https://arxiv.org/abs/2606.02544) · [GitHub](https://github.com/airevo2/SimSD-release) |

## 一句話總結

用一個 plug-and-play 的 masking 策略讓 dLLM 在驗證時擁有「時間上有效」的 token 級上下文，使傳統 AR 式的 token-level speculative decoding（小 dLLM draft、大 dLLM 一次 forward 驗證）能直接用在 dLLM 上，SDAR-1.7B draft + SDAR-8B target 在雙 GPU 上從 9.6 tok/s 提到 63–74 tok/s 且輸出與 target greedy 逐 token 完全一致。

## 要解決的問題

- AR 的 speculative decoding 之所以能一次 forward 驗證多個 draft token，是因為 causal mask 讓每個位置的上下文在時間上固定。
- dLLM 依賴 mask token 與雙向注意力，同一位置的「有效上下文」隨 denoising 步驟改變（鄰居從 mask 變成 token），因此 target 模型對 draft token 的評分不對應任何一致的條件分佈，token 級驗證無法直接成立。
- 既有 dLLM 投機法（Spiffy、FreeDave、PSD）都是 self-speculative，沒有真正利用「小模型 draft」帶來的成本差。

## 核心方法與特色

- **Plug-and-play masking 策略**：在驗證 forward 中，把 draft 序列擺成多個 block，並用 multi-block causal attention 讓每個 draft 位置只看到「在它之前已確定的 token + 本 block 內的 draft」，重建出與 draft 模型生成時一致的條件上下文（temporally valid contexts）。這讓 target 的一次 forward 可對整段 draft 給出可比較的預測。
- **Greedy-match acceptance + variable-length truncate commit**：由左到右比較 target argmax 與 draft token，接受最長匹配前綴，在第一個不匹配處用 target 的預測取代並截斷，再從那裡繼續，保證輸出與 target 自己 greedy 解碼逐 token 相同（lossless w.r.t. argmax）。
- **Draft 端用 block diffusion 小模型**：SDAR-1.7B 以 block-diffusion 方式平行生成整個 draft block，比 AR draft 更快；target SDAR-8B 只做驗證。
- **系統層：雙 GPU 全流水重疊**：draft 與 target 分別放在兩張 GPU，含 speculative target K/V extend，讓 target 在 draft 產生下一 block 時同時驗證；throughput 以 CUDA event 計時。
- **代價**：需要一對詞表相容、同家族的大小 dLLM（SDAR、LLaDA2.0-mini/flash）；MoE target（LLaDA2）因 expert 迴圈無法 CUDA-graph 化，只能 eager 執行。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| SDAR-8B-Chat（target） | 8B | block diffusion |
| SDAR-1.7B-Chat（draft） | 1.7B | block diffusion |
| LLaDA2.0-mini（target 或 self-draft） | 16B（MoE） | 正確性測試 / self-draft |
| LLaDA2.0-flash（target，mini 當 draft） | 約 100B（MoE，bf16 191.6 GiB） | 分片於 4 GPU（naive 模型平行） |

## PPA / 效能數據

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| Throughput | 63–74 tok/s | vanilla SDAR-8B TP=2 ≈ 9.6 tok/s（約 6.5–7.7x） | 2x NVIDIA RTX PRO 6000 Blackwell 96GB，draft/target 各一張 |
| 輸出品質 | 與 target argmax 解碼 token-by-token 相同 | — | lossless |
| 評測資料集 | GSM8K / MBPP / TriviaQA / MMLU | — | N=200，latency 與 quality 各一輪 |
| 論文摘要層級的加速倍率 / acceptance length | 未查到 | — | — |

## 限制 / 備註

- 63–74 tok/s 的比較基準是 TP=2 的 vanilla（單一 8B 模型佔兩張 GPU），而 SimSD 用兩張 GPU 跑兩個模型，硬體用量相同但並非嚴格公平的單卡對比。
- vanilla_cg（CUDA graph）版本的數字未載於 README。
- 論文正文的 acceptance length 與跨模型（LLaDA2）結果未查到。

## 與其他論文的關係

- 與 **Spiffy / FreeDave / PSD**（self-speculative）形成兩條路線：SimSD 是真正的兩模型 SD，速度上限來自 draft 小模型的便宜。
- 與 **DFlash / Domino / DARTree / DFlare** 相反方向：那些用 diffusion 當 drafter 加速 AR target；SimSD 的 target 本身是 dLLM。
- 依賴 **SDAR**（block diffusion）與 **LLaDA2.0** 的 block-causal 結構才能建立時間一致的上下文，對全雙向的 LLaDA-8B / Dream 不直接適用。
- **Bastion**（2605.29727）、**BlockPilot** 等同期工作探討 draft 樹與預算分配，可疊加於 SimSD 的驗證框架。
