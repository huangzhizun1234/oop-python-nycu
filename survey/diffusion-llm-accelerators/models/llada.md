# Large Language Diffusion Models (LLaDA)

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2502.09992`（v3 2025-10；社群筆記標示 NeurIPS 2025，待確認） |
| 作者 / 單位 | Shen Nie, Fengqi Zhu et al.（中國人民大學 高瓴人工智能學院 GSAI、螞蟻集團 Ant Group） |
| 日期 | 2025-02 |
| 類別 | 模型 |
| 連結 | [arXiv](https://arxiv.org/abs/2502.09992) · [GitHub](https://github.com/ML-GSAI/LLaDA) · [LLaDA 1.5 arXiv](https://arxiv.org/abs/2505.19223) |

## 一句話總結

第一個從零開始、以 masked diffusion 目標訓練到 8B 規模的擴散語言模型，證明「LLM 能力不必依賴自迴歸」：在 2.3T tokens 預訓練 + SFT 後，多數 benchmark 與 LLaMA3 8B 相當；但推論仍是「步數 ≈ 生成長度、無 KV cache」的原始形態，速度反而慢於 AR。

## 要解決的問題

- 在 LLaDA 之前，主流認為 LLM 的 scaling、in-context learning、instruction following 都是自迴歸（next-token prediction）的產物；擴散語言模型只在 <1B 規模驗證過，且與 AR 有明顯差距。
- 論文要回答：masked diffusion（前向隨機遮罩、反向預測遮罩 token 的 Transformer）若用相同資料與算力 scale 到 8B，能否達到 AR 的水準，並提供 AR 不具備的雙向建模能力（例如破解 reversal curse）。
- 推論效率並非本文重點，但 LLaDA 成為後續幾乎所有 dLLM 加速論文（Fast-dLLM、dKV-Cache、dLLM-Cache、量化、硬體加速器）的共同 baseline，因此它的推論形態決定了整個領域的瓶頸。

## 核心方法與特色

- **Masked diffusion 作為理論完整的 likelihood 下界**：前向過程以隨機比例 t ~ U[0,1] 遮罩 token（不同於 BERT 固定 15%），模型（無因果 mask 的 Transformer）預測所有被遮罩位置；損失以 1/t 加權的交叉熵，是負對數似然的上界（ELBO）。這使它與 AR 一樣能做 likelihood-based 評估與 scaling law，而非只是 BERT 的生成化。
- **訓練流程與 AR 幾乎相同**：8B 模型在 2.3T tokens 上預訓練，耗費 0.13M H800 GPU-hours，再以 4.5M 對話對做 SFT（SFT 時 prompt 不加噪，只對 response 遮罩）。官方 GUIDELINES 指出只需改「幾行程式」即可從 AR 訓練碼改成 LLaDA。
- **推論：迭代去噪 + remasking 策略**：從全 [MASK] 的固定長度回應開始，每步預測所有遮罩 token，再依「低信心 remasking」把信心最低者重新遮罩；官方建議步數等於回應長度才有最佳品質。另提供 semi-autoregressive remasking（把回應切成 block、由左到右逐 block 去噪），這是後來 block diffusion / KV cache 加速方法的直接前身。
- **Classifier-free guidance（無監督版）**：以無 prompt 條件的預測做 guidance，提升下游任務表現。
- **無 KV cache、速度慢於 AR**：README 明確承認 LLaDA 目前無法使用 KV cache、且比 AR 慢；第三方量測（Fast-dLLM 論文）在 A100 / RTX 4090 上 LLaDA-8B 原始取樣僅約 6.7–6.95 tok/s。
- **LLaDA 1.5（VRPO）**：對 dLLM 做偏好對齊時，ELBO 估計的高變異數會汙染 DPO 梯度；VRPO 分析偏差/變異上界並提出最佳 Monte Carlo 預算分配與 antithetic sampling，以 350K 偏好對訓練，數學/程式/對齊指標全面提升。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| LLaDA-8B-Base | 8B（dense） | 2.3T tokens 從零預訓練 |
| LLaDA-8B-Instruct | 8B | +4.5M SFT pairs |
| LLaDA-1.5 | 8B | LLaDA-8B-Instruct + VRPO（350K 偏好對） |
| 自建 ARM baseline | ≤1B / 7B 級 | 用相同資料訓練的 AR 對照組（scaling 實驗） |
| iLLaDA-8B-Base/Instruct | 8B | 2026-06 釋出，改善 benchmark 與生成效率（本文未涵蓋） |

## PPA / 效能數據

> 軟體論文：speedup、throughput (tok/s)、latency、記憶體、benchmark 準確度變化。

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| 預訓練 tokens / 算力 | 2.3T tokens；0.13M H800 GPU-hours | LLaMA3 8B 為 15T tokens | 8B，從零訓練；1.2T 時曾 crash，降 LR 恢復 |
| MMLU（Base, 5-shot） | 65.9 | LLaMA3 8B 65.4；LLaMA2 7B 45.9 | lm-eval，無 CFG |
| GSM8K（Base） | 70.7（論文）/ 70.3（repo 重跑） | LLaMA3 8B 53.1（論文表）；Dream 7B 77.2 | gen_length=1024, steps=1024 |
| MATH（Base） | 27.3（論文）/ 31.4（repo） | LLaMA3 8B 15.1 | 同上 |
| HumanEval（Base） | 33.5（論文）/ 35.4（repo） | LLaMA3 8B 34.2 | 同上 |
| MBPP（Base） | 38.2（論文）/ 40.0（repo） | LLaMA3 8B 47.4（依記憶，待確認） | 同上 |
| BBH / ARC-C / HellaSwag（Base） | 49.7 / 45.9 / 70.5 | — | repo EVAL.md |
| CMMLU / C-Eval（Base） | 69.9 / 70.5 | LLaMA3 8B 約 50 級（中文任務明顯領先） | lm-eval |
| MMLU / GSM8K / MATH（Instruct） | 65.5 / 78.6 / 42.2 | Qwen2.5-3B-Inst 69.1 / 86.3 / 67.0 | 論文表；repo 純擴散取樣重跑 GSM8K 為 69.4 |
| HumanEval / MBPP / GPQA（Instruct） | 49.4 / 41.0 / 33.3 | LLaMA3 8B Instruct 略高 | gen_length 依任務 3–512，block_length=gen_length |
| 原始推論速度 | ≈6.7–6.95 tok/s | LLaMA/Qwen 同尺寸 AR 約 10× 快 | LLaDA-8B, A100 / RTX 4090, 256 tokens, bs=1（Fast-dLLM 論文量測） |
| LLaDA 1.5 提升 | GSM8K +4.7（→83.3）、HumanEval +3.0（→52.4）、MBPP +1.8（→42.8）、IFEval +4.0（→58.2）、Arena-Hard +4.3 | vs LLaDA-8B-Instruct | VRPO，350K 偏好對 |
| Reversal curse | 反向詩句補全勝過 GPT-4o | GPT-4o | 論文案例研究 |

## 限制 / 備註

- 生成長度需事先固定（固定 context 取樣），無法自然變長；步數 = 長度才達最佳品質，若減少步數品質下降明顯。
- 不能直接使用 KV cache（每步所有位置都會變），因此在未加速的情況下速度遠低於 AR；後續 Fast-dLLM（block KV cache + 信心閾值平行解碼）在 LLaDA-8B 上達 8.1×（GSM8K −0.8 pt）到 27.6× 加速。
- 論文與 repo 的 benchmark 數字因評測工具、steps、CFG 設定不同而有 1–9 分差距（例如 Instruct GSM8K 78.6 vs 69.4）；引用時需固定評測協定。
- 訓練 token 效率仍低於 AR：Dream 7B 以 AR 權重初始化只用 580B tokens 即在多數指標超越 LLaDA。
- LLaDA 1.5 為同團隊後續工作，只改 post-training（VRPO），推論形態與速度與 LLaDA 相同。

## 與其他論文的關係

- 建立在 SMDM（Scaling up Masked Diffusion Models on Text, 2410.18514）、MDLM、RADD 等 masked diffusion 理論之上，並把它推到 8B。
- 是 Fast-dLLM、dKV-Cache、dLLM-Cache、Prophet、DLLMQuant 等幾乎所有 dLLM 推論加速/量化/硬體論文的預設實驗對象；其 semi-autoregressive remasking 是 BD3-LM / LLaDA2.0 / SDAR 這類 block diffusion 的過渡形式。
- 同團隊延伸：LLaDA 1.5（VRPO 對齊）、LLaDA-V（視覺語言）、LLaDA-MoE（與螞蟻合作、從零訓練的 MoE 版）、LLaDA2.0（螞蟻主導、AR→dLLM 轉換到 100B）。
- 與 Dream 7B 形成「從零訓練 vs. AR 初始化」的對照組；Efficient-DLM、Fast-dLLM v2 進一步論證 AR 轉換在速度與資料效率上的優勢。
