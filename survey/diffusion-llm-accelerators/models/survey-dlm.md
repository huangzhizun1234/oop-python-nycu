# A Survey on Diffusion Language Models

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2508.10875`（v1 2025-08；v3 2026-06-04 更新） |
| 作者 / 單位 | Tianyi Li, Mingda Chen, Bowei Guo, Zhiqiang Shen（MBZUAI VILA Lab） |
| 日期 | 2025-08 |
| 類別 | 模型（總覽 / survey） |
| 連結 | [arXiv](https://arxiv.org/abs/2508.10875) · [GitHub Awesome-DLMs](https://github.com/VILA-Lab/Awesome-DLMs) |

## 一句話總結

第一份系統性的擴散語言模型（DLM）綜述：以「擴散空間（連續 / 離散）→ 訓練策略 → 推論優化 → 多模態 → 應用」為主軸建立分類法，指出離散（遮罩）DLM 已在 8B 規模追平同尺寸 AR、工業級 DLM 報告最高約 10× 推論加速，並將「平行度–品質權衡」「長序列的 O(N³) 計算」「基礎設施成熟度」列為核心挑戰；其配套 Awesome-DLMs 清單持續追蹤至 2026。

## 要解決的問題

- 2025 年 DLM 論文爆發（LLaDA、Dream、Mercury、Gemini Diffusion、Fast-dLLM、dKV-Cache …），缺乏統一的術語與分類；尤其推論加速方法（unmasking 策略、remasking、cache、guidance、蒸餾、量化）散落各處，難以比較。
- 本 survey 的目標是給出原理（連續 vs 離散擴散、ELBO 與遮罩損失的關係）、訓練（預訓練 / AR 初始化 / RL 對齊）、推論（各類加速）與應用的全景，並整理開放問題。

## 核心方法與特色

- **依擴散空間分類**：連續 DLM（在 embedding 空間預測噪聲 / 速度，如 Diffusion-LM、SSD-LM）與離散 DLM（在 token 空間做遮罩 / 轉移，如 D3PM、SEDD、MDLM、LLaDA）；並將 BD3-LM 這類 block 混合架構視為「平行生成 + KV cache」的折衷。survey 的判斷是離散遮罩擴散已成主流。
- **訓練策略分類**：從零預訓練（LLaDA）、AR 初始化 / 續訓（DiffuLLaMA、Dream）、互補遮罩、遮罩排程與重加權、蒸餾、RL 對齊（d1 / diffu-GRPO、VRPO、DCoLT）；引用 DCoLT 在 GSM8K +9.8% 作為 post-training 提升推理的例子。
- **推論優化分類（本 survey 對「效率章節」的分法）**：
  1. Unmasking / 平行解碼策略：信心 / 熵 / 隨機的 token 選擇、每步接受數的排程，並點出「Parallel Decoding Curse」（一步接受太多獨立取樣的 token 會產生不連貫組合）。
  2. Remasking：已解碼 token 的回溯與修正（ReMDM、Backtracking Enhanced Remasking）。
  3. Prefilling / Caching：標準 KV cache 在迭代細化下失效，需近似或混合方案（Fast-dLLM block cache、dKV-Cache、dLLM-Cache）。
  4. Guidance：CFG 與雙向上下文引導。
  5. Sampling / 步數蒸餾、context extension、稀疏計算、回應長度控制。
  6. Quantization（PTQ）與 pruning：列為進行中 / 未來方向。
- **多模態與應用**：MMaDA、LLaDA-V、Dream-VL 等統一多模態 DLM；程式（Mercury Coder、Dream-Coder）、生物序列等應用。
- **挑戰與方向**：平行度–效能權衡、長上下文（迭代 × 全序列注意力 ≈ O(N³)）、與 AR 相比的基礎設施差距（serving 引擎、量化工具）、8B 以上 scaling（撰寫時最大公開 DLM 為 8B；後續 LLaDA2.0 到 100B）、量化 / 剪枝、統一多模態推理。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| LLaDA-8B / LLaDA 1.5 | 8B | 主要開源代表；survey 引「與 LLaMA3-8B 相當」 |
| Dream 7B | 7B | AR 初始化代表 |
| Mercury（Coder） | 未揭露 | 商用，1,109 / 737 tok/s |
| Gemini Diffusion | 未揭露 | 商用，1,479 tok/s |
| Seed Diffusion | 未揭露 | 2,146 tok/s on H20（Awesome 清單收錄） |
| LLaDA-MoE、LLaDA2.0（100B） | 7B-A1B / 100B-A6B | 清單後續收錄 |
| BD3-LM、MDLM、SEDD、D3PM | ≤1B | 基礎方法 |
| MMaDA、LLaDA-V | 8B | 多模態 |

## PPA / 效能數據

> 軟體論文：speedup、throughput (tok/s)、latency、記憶體、benchmark 準確度變化。

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| Survey 總結的工業 DLM 加速 | 「up to ~10× inference acceleration at AR-comparable quality」 | 速度型 AR | 引 Mercury / Gemini Diffusion 宣稱 |
| Survey 總結的學術 DLM 加速 | 「2–4× speedup over AR while maintaining comparable quality」 | AR | 引 block / cache 類方法（筆記摘錄） |
| 8B 規模品質 | LLaDA-8B 與 LLaMA3-8B on par | AR | — |
| Post-training 推理提升 | DCoLT：GSM8K +9.8% | SFT 基線 | survey 引用 |
| Fast-dLLM（survey 引用） | LLaDA 8.1×–27.6×，GSM8K −0.8 pt | LLaDA 原始取樣 | 本 survey 「caching」類代表 |
| 自身無新實驗 | — | — | 綜述論文 |

## 限制 / 備註

- 為綜述，無原創實驗；數字皆轉引，且 v1 撰寫時（2025-08）尚未收錄 LLaDA2.0、SDAR、Fast-dLLM v2、Efficient-DLM、Nemotron、Set Diffusion 等後續工作（v3 與 Awesome-DLMs 清單有部分補充）。
- 對「效率」的分類以推論策略為主，硬體加速器（NPU / ASIC 設計）與系統 serving（dInfer、SGLang 整合）著墨少；本 survey 資料夾的 hardware / systems 子類需自行補充。
- 同期另有《Discrete Diffusion in Large Language and Multimodal Models: A Survey》（2506.13759）與 LiQiiiii/DLLM-Survey，分類法略有差異（後者把 Training / Inference / Quantization / Other 分開，並細分 unmasking、remasking、prefilling-caching、guidance、sampling、context extension、sparse computation、response length control）。

## 與其他論文的關係

- 為本資料夾所有模型類論文（LLaDA、Dream、Mercury、Gemini Diffusion、Seed Diffusion、BD3-LM）提供統一的術語與定位；其「推論優化」六類可直接對應本 survey 的 caching / decoding / quantization / systems 子資料夾。
- 與 2506.13759（離散擴散綜述）互補：本篇涵蓋連續 + 離散 + 多模態，後者聚焦離散與多模態。
- Awesome-DLMs 清單持續收錄後續模型（LLaDA2.0、DiffusionGemma、Efficient-DLM 等），可作為本 survey 更新來源。
- 其指出的「Parallel Decoding Curse」與「KV cache 失效」正是 Fast-dLLM、dKV-Cache、LLaDA2.0-CAP、Set Diffusion、Nemotron self-speculation 等後續工作要解決的核心問題。
