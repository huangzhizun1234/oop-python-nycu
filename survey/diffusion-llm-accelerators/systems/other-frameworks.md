# 主流推論框架對 dLLM 的官方支援彙整（vLLM / llama.cpp / Ollama / TensorRT-LLM / 其他）

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | 非單一論文；彙整 vLLM 官方部落格（2026-06-10）、llama.cpp PR #14644 / #14771、Ollama PR #12125、TensorRT-LLM 文件、NVlabs/Fast-dLLM README、inclusionAI/LLaDA2.0 README |
| 作者 / 單位 | vLLM 團隊 + Google DeepMind + NVIDIA；ggml-org（llama.cpp）；Ollama；NVIDIA（TensorRT-LLM） |
| 日期 | 2025-07（llama.cpp）～ 2026-06（vLLM） |
| 類別 | 系統與 serving |
| 連結 | [vLLM DiffusionGemma blog (GitHub source)](https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-06-10-diffusion-gemma.md) · [llama.cpp diffusion README](https://github.com/ggml-org/llama.cpp/blob/master/examples/diffusion/README.md) · [llama.cpp PR #14644](https://github.com/ggml-org/llama.cpp/pull/14644) · [llama.cpp PR #14771](https://github.com/ggml-org/llama.cpp/pull/14771) · [Ollama PR #12125](https://github.com/ollama/ollama/pull/12125) · [TensorRT-LLM supported models](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/models/supported-models.md) |

## 一句話總結

截至 2026-09，主流 AR 推論框架對 dLLM 的官方支援程度差異很大：SGLang（見 `sglang-dllm.md`）最完整、vLLM 於 2026-06 以 DiffusionGemma 為第一個原生 dLLM（H200 FP8 batch-1 達 1,288 tok/s）、llama.cpp 自 2025-07 起有 Dream / LLaDA / RND1 的 CLI 支援但無 KV cache、Ollama 仍停在 draft PR、TensorRT-LLM 則尚無文字 dLLM 支援。

## 要解決的問題

- 學術 dLLM 加速工作（Fast-dLLM、dInfer 等）多為獨立程式碼，缺乏 paged KV、continuous batching、量化、多卡並行與 OpenAI 相容 API；使用者要把 dLLM 放進既有的 AR serving 基礎設施（vLLM / SGLang / TensorRT-LLM / llama.cpp）。
- dLLM 的執行模式（雙向注意力、迭代精修、block 生成、每步自定義取樣）與 AR 的「一步一 token」路徑不相容，各框架必須找到一個既有抽象來對映。

## 核心方法與特色

- **vLLM：以 DiffusionGemma 為首個原生 dLLM（2026-06）**。DiffusionGemma 是 Google DeepMind 的 26B 離散擴散模型（Gemma4 backbone），一次平行去噪 256-token 的 block。vLLM 利用 Model Runner V2 的 `ModelState` 抽象讓模型自定義輸入準備與 per-request 狀態，並**重用 speculative decoding 資料路徑**：每一步把當前 canvas 視為一組 draft token，整體接受或整體拒絕。支援 per-sequence 的 causal / bidirectional 注意力切換、對稱 sliding window、entropy-bound denoising 與 self-conditioning 取樣，automatic prefix caching 不需修改即可運作；提供 FP8 與 NVFP4 checkpoint。在此之前（2026-04）vLLM 的 dLLM 支援仍屬實驗性（PR #57250）；LLaDA / Dream 等學術 dLLM 尚未原生支援，NVlabs/Fast-dLLM README 的 TODO 仍列「vLLM support」。
- **llama.cpp：`llama-diffusion-cli`（2025-07 起）**。PR #14644（2025-07-16 合併）以 Dream 7B 為參考實作，新增 `llm_arch_is_diffusion()`、`llama_vocab_get_mask()` API 與 diffusion CLI；PR #14771（2025-07-31 合併）加入 LLaDA-8B-Instruct 並統一兩種模型於同一範例。提供五種 unmask 演算法（confidence / entropy / margin / random / origin）與兩種排程（Dream 式 timestep-based `--diffusion-eps`、LLaDA 式 block-based `--diffusion-block-length`），後續加入 RND1；GGUF 轉換帶 diffusion 參數。設計上**不使用 KV cache**，每步對整段上下文（上限 2048）做一次 prefill 級 forward，且 logits 由 GPU 拷回 CPU 取樣——PR 初期 GPU→CPU logits 傳輸佔每步 87%（104 ms vs 10 ms 計算），優化後仍屬「教育用途」等級。
- **Ollama：PR #12125（draft）**。在 API 層新增 `DiffusionOptions`（steps 預設 128、epsilon、block length、algorithm、temperature、CFG scale、Gumbel noise 等），依架構（dream / llada）自動偵測 diffusion 模型，僅支援 `/api/generate`，拒絕 `/api/chat`；截至 2026-03 仍為 draft，未合併。底層依賴 llama.cpp 的 diffusion 實作。
- **TensorRT-LLM：無文字 dLLM 支援**。官方 supported-models 與 release notes 只列出視覺生成用的 diffusion 模型（FLUX、Wan、LTX，beta），未見 LLaDA / Dream / LLaDA2.0 / DiffusionGemma 等文字 dLLM；NVIDIA 對 dLLM 的投入集中在研究端（Fast-dLLM v1/v2、Fast-dVLM、Fast-dDrive）與 vLLM DiffusionGemma 的協作。SGLang 路線圖提到的「Nemotron Labs Diffusion + FastDiffuser」顯示 NVIDIA 的 dLLM 模型優先走 SGLang。
- **模型方官方引擎**：Ant Group 為 LLaDA2.0 建了基於 **dInfer + SGLang** 的客製引擎（KV-cache 重用、block-level 平行解碼），宣稱 2.1× 加速、LLaDA2.0-flash-CAP 最高 535 tok/s（見 `dinfer.md`、`sglang-dllm.md`）。Inception Labs 的 Mercury 與 Google 的 Gemini Diffusion 屬閉源 API 服務，公開數字未查證（Mercury 宣稱 H100 上 1,000+ tok/s，依記憶，待確認）。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| DiffusionGemma | 26B（Gemma4 backbone，256-token block） | vLLM 首個原生 dLLM；FP8 / NVFP4 |
| Dream-7B（Dream-v0） | 7B | llama.cpp PR #14644；Ollama draft |
| LLaDA-8B-Instruct | 8B | llama.cpp PR #14771；Ollama draft |
| RND1 | 未查到 | llama.cpp（entropy-based） |
| LLaDA-MoE-7B-A1B / Dream-Coder / CODA | 7B(1B active) / 7B / — | llama.cpp 討論中提及，未確認實作 |
| LLaDA2.0-mini / flash | 16B / 100B MoE | dInfer + SGLang 官方引擎；vLLM 未查到 |
| Fast-dLLM v2 | 7B（Qwen2.5 backbone） | vLLM 整合仍在 TODO |

## PPA / 效能數據

> 軟體論文：speedup、throughput (tok/s)、latency、記憶體、benchmark 準確度變化。

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| vLLM DiffusionGemma 吞吐 | 1,288 tok/s | 約 6× AR baseline、約 3× MTP baseline | H200，FP8，batch size 1 |
| vLLM DiffusionGemma 吞吐 | 1,008 tok/s | 約 5× AR、約 2.6× MTP | H100，FP8，batch size 1 |
| vLLM DiffusionGemma 精度 | 初步評測於 AIME 2025、GPQA Diamond、GSM8K（數字未查到） | — | — |
| llama.cpp Dream 7B（初版） | GPU 計算約 10 ms/step，logits D2H 104 ms/step（87%） | — | PR #14644 討論；優化後整段生成約 19 s |
| llama.cpp Dream 7B（CPU only） | 約 1,034 ms/step | — | PR #14644 |
| llama.cpp LLaDA-8B | 400–900 ms/step（依 offload 層數）；ARM 上同題約 11 分鐘 vs Llama-3.1-8B 27 秒（約 24× 慢） | Llama-3.1-8B | PR #14771 討論，無 KV cache |
| LLaDA2.0 官方引擎 | 2.1× 加速；LLaDA2.0-flash-CAP 最高 535 tok/s | AR 基準 | dInfer + SGLang（硬體未查到） |
| TensorRT-LLM | 無文字 dLLM 數據 | — | 不支援 |

## 限制 / 備註

- vLLM 的 dLLM 支援目前是「單一模型驅動」（DiffusionGemma），其 spec-decode 對映假設 block 內整體接受/拒絕；是否能推廣到 LLaDA / Dream 的 threshold 逐 token 接受尚未查到官方說明。
- llama.cpp 的實作無 KV cache、上下文 2048、每步 pp2048，效能與 GPU 框架相差數十倍，主要價值是讓 dLLM 能在 CPU / Apple Silicon 上跑 GGUF 量化模型。
- Ollama 支援尚未合併；TensorRT-LLM 文件無 dLLM。以上狀態皆為 2026-09 搜尋所見，變動快。
- 各框架數字的硬體與 batch 條件不同（H200 FP8 bs=1 vs ARM CPU），不可直接比較。

## 與其他論文的關係

- **SGLang dLLM 框架**（`sglang-dllm.md`）是目前支援最廣的 AR 框架（LLaDA2.x、SDAR、Fast-dLLM v2、DiffusionGemma、Nemotron diffusion 皆在路線圖），vLLM 選擇 speculative-decoding 路徑、SGLang 選擇 chunked-prefill 路徑，兩者都印證 **Sangam**（`sangam.md`）「重用 AR stack」的論點。
- **dInfer**（`dinfer.md`）是 Ant Group 的模型方引擎，並在 v0.2 與 SGLang 整合；**HERALD**（`herald.md`）建構在 SGLang 上。
- **Fast-dLLM / Fast-dLLM v2**（NVlabs）是多數學術系統（BlockServe、ODB-dLLM、dLLM-Serve）的基底或 baseline，但其自身尚未進入 vLLM / TensorRT-LLM。
- llama.cpp 的 confidence / entropy / margin 演算法對應學術上的 low-confidence remasking（LLaDA）、entropy-based（Dream）等解碼策略；缺乏 KV cache 使 **Fast-dLLM** 式 block cache 成為其最直接的改進方向。
