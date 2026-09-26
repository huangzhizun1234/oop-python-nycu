# DART / d-PLENA 推論流程深入解析：GEMM 切法、partial sum 累加、cycle 數

> 本檔補充 `npu-dllm-sampling.md`，聚焦「一個 denoising step 在 DART 上怎麼跑」。
> 資料來源：(1) arXiv 2601.20706 **v1**（題名 *Beyond GEMM-Centric NPUs*，架構名 **d-PLENA**）全文；(2) DART 的 GEMM 引擎完全繼承自同團隊的 **PLENA**（arXiv 2509.09505, ISCA），其硬體章節 LaTeX 原文；(3) 2601.20706 **v2**（改名 DART，2026-04）只取得搜尋摘要片段。
> 標「**[推導]**」的數字是依論文給的 dataflow 算出來的，不是論文明寫；標「**未查到**」的是 v2 全文才有。

---

## 1. 全貌：一個 block 的生成流程

DART 假設 Fast-dLLM 式的 **block-wise 解碼**：整段生成長度 L_tot 切成若干 block（block 長 L，例如 32），block 之間是 AR 順序，block 內做 T 步 diffusion。每個 block 的流程：

```
for block in L_tot / L blocks:
    ── warm step ──────────────────────────────────────────────
    Transformer Engine 跑「整段序列」（prompt + 已解碼 prefix + active block + masked suffix）
        → 產生所有位置的 K/V，寫進 KV cache（此時順便做 BAOS 線上校準，量化成 MX 4-bit）
        → 最後一層輸出 logits [B × L × V]，寫到 HBM（MX 格式）
    Sampling Engine 跑 Phase ①~④ → 更新 x（unmask 前 k 個最有信心的位置）
    ── refine steps（t = 2 … T）────────────────────────────────
    Transformer Engine 只跑 active block 的 L 個位置
        prefix-cache 模式：從 active block 起重算，suffix 的 KV 暫時重算但不存
        dual-cache  模式：只算 active block，K/V 原地覆寫 cache；suffix KV 沿用 warm step 的
        → logits [B × L × V] → HBM
    Sampling Engine Phase ①~④ → 更新 x
```

兩個引擎的分工：

| 引擎 | 負責 | 資料格式 |
|---|---|---|
| Transformer Engine（PLENA 的 Matrix Unit + Vector Unit） | QKV/O projection、FFN、雙向 FlashAttention | 權重、KV 存 Matrix SRAM（MX 格式，KV 用 BAOS MXINT4）；activation 存 Vector SRAM（BF16），進脈動陣列前動態量化成 MXINT8 |
| Vector-Scalar Sampling Engine | Stable-Max 信心分數、argmax、top-k、mask 更新 | logits 在 HBM 為 MX（v1 實驗用 MXFP8 E4M3），進 Vector SRAM 經 Dequantizer 轉 BF16；純量走 FP SRAM（BF16）與 Int SRAM（INT32） |

v1 profiling（A6000 + dInfer/vLLM 後端，LLaDA-8B-Instruct 與 LLaDA-MoE-7B-A1B，B=1~32、T=1~32、gen 64~1024、block 8~64）：sampling 佔端到端延遲最高 **71%**（MoE + dual-cache 時）。這就是為什麼 DART 把 sampling 做成獨立引擎。

---

## 2. GEMM 怎麼切：flattened systolic array

### 2.1 基本單元與參數

| 參數 | 意義 | PLENA 論文出現過的值 |
|---|---|---|
| `BLEN` | 一個 **sub-array** 是 BLEN × BLEN 個 PE 的方陣；也是輸出 tile 的邊長 | 4、8、16、32 |
| `MLEN` | 一個 flattened row 每個 cycle 從 Matrix SRAM / Vector SRAM 各讀進來的向量寬度 | 128、512、1024 |
| `MLEN / BLEN` | 一個 flattened row 裡 sub-array 的數量（沿 **K**、也就是 reduction 方向排成一列） | 4×1024 配置 → 256 個 4×4 sub-array |
| `HLEN` | FlashAttention 模式下每個 core 負責的 head 維度 | 128（Llama-3 head dim） |
| `VLEN` | Vector Unit 的 lane 數（softmax 的 max/sum/exp 用） | 16 ~ 2048 |

PLENA 對比用的配置是 **4 × 1024**（BLEN=4，MLEN=1024，共 4096 個乘法器），跟 baseline 的 64 × 64 方陣乘法器數相同。DART v2 說「full Matrix Unit 把這個結構複製成一個 grid，沿 row 與 column 方向以 BLEN 為步長 tiling」，也就是多個 flattened row 並排；grid 的數量 **未查到**。

### 2.2 一個 tile 的定義

一個 flattened row 一次做的 GEMM 是

```
(BLEN, MLEN) × (MLEN, BLEN) → (BLEN, BLEN)
   A tile        W tile        輸出 tile
```

- A 來自 **Vector SRAM**（左側進入），是 BLEN 個 token 的 activation 切出 MLEN 長的 K 片段。
- W 來自 **Matrix SRAM**（上方進入），是 MLEN × BLEN 的權重片段；Matrix SRAM 支援 **transpose-on-read**（每個邏輯 row 打散到多個 bank，行讀與列讀不衝突），所以 QK^T 不需要顯式轉置。
- **每個 cycle**，flattened row 各從兩個 SRAM 讀進一條 MLEN 寬的向量，buffer + 重排後切成 MLEN/BLEN 段、每段 BLEN 寬，第 s 段送給第 s 個 sub-array（從上與從左各一段）。
- 所以 **K 維被切成 MLEN/BLEN 份，每個 sub-array 只負責其中一份的 reduction**；每個 sub-array 用 output-stationary dataflow：BLEN×BLEN 個 PE 各自固定負責輸出 tile 的一個元素，operand 沿 K 流過去，partial sum 留在 PE 裡。

**[推導] 一個 tile 的 streaming cycle 數**：tile 有 BLEN·MLEN·BLEN 個 MAC，flattened row 每 cycle 做 (MLEN/BLEN)·BLEN² = MLEN·BLEN 個 MAC，所以一個 (BLEN,MLEN)×(MLEN,BLEN) tile 需要 **BLEN 個 cycle** 的資料流（同時也是「每側 BLEN·MLEN 個元素 ÷ 每 cycle MLEN 個」）。

### 2.3 K 比 MLEN 長時怎麼辦

LLaDA-8B 的 hidden 是 4096、FFN 中間層 12288，都比 MLEN 大。做法是 **沿 K 連續餵 K/MLEN 個 tile，partial sum 一直留在 PE 裡不寫回**（output-stationary 的意義就在這），全部 K 流完才做一次跨 sub-array 的加總。論文原話：「only one cross-array summation is required when computing GEMM along the large reduction dimension」，而且「the array is fully pipelined, eliminating idling bubbles between consecutive GEMM tiles」。

輸出矩陣 (M, N) 則切成 (M/BLEN) × (N/BLEN) 個輸出 tile，由 grid 裡的多個 flattened row 分攤（每個 row 在同一時間負責一個 BLEN×BLEN 輸出 tile）。

### 2.4 dLLM 的 M 是多少

這點跟 AR 不同，決定了利用率：

| 階段 | GEMM 的 M（token 數） | 備註 |
|---|---|---|
| warm step | B × L_tot（prompt + prefix + block + suffix 全部） | 等於 prefill，M 很大，方陣或 flattened 都能吃滿 |
| refine step（dual-cache） | B × L（只有 active block，例如 16×32 = 512） | 仍遠大於 AR decode 的 M=B，所以 dLLM 的 forward 是 compute-bound 偏多 |
| attention | 雙向：L_tot × L_tot 的 dense score 矩陣，**沒有 causal 三角可跳** | v2 特別指出這點 |

因此 flattened array 在 dLLM 的主要價值不是 AR 那種「M=1 時 99% PE 閒置」的救援，而是 (a) 沿 K 的長 reduction 用一次 M_SUM 就結束、(b) FlashAttention 時把 row 切成多個 per-head core 並行處理多個 head（見 §4）。

---

## 3. Partial sum 怎麼加

三個層次，由內而外：

1. **PE 內（沿 K 的時間累加）**：每個 PE 收 MX 格式的 a、w（元素與 block scale 分開串流進來），做 INT 乘加，累加器是 INT。K/MLEN 個 tile 連續流過時就是一直加在這個累加器上。
2. **sub-array 之間（跨 K 片段的空間累加）**：全部 K 流完後，MLEN/BLEN 個 sub-array 各持有一個 BLEN×BLEN 的 partial tile；一條 **result adder tree** 把它們加成一個 BLEN×BLEN 最終 tile。這一步由專用指令 **`M_SUM`** 觸發，整個 K reduction 只做一次。
3. **寫回**：加總結果從 INT 轉成 activation 精度（BF16），寫回 Vector SRAM。之後的 bias/norm/activation function 由 Vector Unit 做。

**[推導] 完整一個輸出 tile 的延遲**（單一 flattened row，K 為 reduction 長度）：

```
cycles ≈ (K / MLEN) × BLEN        ← streaming（主要項）
       + (2·BLEN − 2)             ← output-stationary 陣列的 fill / drain skew
       + log2(MLEN / BLEN)        ← adder tree 深度
       + 少量                     ← INT→BF16 轉換與寫回
```

代入 PLENA 的 4×1024 配置與 LLaDA-8B：

| GEMM | K | streaming cycles | 加上 skew(6) + tree(8) 的首個 tile 延遲 |
|---|---|---|---|
| Q/K/V/O projection | 4096 | 4096/1024 × 4 = **16** | ≈ 30 |
| FFN up / gate | 4096 | **16** | ≈ 30 |
| FFN down | 12288 | 12288/1024 × 4 = **48** | ≈ 62 |

因為 pipeline 化，穩態吞吐是「每 16（或 48）cycle 出一個 4×4 tile」。以 refine step 的 FFN up-proj 為例：M=512、N=12288 → 128 × 3072 = 393,216 個輸出 tile × 16 cycle ≈ 6.3M cycle / grid 中的 row 數。這也說明 DART 一定是多 row 的 grid，單一 row 不夠。

**MX scale 怎麼進累加**：PLENA 只寫「scales and elements are streamed separately to each sub-array」與「PE consumes MX inputs and performs accumulation in INT precision」，沒有寫 block scale（MX 每 32 個元素共用一個 8-bit 指數）是在 PE 內每 32 個 MAC 套一次、還是在 adder tree 前套。以「INT 累加、最後轉 BF16」的描述判斷，最合理的實作是每個 MX block 的 INT 部分和乘上 (s_a · s_w) 後再加進 FP/寬 INT 累加器，但這是 **[推導]**。

---

## 4. 雙向 FlashAttention 怎麼切

- Matrix Unit 切成多個 **flattened core**，每個 core 做 `(BLEN, HLEN) × (HLEN, BLEN)`，同時處理 `MLEN / HLEN` 個 head（4×1024 配置、HLEN=128 → 8 個 head 並行）。
- 每個 core 的流程照 FlashAttention-2 的 tile 化：`H_LOAD_M` 預取 K tile 到 Matrix SRAM（transpose-on-read 供 QK^T）→ `M_*` 算 S = Q·K^T 的 BLEN×BLEN tile → Vector Unit 以 VLEN lane 做 row-wise max / exp / sum（online softmax，精度用較高的 FP，例如 FP12）→ `M_*` 算 P·V → 累加 O 與 running max/sum，全程留在 on-chip。
- dLLM 的差別只有一個：**沒有 causal mask**，score 矩陣是 L_tot × L_tot 全滿，tile 數量是 (L_tot/BLEN)² 而非一半；DART v2 稱之為 "bidirectional FlashAttention support"。
- 記憶體側：HBM 存 MX；AXI master 256-bit 資料寬度（每 beat 32 B）、burst 128（每 burst 4 KB）、3 個 outstanding write / 4 個 outstanding read；HBM2e 模型對 AMD Alveo V80（2 stack、64 pseudo-channel、819 GB/s）校正，誤差 +5.3%（write）/ +3.3%（read）。

---

## 5. Sampling Engine：四個 phase 與 cycle 分解

### 5.1 為什麼要改寫 softmax（Stable-Max）

LLaDA 的信心分數是「該位置預測分佈的最大機率」。PyTorch 寫法是 `p = softmax(logits); conf = p[argmax]`，要把 V=126k 長的機率向量整條算出來再取一個元素。DART 改成：

```
m        = max(logits)              ← Reduction Unit（V_RED_MAX_IDX 同時給 max 與 argmax）
e        = exp(logits − m)          ← Elementwise Unit + FP Unit 的 e^x（原地覆寫 logits buffer）
sum_exp  = Σ e                      ← Reduction Unit（V_RED_SUM）
conf     = 1 / sum_exp              ← FP Unit 的 1/x（純量）
```

因為 max 位置的 e 恰為 1，所以 conf = 1/sum_exp，不需要機率向量；exp 結果原地覆寫 logits，Vector SRAM 只需一份 buffer。v2 把這四步各對應到一個專用單元，並證明取樣可從 FP64 降到 MXFP8 而品質不變，把取樣佔比從 ~70% 壓到 <10%。

### 5.2 四個 phase（每個 batch b、每個位置 l 各跑一次 Phase ①②；Phase ③④ 每個 batch 跑一次）

| Phase | 動作 | 指令 | 資料流 |
|---|---|---|---|
| ① HBM → Vector → Scalar | 把該位置的 V 個 logits 以 `V_chunk` 為單位預取到 Vector SRAM；每個 chunk 再以 VLEN 為單位做 Stable-Max 與 argmax，得到一個 FP 純量 conf 與一個 INT 純量 token id | `H_PREFETCH_V`、`V_RED_MAX_IDX`、`V_RED_SUM`、elementwise exp | 邊緣模式 V_chunk < V 要分 R = V/V_chunk 輪；效能模式 R=1 整條 logits（甚至 V·L 個）一次預載 |
| ② Scalar 寫回 | conf 寫進 FP SRAM[l]，token id 寫進 Int SRAM[l] | `S_ST_FP`、`S_ST_INT` | 兩個純量域實體分離，避免對齊衝突與控制路徑干擾 |
| ③ Scalar(FP) → Vector → Scalar(INT) | 把 FP SRAM 裡 L 個 conf 組回一條 L 長的向量；用 streaming insertion top-k 產生 boolean transfer mask（只考慮仍是 mask 的位置） | `S_MAP_V_FP`、`V_TOPK_MASK` | top-k 硬體：k 個並行比較器 + shift register，面積 O(k)，比較次數 O(L·k) |
| ④ Scalar(INT) | 依 mask 做 element-wise select：被選中的位置寫入新 token id，其餘維持原值（等價於兩次 `torch.where`） | `V_SELECT_INT` | 在 Int SRAM 上直接做，結果經 FIFO 送給 host |

Gumbel-max 溫度取樣在 v1 省略，列為 future work。

### 5.3 論文給的 cycle 數（v1 Table II / III）

工作負載：T=1、B=16、L=32、V=126k（LLaDA 詞彙量）、R=1、1 GHz、7nm、HBM2e。

| 配置 | 延遲 | 對 A6000 (2.51 ms) | Vector SRAM | Int SRAM | FP SRAM |
|---|---|---|---|---|---|
| d-PLENA VLEN=512 | 3.41 ms | 0.73× | 8 MB | 8 kB | 1 kB |
| d-PLENA VLEN=1024 | 1.79 ms | 1.40× | 8 MB | 8 kB | 2 kB |
| d-PLENA VLEN=2048 | 0.99 ms | **2.53×** | 8 MB | 8 kB | 4 kB |

VLEN=2048 的指令級分解（總 991,038 cycles）：

| 類別 | Cycles | 佔比 | 說明 |
|---|---|---|---|
| Vector | 477,696 | 48.2% | V_RED_MAX_IDX、V_RED_SUM、exp |
| Memory | 392,320 | 39.6% | H_PREFETCH_V，維持 67.7 GB/s |
| Scalar | 117,412 | 11.8% | S_ST_*、1/x、S_MAP_V_FP |
| Control | 3,609 | 0.4% | |

**[推導] 換算成每個位置**：B·L = 512 個位置 → 每位置約 1,936 cycles，其中 vector 約 933 cycles。每位置 126k/2048 ≈ 62 個 VLEN chunk，Stable-Max 要掃三遍（max、exp、sum）→ 約 186 次向量指令 → 每次約 5 cycles，符合「reduction 有 log2(VLEN)=11 級但 pipeline 化」的預期。Memory 側：512 個位置 × 126k × 1 B（MXFP8）≈ 64.5 MB，以 67.7 GB/s 傳約 0.95 ms，幾乎等於總延遲，說明這個 workload 在 DART 上已接近 HBM-bound；VLEN 再加大也只是壓縮 vector 那 48%。

參數掃描結論（Fig. 5，L=64）：延遲對 B、T、V 近似線性；V_chunk 從 128 增到約 4k 之後延遲與有效頻寬飽和，所以邊緣裝置不需要大 Vector SRAM（< 64 kB 即可跑，只是慢）；SRAM 需求由 B 與 V_chunk 決定，與 T、V、VLEN 幾乎無關。

### 5.4 後合成 PPA（v1 Table IV，7nm OpenROAD PDK，1 GHz）

| VLEN | Vector datapath 面積 | 總功耗 | 備註 |
|---|---|---|---|
| 512 | 0.731 mm² | 381.72 mW | Scalar 部分固定 661 µm² |
| 1024 | 1.464 mm² | 762.65 mW | 面積與功耗隨 VLEN 線性成長 |
| 2048 | 2.931 mm² | 1524.53 mW | |

（這是 sampling engine 的數字，不含 Transformer Engine。整顆 DART 的面積/功耗在 v2，**未查到**。）

---

## 6. KV cache 與 BAOS 在流程中的位置

- KV 以 MX 格式存 Matrix SRAM / HBM，block 與 scale 分開存放以對齊 2 的冪次邊界。
- **BAOS**：每個 block 的 warm step 本來就要重算整段 K/V，DART 把這一步當成免費的線上校準點：在 warm step 對每個 channel 算 scaling factor，把「平滑後」的 K 寫進 cache；refine step 的 attention 則把反向 scaling 融進 Q（乘在 query 上），而不是去反量化 cached K。因為 dLLM 的 channel 離群值會隨 denoising step 漂移，AR 式離線靜態校準會失準，BAOS 讓 MXINT4 KV 在 GSM8K 上持平 BF16。
- dual-cache 模式下 active block 的 K/V 每個 refine step **原地覆寫**（非 append-only），這是 DART 說「AR NPU 的 append-only KV 假設不成立」的具體原因。

---

## 7. 這份解析查不到、需要看 v2 全文的東西

- Transformer Engine grid 的實際 BLEN / MLEN / row 數，以及與 A6000 / H100 比較時的乘法器數量對齊方式。
- 整顆 DART（含 Transformer Engine）的面積、功耗、SRAM 總量。
- v2 的 model-phase 與 sampling-phase 各自的 cycle 數與 4.91× / 2.06× 的逐項來源。
- MX block scale 在 PE 累加鏈中的確切套用點。
