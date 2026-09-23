# DARTree: Speculative Diffusion Decoding with Autoregressive Draft Trees

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2608.13524` |
| 作者 / 單位 | Tianyi Li, Yaxin Luo, Xinyi Shang, Zhiqiang Shen（VILA Lab, MBZUAI） |
| 日期 | 2026-08 |
| 類別 | 平行與投機解碼（diffusion drafter + AR target 的 tree 驗證） |
| 連結 | [arXiv](https://arxiv.org/abs/2608.13524) · [GitHub](https://github.com/VILA-Lab/DARTree) |

## 一句話總結

在「block diffusion 當 drafter、AR 模型當 target」的投機解碼中，把既有的 AR correction head 從單條 draft chain 擴展成一棵 draft tree（逐深度批次修正、再做 best-first 剪枝），單輪驗證平均接受最高 12.97 個 token，對 AR 解碼達 9.73x 無損加速。

## 要解決的問題

- Diffusion drafter（DFlash、Domino 等）能一次平行產生整個 draft block，但其各位置分佈是 marginal，不是沿著某條具體 draft 路徑條件化的，因此 draft 內部常常自相矛盾，接受長度受限。
- Domino 用一個小的 AR correction head 沿單一鏈修正 draft，但只能給一條路徑；擴成樹時，如果對每個節點逐一跑 correction head（best-first 展開 + heap），會退化成序列運算，抵銷平行優勢。
- 這裡的 target 是 **AR 模型（Qwen3）**，屬「diffusion 加速 AR」；與本資料夾多數「加速 dLLM 本身」的論文方向相反，收錄作為對照。

## 核心方法與特色

- **Depth-wise batched AR correction**：從一個 block-parallel diffusion draft 出發，樹的每一層（深度）所有節點一次 batch 進 correction head 打分並展開，用「先建固定寬度的候選 supertree、再剪枝」取代「邊展開邊用 heap 選」，讓 correction head 的推論與序列 heap 操作解耦。
- **兩種變體**：DARTree (fixed) 把固定驗證預算（64 節點）分配到各深度後直接驗證；DARTree (pruned) 先建更寬的 supertree（寬度 12），再做延後的 top-B 剪枝選出最終驗證樹。
- **樹狀驗證**：target（Qwen3-4B / 8B）用 tree attention 一次 forward 驗證整棵樹，接受最長匹配路徑；輸出分佈與 target 自身解碼一致（lossless）。
- **Training-free**：直接沿用預訓練好的 Domino drafter（Qwen3-4B-Domino-b16，block 16）與其 correction head，不需再訓練；draft block size 16、每位置 64 個候選 token。
- **代價**：驗證樹越大，target 一次 forward 的 token 數越多；在算力飽和的大 batch serving 下，接受長度不會完全轉為 wall-clock 加速。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| Qwen3-4B（AR target） | 4B | — |
| Qwen3-8B（AR target） | 8B | — |
| Qwen3-4B-Domino-b16（block diffusion drafter + AR correction head） | 4B 級 | draft block 16 |

（本文未評測 LLaDA / Dream 等 dLLM 作為 target。）

## PPA / 效能數據

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| 平均接受長度 | up to 12.97 tokens / 驗證輪 | DFlash（+98.6%）、Domino（+27.9%） | 同設定 |
| 無損加速 | up to 9.73x | 本機量測的 AR 解碼 | 7 個數學 / 程式 / 對話 benchmark，4 種模型-溫度組合皆最佳 |
| 硬體 | NVIDIA RTX 6000 Ada（亦支援 Huawei Ascend NPU） | — | Python 3.12 / PyTorch 2.8 |
| 逐 benchmark 數字 | 未查到 | — | — |

## 限制 / 備註

- 這是 **AR 模型的加速方法**：diffusion 只作 drafter。與 dLLM 解碼論文的對照意義在於：它證明 block diffusion 的平行 draft 能力，搭配 AR 驗證後可達到比純 dLLM 解碼更高的無損加速（9.7x vs. dLLM self-speculative 的 2–3x）。
- 依賴已訓練好的 Domino drafter 與 correction head，換 target 家族需要對應的 drafter。
- 9.73x 是相對「本機量測 AR」而非優化推論引擎（如 vLLM）的 AR。

## 與其他論文的關係

- 直接建立在 **Domino**（AR correction chain）與 **DFlash**（block diffusion for flash SD, 2602.06036）之上，把 chain 擴成 tree。
- 與 **Bastion**（tree-structured block diffusion drafting, 2605.29727）、**DFlare**（2606.02091）同屬 2026 年「diffusion drafter 樹狀化」的競爭工作。
- 與 **SimSD**（dLLM target）和 **Spiffy / PSD**（dLLM self-speculation）方向相反，可視為「diffusion 加速 AR」對照組。
- 與 **Jacobi Forcing** 一樣都保留 AR 的最終品質，但 Jacobi Forcing 改造 target 本身，DARTree 不動 target。
