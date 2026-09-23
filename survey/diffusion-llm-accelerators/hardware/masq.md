# MASQ: Accelerating Masked Diffusion via Stage-Wise Multi-Precision Quantization

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2605.23226` |
| 作者 / 單位 | Seeyeon Kim et al.（共 4 位作者；單位未在搜尋結果確認） |
| 日期 | 2026-05 |
| 類別 | 硬體加速器 / 量化 |
| 連結 | [arXiv](https://arxiv.org/abs/2605.23226) · GitHub：未查到 |

> **重要備註：MASQ 針對的是「影像 masked diffusion」（Stable Diffusion v1 的 text-guided inpainting、SDEdit 筆刷編輯），不是 masked diffusion 語言模型 (LLaDA/Dream)。** 論文中的 "mask" 指影像上要重繪的空間區域，而非 dLLM 的 [MASK] token。收錄於此是因為其「依 mask 分配精度的多精度 MPU」設計思路對 dLLM 加速器有直接啟發（見最後兩節）。

## 一句話總結

MASQ 是軟硬體共同設計的 masked-diffusion 加速器：觀察到影像 inpainting/editing 每個去噪步都對整張特徵圖計算，但只有 mask 區域真正變化，因此依「空間位置（在 mask 內/邊界/外）+ 語意重要性 + 去噪階段」把各 token/區塊動態指派為 MXINT8 / MXINT4 / MXINT2 精度，並用 mask-aware 多精度矩陣單元 (MP-MPU, 32 個 BMPE) 執行，相較 A100 最高 16.06x speedup、4.18x 能效，相較 Jetson Orin NX 5.39x speedup、4.93x 能效。

## 要解決的問題

- 影像 masked diffusion（inpainting、局部編輯）每個 timestep 都對整張圖跑完整 UNet/Transformer，但實際只有 mask 內的區域需要生成；mask 外的區域幾乎不變卻仍以全精度計算，造成大量冗餘。
- 既有做法（跳過 mask 外計算、或全圖統一低位元量化）不是損失邊界一致性，就是無法在保證品質下降到極低位元；且 GPU 缺乏依 token 動態切換精度的硬體支援。

## 核心方法與特色

- **Stage-wise 多精度指派 (MXINT8/4/2)**：把特徵 token 依「是否在 mask 內、離 mask 邊界多遠、語意注意力重要性」分成不同層級，再依去噪階段（早期結構 vs 晚期細節）動態調整：mask 核心與早期步驟用 MXINT8，過渡區用 MXINT4，mask 外/晚期穩定區用 MXINT2。用 MX（microscaling）區塊共享指數格式讓低位元仍保有動態範圍。
- **Mask-aware Multi-Precision MPU (MP-MPU)**：核心運算引擎由 32 個 block-wise multi-precision PE (BMPE) 組成，每個 BMPE 可依區塊精度標籤以 INT8/INT4/INT2 模式工作（低位元時把乘法器拆成多路以提升吞吐）；合成配置為 32 個 MP-MPU。
- **Mask Manager + Quantizer + VPU**：Mask Manager 產生並追蹤每個區塊的精度標籤（隨 timestep 更新）；Quantizer 線上做 MX 量化；VPU 處理 softmax、normalization 等非矩陣算子，並針對這些算子做了最佳化避免成為 Amdahl 瓶頸。
- **Timestep-aware 排程**：依不同 timestep 的精度分佈重新排程區塊，讓同精度的區塊聚集、MP-MPU 的模式切換最少、利用率最高。
- **品質保證**：在 EditBench 與 LAION inpainting 集合、SDEdit 編輯任務上主張品質「preserved」（具體 FID/LPIPS 數字未查到）。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| Stable Diffusion v1（UNet） | ~0.86B UNet | text-guided inpainting；EditBench、LAION inpainting 集合 |
| SDEdit（基於 SD） | 同上 | stroke-based 影像編輯 |
| （無 dLLM） | — | **未評估任何 diffusion 語言模型** |

## PPA / 效能數據

> 硬體論文：製程、面積、功耗、頻率、能效 (TOPS/W)、對比 GPU 的 speedup / energy。

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| 頻率 / 電壓 | 800 MHz @ 0.8 V | — | 合成結果 |
| 面積 | 6.15 mm² | — | 32 個 MP-MPU（每個 32 BMPE）+ 2 MiB on-chip memory |
| 製程 | 論文未在公開摘要提供 / 未查到 | — | — |
| 功耗 | 未查到 | — | — |
| Speedup | 最高 16.06x | NVIDIA A100 | 影像 inpainting/editing 工作負載 |
| Speedup | 最高 5.39x | NVIDIA Jetson Orin NX | 同上 |
| 能效 | 4.18x | A100 | 同上 |
| 能效 | 4.93x | Orin NX | 同上 |
| 品質 | 「preserved」；FID/LPIPS 數值未查到 | FP16 baseline | EditBench / LAION / SDEdit |

## 限制 / 備註

- **與 dLLM 無直接關係**：所有評估都是影像 diffusion（SD v1 UNet）。標題的 "Masked Diffusion" 容易與 masked diffusion language model 混淆，收錄時務必標明。
- 加速主要來自「mask 外區域用 INT2」的位元縮減，若 mask 覆蓋整張圖（純生成而非編輯）效益會大幅下降。
- 製程、功耗未在公開摘要中；A100 的 16.06x 很可能包含 GPU 缺乏 INT2/INT4 tensor core 支援的因素。

## 對 dLLM 加速器設計的啟示

- dLLM 每步也只有一小部分 token 真正被更新（被 unmask 的 token、及其鄰近位置），其餘 token 的活化在相鄰步驟間變化很小——這與影像 inpainting「mask 內 vs 外」的結構同構。MASQ 的「依 mask 狀態 + 去噪階段指派 MXINT8/4/2」可直接映射到「已定 token / 剛 unmask 的 token / 仍為 [MASK] 的 token」三類，配合 block-diffusion 的 prefix 不變性更明顯。
- MP-MPU 的 block-wise 多精度 PE 與 **block-diffusion-edge-hw.md** 的 BRQ-KV / DAT-FFN（依飄移決定精度）、**DART** 的 MX 格式 KV 量化 (BAOS) 屬於同一設計空間：對 dLLM 而言，精度標籤應由「token 的信心/穩定度」而非空間 mask 決定。

## 與其他論文的關係

- 屬於影像 diffusion 加速器脈絡（SD-Acc、DiSC、MixDiT 的 MX 混合精度、SQ-DM 的量化 + 時間稀疏）；MixDiT 同樣用 MX 混合精度但沒有 mask-aware。
- 與 **activation-concentration.md** 同為影像 diffusion 硬體研究，但關注點不同：前者是精度層級的冗餘，後者是稀疏度層級的冗餘；兩者都提醒 dLLM 加速器設計者「不要只看 element-level 的統計」。
- 與 **DART (npu-dllm-sampling.md)** 在 MX 格式量化上可互相參考，但 DART 針對語言、MASQ 針對影像。
