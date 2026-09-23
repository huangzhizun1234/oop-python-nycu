# Beyond Scattered Acceptance: Fast and Coherent Inference for DLMs via Longest Stable Prefixes (LSP)

| 欄位 | 內容 |
|---|---|
| arXiv / 出處 | `2603.05454`（ML Anthology 索引為 ICLR 2026，待確認） |
| 作者 / 單位 | Pengxiang Li, Joey Tsai, Hongwei Xue, Kunyu Shi, Shilin Yan（單位未查到） |
| 日期 | 2026-03 |
| 類別 | 平行與投機解碼（commit 排程 / KV cache 友善） |
| 連結 | [arXiv](https://arxiv.org/abs/2603.05454) · GitHub：未查到 |

## 一句話總結

指出「散落式接受」（在序列各處零星 commit 高信心 token）會打碎 KV cache、破壞記憶體局部性並反覆修補不穩定邊界；改為每步找出「最長的、左對齊的穩定前綴」並對齊到語言 / 結構分隔符後一次整塊 commit，training-free 地把 LLaDA-8B / Dream-7B 加速最高 3.4x 且品質持平或略升。

## 要解決的問題

- 主流 dLLM 平行解碼（confidence threshold）在任何位置只要信心夠就 commit，結果 committed token 散布在序列中間，中間夾著 mask：
  - KV cache 無法連續 append，只能整段重算或做碎片化更新；
  - 已 commit 的 token 兩側仍是 mask，之後上下文變化時常需要「修補」（重新評估或撤銷），造成重複計算。
- 純左到右（semi-AR）又放棄了 dLLM 的平行度。

## 核心方法與特色

- **Monolithic prefix absorption**：每步只 commit 一個從左邊界開始、連續的 token 區段（prefix），區段內所有 token 都被判定為「穩定」；區段以外即使有高信心 token 也暫不 commit。這讓已定案區域永遠是連續的前綴。
- **單次 forward 的穩定性評估**：用當步的一次 forward 對每個位置計算穩定性（信心 / 與前一步預測是否一致等訊號），沿左邊界往右掃，找出最長的連續穩定區段。
- **邊界對齊到自然分隔符**：把區段右邊界往回 snap 到最近的語言或結構分隔（標點、空白、程式碼的括號 / 換行等），避免在詞或語法單位中間截斷，減少下一步修補。
- **KV cache 變成連續 append**：因為 commit 的永遠是前綴，dLLM 的 prefix cache 可以像 AR 一樣直接追加，不需碎片化更新或整段 refresh，這是速度來源之一；另一來源是省掉對不穩定邊界的重複評估。
- **代價**：平行度受「最長穩定前綴」限制，若序列中段有一個不穩定 token，其右邊的高信心 token 都要等；對需要「先寫結尾再填中間」的任務不利。

## 使用的模型 (Model)

| 模型 | 參數量 | 備註 |
|---|---|---|
| LLaDA-8B | 8B | 數學推理、程式生成、CJK 多語、創意寫作 |
| Dream-7B | 7B | 同上 |

## PPA / 效能數據

| 指標 | 數值 | 對比基準 | 條件 / 設定 |
|---|---|---|---|
| 加速 | up to 3.4x | 標準解碼（scattered acceptance） | LLaDA-8B / Dream-7B，數學、程式、CJK、創意寫作 |
| 品質 | 持平或略升 | 標準解碼 | 逐項數字未查到 |
| GPU / tok/s | 未查到 | — | — |

## 限制 / 備註

- 具體 benchmark 數字與 GPU 未查到。
- 方法本質上把 dLLM 推向「自適應長度的 AR」，與 semi-AR block 解碼的差別在 block 長度由穩定性動態決定且對齊分隔符。
- 第一作者與 Prophet 相同，兩篇可合讀。

## 與其他論文的關係

- 與 **Prophet**（同作者）互補：Prophet 決定何時 all-in，LSP 決定每步 commit 哪一段。
- 對 **Fast-dLLM** 的 scattered threshold commit 提出結構性批評，並解決其 KV cache 碎片問題。
- 與 **Swordsman**（entropy 邊界切 block）同樣關注「邊界要對齊語言單位」，一個用 entropy、一個用分隔符 + 穩定性。
- 與 **CDLM / SDAR** 這類 block-causal 模型殊途同歸：都追求連續 prefix cache，但 LSP 不需訓練。
