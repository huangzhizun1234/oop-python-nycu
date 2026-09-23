# 子代理共同指示（撰寫完成後此檔會被刪除）

目標：為每篇指定論文各寫一個 Markdown 檔，放在指定子資料夾，語言用繁體中文（專有名詞、模型名、指標保留英文）。

## 資料來源限制
- arxiv.org、huggingface.co、alphaxiv、semanticscholar 都被 proxy 擋住，WebFetch 只能抓 github.com（以及少數其他網域）。
- 主要用 WebSearch：對每篇論文至少做 2~3 次不同查詢（標題全名、"標題 + speedup/throughput"、"標題 + abstract"、"標題 + github"），從搜尋摘要中擷取數字。
- 若論文有 GitHub repo，用 WebFetch 抓 README 取得結果表格。
- 也可以用 WebFetch 抓 github.com 上的 awesome list（例如 https://github.com/VILA-Lab/Awesome-DLMs 、 https://github.com/LiQiiiii/DLLM-Survey 、 https://github.com/ML-GSAI/LLaDA ）補充。
- 嚴禁捏造數字。查不到的數字寫「論文未提供 / 未查到」。每個數字後面盡量註明條件（模型、benchmark、GPU、batch size）。
- 你自己的既有知識可用來補充方法描述，但數字必須以查到的為準；若只有記憶中的數字、查不到佐證，請標註「(依記憶，待確認)」。

## 檔案格式
- 嚴格依照 `survey/diffusion-llm-accelerators/_TEMPLATE.md` 的章節結構。
- 檔名：`<short-name>.md`，小寫、用連字號，例如 `fast-dllm.md`、`npu-design-dllm-sampling.md`。
- 「核心方法與特色」要「詳細但精煉」：3~6 個 bullet，每個 bullet 1~3 句，說清楚機制（為什麼能加速、代價是什麼），不要只抄 abstract。
- 「PPA / 效能數據」用表格，硬體論文務必找製程、面積、功耗、頻率、能效、對 GPU 的 speedup/energy；軟體論文找 speedup、tok/s、accuracy 變化。
- 「使用的模型」列出論文評估的所有 dLLM（LLaDA-8B、Dream-7B、LLaDA-1.5、LLaDA-MoE、SDAR、Fast-dLLM v2 等）與參數量。
- 「與其他論文的關係」寫 2~4 條，指出它建立在哪些工作上、與哪些工作互補或競爭。

## 回報格式
完成後回覆一個表格，每列：`檔名 | 論文標題 | arXiv ID | 一行中文摘要（<=40字） | 最亮眼的一個數字`。
