# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Trọng Minh  **MSSV**: 02496  **Ngày**: 2026-10-07
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB (14.6 GB khả dụng, sm_75)`

> **Phiên bản**: `v2_2026-10-07` — **iteration 2, full eval** (`EVAL_LIMIT` rỗng ⇒ `n_target=50`, `n_regression=15`).
> Xem `docs/RUNS.md` để biết trạng thái từng bản chạy.
>
> Mọi con số dưới đây khớp với file trong `results/`. Grader kiểm tra chéo.
> Bảng số được **sinh tự động** bằng `python tools/report_tables.py` (nguồn: `results/*.json`),
> không chép tay — kiểm tra bằng `python tools/report_tables.py --check`.

> ### 🎯 Kết luận ngắn: phán quyết ĐẢO CHIỀU từ PASSED sang FAILED
>
> Ở iteration 1 (smoke, `n_target=8`), cổng hồi quy **PASSED** với `target Δ = +0.250`.
> Ở iteration 2 (full eval, `n_target=50`) — **cùng base model, cùng seed 42, cùng 30 step,
> cùng corpus, cùng prompt** — cổng **FAILED**: `target Δ = +0.205` nhưng
> **`regression Δ = −0.202`**, vượt xa ngưỡng chịu đựng `0.02`.
>
> **Tăng kích thước tập eval đã lật ngược kết luận.** Xem §5.3 và §8 để biết vì sao đây là
> kết quả đáng giá nhất của cả lab, không phải một bước lùi.

---

## 1. Setup

| | |
|---|---|
| Dataset | **250 ticket CSKH tiếng Việt → JSON triage 4 trường** (corpus mặc định của lab) |
| Train / val | **225 / 25** (seed 42) |
| `max_length` | **1024** — p95 đo được là **98** *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | **2.0 / 30** |
| Batch hiệu dụng | 1 × 16 = **16** (< trần 32 của deck §11.4) |
| Precision | **fp16** — T4 là Turing (sm_75), **không có bfloat16** |
| **Eval** | **`EVAL_LIMIT` rỗng ⇒ 50 target + 15 regression** (`smoke_mode: false`) |

**Template có giữ khối `think` không?** **Có** — `results/template_check.json`:
`verdict = "reasoning preserved — safe to train on traces"`, `body_present: true`.
Chuỗi render thật:

```
<|im_start|>user
2+2?<|im_end|>
<|im_start|>assistant
 thinking
buoc 1: kiem tra. buoc 2: tra loi.
</think>

4<|im_end|>
```

Nghĩa là reasoning traces **sẽ** tới được hàm loss nếu dataset có chúng. Dataset mặc định thì
không có (xem §5.4) — nhưng điều đó là do dữ liệu, không do template.

### Vì sao chọn `max_length=1024` khi p95 chỉ gợi ý 256 (rubric 1.3)

`results/token_stats.json` đo trên 250 mẫu train: `mean 93.1`, `p50 93`, `p95 98`, `p99 100`,
`max 101`, và `suggested_max_length = 256` (p95 làm tròn lên luỹ thừa 2).

Tôi **giữ 1024** của tier `T4` thay vì hạ xuống 256, vì ba lý do:

1. `max_length` là **trần dùng chung cho cả bốn run** (`correct`, `attn_only`, `wrong_lr`,
   `qlora`). Đổi nó giữa các run sẽ làm phép so sánh mất nghĩa — và cả lab này là về so sánh.
2. Với `per_device_batch=1` và `padding_free=False` (Turing không có FlashAttention-2, xem
   `labkit/device.py`), "padding" chỉ là phần đuôi của **một** chuỗi mỗi bước. Ở 1024 so với
   256, chi phí thêm nằm ở attention trên phần đuôi đó — đo được, nhưng nhỏ so với 9.32 GB
   trọng số đã nạp.
3. Đây là **số đo, không phải số đoán**: tôi biết p95 = 98, biết 256 là mức đủ, và chọn 1024
   một cách có ý thức để giữ tính so sánh được. Log NB1 in cảnh báo
   `⚠ p95 gợi ý max_length=256 nhưng tier đang đặt 1024` — tôi đã đọc nó và đây là câu trả lời.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | **0.4149** (39/94 token) |
| Câu trả lời nằm trong loss | **`true`** |
| Câu hỏi KHÔNG nằm trong loss | **`true`** |

Cả hai assert trong NB1 đều xanh. Đoạn **được tính loss** (`supervised_preview`):

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Đoạn **bị che** (`masked_preview`) — chứng minh câu hỏi không nằm trong loss:

```
<|im_start|>system
Phân loại ticket sau.<|im_end|>
<|im_start|>user
Alo shop, mình đặt balo laptop mã đơn VN411453. Cho tôi trả lại. Đã 3 ngày rồi. Cho tôi hỏi.<|im_end|>
<|im_start|>assistant
 thinking

```

Đọc kỹ hai khối trên: loss chỉ rơi vào **`</think>` + JSON nhãn + `<|im_end|>`**. Toàn bộ
system prompt, ticket của khách, và cả khối `think` rỗng đều nằm ngoài loss.

**Vì sao `supervised_fraction` chỉ 0.41 chứ không cao hơn.** Trên tập train đầy đủ 225 mẫu,
con số là **9014/20951 = 43.0%** (log NB3). Phần bị che lớn vì mỗi mẫu có một system prompt
lặp lại và một ticket dài, còn câu trả lời chỉ là 4 trường JSON ngắn. Nếu con số này ≥ 0.95
thì đó là dấu hiệu đang tính loss **cả trên prompt** — rubric 1.1 nói rõ như vậy.

**Vì sao điều này quan trọng hơn mọi biến thể LoRA.** Chế độ `everything` cho
`supervised 94/94 (100%)`: câu hỏi nằm trong loss, model sẽ học cách **viết lại ticket của
khách**. Đó chính là triệu chứng deck §22 mô tả, và nó chỉ lộ ra sau khi train xong vài giờ.
NB1 cho thấy cả hai mặt cạnh nhau nên không phải tin ai cả.

**Về `masked-think` / `response-only` trên corpus này.** Hai chế độ này **là no-op** ở đây:
cả 250 câu trả lời là JSON trần, và chat template của Qwen3.5 đóng khối `think` rỗng **bên
trong generation prompt** — nên không còn gì trong `[start, end)` để bỏ qua, và mask sinh ra
giống hệt `assistant-only`. `labkit.data.to_training_dataset` cảnh báo `no-op on this corpus`
đúng vì lý do này. Đây là lý do §5.4 có `valid_trace_rate = 0.0`.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

Bảng sinh từ `results/verdict.json`:

| run | target | regression | format | latency_ms | n |
|---|---|---|---|---|---|
| (a) base + naive prompt | 0.0 | 0.7911 | 0.0 | 3273.6 | 50 |
| (b) base + optimized prompt | 0.765 | 0.7911 | 1.0 | 1043.2 | 50 |
| (c) LoRA fine-tune | 0.97 | 0.5889 | 1.0 | 1373.8 | 50 |

**(b) có thật sự mạnh hơn (a) không?** **Có** — `verify.py` xác nhận
`[ ok ] baseline (b) beats (a)  (a)=0.000 -> (b)=0.765`.

Và đây là điểm đáng chú ý nhất ở §3: **(b) vừa chính xác hơn vừa RẺ HƠN (a) 3.14×**
(3273.6 ms → 1043.2 ms). Prompt ngây thơ `"Phân loại ticket sau."` khiến model trả lời bằng
văn xuôi tiếng Việt — `format = 0.000`, không parse được JSON, và sinh dài nên chậm. Prompt
tối ưu có schema + ví dụ nên model đi thẳng vào JSON ngắn. Nghĩa là **prompt engineering
không phải là "chi phí để so sánh" mà tự nó đã là một cải thiện lớn** — đúng luận điểm
deck §21: bản fine-tune phải vượt *một baseline đã được prompt tử tế*, không phải một
baseline bị dựng lên cho yếu.

**Bạn có sửa `OPTIMIZED_PROMPT` không?** **Không.** `optimized_prompt_sha = 719e74d3b6232053`
khớp bản gốc của lab; `verify.py` xác nhận `[ ok ] baseline (b) prompt unmodified`. Tôi không
làm yếu (b) — đó là cách chính để gian lận lab này và rubric gọi thẳng là lỗi liêm chính.
Đây cũng là điều kiện để phép so sánh ở §5 có nghĩa: (b) không bị đụng tới giữa hai iteration.

**Điểm mấu chốt cho §5:** cả (a) và (b) đều đạt **regression 0.7911** trên 15 câu hỏi kiến
thức phổ thông — tức **11.87/15 câu đúng**. Hai con số giống hệt nhau vì cả hai đều là base
model chưa fine-tune. Đó là mốc để đo mất mát ở §5.

---

## 4. Giải phẫu cấu hình sai (NB4)

Bảng sinh từ `results/runs.csv` và `results/autopsy.json`:

| run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32464896 | 0.0001 | 0.6255 | **0.97** | 389.9 | 8.78 |
| `attn_only` | attn-only | 283 | 32456704 | 0.0001 | **0.538** | **0.97** | 258.5 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32464896 | 1e-05 | 1.5702 | **0.0** | 396.9 | 8.78 |
| `qlora` | text-linear | 16 | 32464896 | 0.0001 | 0.7058 | 0.94 | 468.7 | **3.86** |

Cả bốn run dùng **cùng 30 optimizer step** (`verify.py`: `[ ok ] all runs share ONE step budget
['attn_only','correct','qlora','wrong_lr'] at 30 steps`), nên khác biệt duy nhất giữa mỗi
contrast và `correct` là **đúng một biến**.

**Xếp hạng theo `target`:** `correct` = `attn_only` (0.97) > `qlora` (0.94) > `wrong_lr` (0.0).
**Xếp hạng theo `final_loss`:** `attn_only` (0.538) > `correct` (0.6255) > `qlora` (0.7058) > `wrong_lr` (1.5702).

**Hai thứ tự này KHÁC NHAU ở vị trí đầu tiên**, và ở full eval điều đó còn rõ hơn ở
iteration 1 vì khoảng cách `qlora` vs `correct` giờ đo được trên 50 mẫu (0.94 vs 0.97 =
**6 trường sai trên 200**). Đây là kết quả đáng giá nhất tôi đo được trong lab, bàn ở 4.1.

### 4.1 — `attn_only` vs `correct`: rank có phải đòn bẩy không?

`attn_only` dùng **cùng ngân sách tham số** với `correct`: 32,456,704 vs 32,464,896, lệch
**0.025%** — thoải mái dưới ngưỡng 5% của rubric 2.1 (`verify.py`: `[ ok ] attn_only is a FAIR
contrast  32,456,704 vs 32,464,896 trainable params`). Để đạt ngân sách đó, rank phải nâng từ
**16 lên 283** (alpha 566) — tức gấp ~17.7×.

Trên tập target đầy đủ 50 mẫu, `attn_only` **HOÀ** với `correct`: cùng **0.97** và cùng format
1.0. Trên `final_loss` thì `attn_only` **thắng** (0.538 < 0.6255). Thứ tự hai bảng **khác
nhau**, và `attn_only` còn **nhanh hơn 1.56×** khi suy luận (881.8 ms vs 1373.8 ms) vì chỉ có
2 module được gắn adapter thay vì 12.

Điều đó nói gì về *rank* so với *vị trí*? **Trên tác vụ hẹp này, không cái nào là đòn bẩy
lớn.** Nâng rank từ 16 lên 283 để bù ngân sách **không mua được gì trên tập target** — nếu
rank là đòn bẩy thật, `attn_only` @ r=283 phải thắng rõ; nó không thắng. Và việc gắn adapter
vào toàn bộ linear của text decoder (`correct`) so với chỉ q,v (`attn_only`) cũng không tạo ra
khác biệt target đo được. Deck §11.2 gọi attention-only là "Lỗi #1"; số đo của tôi **không**
ủng hộ việc nó *thua* trên tác vụ này — nó hoà. Đó là một kết quả đúng và tôi báo cáo đúng
như nó là.

> **Độ phân giải giờ đã đủ.** Ở iteration 1 tôi phải nói "hoà này có thể là nhiễu vì n=8".
> Ở full eval 50 mẫu, 1 mẫu = 0.02 và **cả hai đều đạt 0.97 = 194/200 trường đúng** — hoà
> trên 200 trường, không phải trên 32. Kết luận vững hơn hẳn.

### 4.2 — `wrong_lr`: chỉ khác đúng một con số

`wrong_lr` khác `correct` **duy nhất** ở `learning_rate`: 1e-5 thay vì 1e-4 — tức bằng đúng
thang full-fine-tune, sai 10× so với "vùng không hối tiếc" của deck §11.3.

Đường loss khác nhau rõ rệt. `correct`: 2.163 → 1.378 → 0.1385 → 0.02859 → 0.01695 → 0.02747.
`wrong_lr`: 2.163 → 2.066 → 1.606 → 1.326 → 1.141 → 1.119. Nó **giảm rồi phẳng** ở ~1.1,
không bao giờ đi xuống dưới 1.0; `final_loss` 1.5702. Trên tập target: **0.000 target, 0.000
format, và latency tệ nhất trong bốn run (5164.6 ms)** — model trả lời lan man, không parse
được JSON.

Nếu chỉ nhìn loss mà không biết LR, kết luận sai sẽ là: *"run này học kém vì dữ liệu quá ít
hoặc thiếu step"* — và cách sửa sai theo sẽ là tăng epoch hoặc tăng rank. Cả hai đều vô ích:
nguyên nhân là **LR sai thang 10×**, không phải thiếu capacity. Đây là run **duy nhất** mà
cột `final_loss` xếp hạng đúng (1.5702 rõ ràng tệ nhất) — vì sai 10× là sai đủ lớn để lộ ra
ngay cả trên một chỉ số tồi. Đúng như deck §11.3 dự đoán.

### 4.3 — `qlora`: tiết kiệm bao nhiêu, trả giá bằng gì?

`qlora` tiết kiệm **3.86 GB vs 8.78 GB = 56% VRAM**. Đó là con số lớn và có thật.

Trả giá bằng **bốn thứ**:

1. **Chất lượng thấp hơn**: target 0.94 vs 0.97 (−0.03, tương đương **6 trường sai trên 200**
   so với 194/200 của `correct`).
2. **Train chậm hơn**: 468.7 s vs 389.9 s (+20%) — lượng tử/giải lượng tử tốn thời gian.
3. **Suy luận chậm hơn**: 1749.3 ms vs 1373.8 ms (+27%).
4. **Cần vá precision**: log NB4 in `precision fix: recast 496/496 trainable tensors bf16 ->
   fp32 for the fp16 GradScaler`. Trên T4 (không có bf16 phần cứng), TRL trả về trọng số LoRA
   bf16 trong khi `fp16=True` đã bật GradScaler — và kernel
   `_amp_foreach_non_finite_check_and_unscale_cuda` **không có overload BFloat16**, nên run
   sẽ chết ở optimizer step đầu nếu không recast. Ở 16-bit thì `recast 0 of 496` (no-op).

**Số đo của tôi có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?** **Có,
một nửa.** Deck §13 nói sai số lượng tử hoá của dòng 2026 này cao hơn bình thường. Đo được:
VRAM tiết kiệm là **thật và lớn** (56%), nhưng chất lượng **mất** (−0.03 target) và bạn trả
thêm thời gian ở cả train lẫn suy luận. Với tier `T4`, bf16/fp16 LoRA **vẫn vừa** (8.78 GB
trên 14.6 GB khả dụng), nên không có lý do phải đánh đổi. QLoRA chỉ đáng cân nhắc khi VRAM là
ràng buộc cứng — ví dụ tier `LAPTOP` 8 GB.

---

## 5. Phán quyết (NB5) — **FAILED**

**Kết quả cổng hồi quy**: **FAILED**
`target Δ = +0.205` · `regression Δ = −0.202` · `valid_trace_rate = 0.0`

`results/verdict.json`:
`"general capability regressed by 0.202 (tolerance 0.020). See deck §6.3 — add 1-5% replay data."`

### 5.1 Diễn giải

Bản fine-tune **thắng** ở phần nó được huấn luyện: target **0.97 vs 0.765** của baseline (b)
— chênh **+0.205**, tương đương **194/200 trường đúng so với 153/200**. Format hoàn hảo
(1.0), và suy luận chỉ chậm hơn (b) 1.32×. Nếu chỉ nhìn target, đây là một thắng lợi rõ ràng.

Nhưng cổng hồi quy đòi **hai** điều kiện, và điều kiện thứ hai đã sập: general capability
tụt từ **0.7911 xuống 0.5889**, tức **−0.2022**, trong khi ngưỡng chịu đựng là `0.02`. Đó là
**gấp 10.1× ngưỡng**. Quy ra số câu: **11.87/15 → 8.83/15**, tức mất khoảng **3 câu** trong
15 câu hỏi kiến thức phổ thông (Hà Nội, 1024, Nguyễn Du, 12 tháng, Sài Gòn…).

Đây chính là "thảm hoạ quên" mà deck §6.3 mô tả, và cổng hồi quy bắt được nó **trong khi
target vẫn tăng**. Nếu lab này chỉ chấm bằng target — hoặc bằng perplexity — thì bản
fine-tune này đã được khen và đem đi deploy.

### 5.2 Vì sao điều này xảy ra

Có ba nguyên nhân cộng dồn, xếp theo mức độ chắc chắn:

1. **Không có replay data.** 225 mẫu train **100% là ticket CSKH → JSON**. Không một mẫu nào
   là hội thoại phổ thông. Model được huấn luyện 30 step liên tục trên một phân phối hẹp duy
   nhất, nên nó **quên** cách trả lời câu hỏi ngoài miền đó. Deck §6.3 đề xuất trộn 1–5% dữ
   liệu phổ thông; bản chạy này trộn **0%**. Đây là nguyên nhân trực tiếp và có thể sửa ngay.
2. **Tỉ lệ dữ liệu huấn luyện/tổng phân phối quá cao.** 225 mẫu × 2 epoch = 30 step, mỗi
   step đều là triage. Không có tín hiệu nào kéo model về hành vi tổng quát.
3. **Định dạng đầu ra bị áp đặt.** Sau fine-tune, model có xu hướng trả lời *mọi thứ* theo
   khuôn JSON 4 trường. Trên câu hỏi "Thủ đô của Việt Nam là thành phố nào?", câu trả lời
   đúng cần chứa `"Hà Nội"`; nếu model trả về một object JSON hoặc văn phong triage thì
   `keyword_recall` = 0.

**Bằng chứng ủng hộ nguyên nhân #1 là chính:** regression của (a) và (b) **giống hệt nhau**
(0.7911 cả hai) vì cả hai đều là base model. Chỉ có (c) — bản duy nhất được train — tụt.
Vậy mất mát đến **từ việc huấn luyện**, không từ prompt và không từ tập eval.

### 5.3 ⚠️ Phán quyết ĐẢO CHIỀU: iteration 1 PASSED, iteration 2 FAILED

Đây là kết quả quan trọng nhất của cả lab, và tôi trình bày nó một cách có hệ thống.

| | **Iteration 1** (smoke) | **Iteration 2** (full eval) |
|---|---|---|
| `EVAL_LIMIT` | `8` | *(rỗng — full)* |
| `n_target` / `n_regression` | **8 / 8** | **50 / 15** |
| `smoke_mode` | `true` | `false` |
| (a) target / regression | 0.0 / 0.75 | 0.0 / 0.7911 |
| (b) target / regression | 0.6875 / 0.75 | 0.765 / 0.7911 |
| (c) target / regression | **0.9375** / 0.75 | **0.97** / **0.5889** |
| `target Δ` | +0.250 | +0.205 |
| `regression Δ` | **+0.000** | **−0.202** |
| **Phán quyết** | **PASSED** | **FAILED** |
| `verify.py` | `[ FAIL ] full eval set used` | `[ ok ] full eval set used  50 target items` |

**Bất biến giữa hai lần chạy** (kiểm chứng được, nên phép so sánh là công bằng):

| Yếu tố | Giá trị | Bằng chứng |
|---|---|---|
| Corpus | `train_seed` sha `2a58fe45d7315da0`, `eval_target` sha `2991050fefb848c8` | `data/checksums.json`; `verify.py` `[ ok ] eval sets unmodified` |
| Prompt (b) | sha `719e74d3b6232053` | `verify.py` `[ ok ] baseline (b) prompt unmodified` |
| Base model | `unsloth/Qwen3.5-4B` | `results/runs.csv` |
| Seed / steps | 42 / 30 | `verify.py` `[ ok ] all runs share ONE step budget` |
| Trainable params | 32,464,896 | `runs.csv` |

**Điều duy nhất thay đổi là kích thước tập eval.** Đó là điều kiện lý tưởng để kết luận:
không có biến gây nhiễu nào.

**Vì sao kết luận lật ngược?** Vì ở iteration 1, **tập regression chỉ có 8 mẫu** trong khi nó
có 15 câu. Ngưỡng chịu đựng là `0.02` — trên 8 mẫu, bước nhảy nhỏ nhất là `1/8 = 0.125`,
tức **gấp 6.25× ngưỡng**. Nói cách khác: **ở n=8, cổng hồi quy không có khả năng phát hiện
vi phạm**, vì mọi thay đổi đo được đều hoặc bằng 0 hoặc đã vượt ngưỡng từ lâu. Nó báo
`regression Δ = +0.000` không phải vì model không quên, mà vì **8 mẫu không đủ độ phân giải
để thấy nó quên**.

Trên 15 mẫu, `1/15 = 0.0667` — vẫn thô hơn ngưỡng 0.02 khoảng 3.3×, nhưng đủ để thấy mất mát
lớn (3 câu). Một tập regression **15 mẫu vẫn còn quá nhỏ** để đo chính xác mức quên; nhưng nó
đủ để **bác bỏ** kết luận PASSED sai trước đó.

### 5.4 Vì sao `valid_trace_rate = 0.0` — và tại sao nó KHÔNG phải reasoning-trace collapse

Đây là điểm dễ bị đọc sai nhất, nên tôi nói thẳng: **0.0 ở đây là hệ quả cấu trúc của dữ
liệu, không phải suy luận bị phá huỷ.**

Ba lý do, kiểm chứng được:

1. **Dataset không có trace.** Cả 250 câu trả lời huấn luyện là **JSON trần**, ví dụ
   `{"intent": "doi_tra", ...}`. Không câu nào chứa khối `think`.
2. **Template đóng `think` rỗng trong generation prompt.** Như `mask_proof.json` cho thấy,
   chuỗi `think` mở/đóng rỗng được render như một phần của *generation prompt*, không phải
   của câu trả lời.
3. **Eval sinh với `enable_thinking` mặc định = `False`**, nên model không bao giờ phát ra
   khối `think` để mà đo.

`valid_reasoning_trace()` đòi khối `think` **tồn tại, đóng, và không rỗng** (≥10 ký tự thân).
Không có `think` ⇒ 0.0. Chính `labkit` xác nhận chẩn đoán này: `data.to_training_dataset`
cảnh báo `mask_mode='masked-think' is a no-op on this corpus` với cùng lý do.

**Phân biệt quan trọng:** `regression Δ = −0.202` là **quên thật** (đo được, có nguyên nhân,
có cách sửa). `valid_trace_rate = 0.0` là **không đo được** (không có gì để đo). Hai con số
này trông cùng "tệ" nhưng bản chất ngược nhau, và gộp chúng lại sẽ là một kết luận sai.

---

## 6. Định tính

Nguồn: `results/qualitative.json` (50 mẫu của tập target, sắp theo `ft_score`).

**Thống kê:** 44/50 mẫu đạt **1.0** (đúng cả 4 trường); **6/50 mẫu** đạt **0.75** (sai đúng
1 trường). Không mẫu nào dưới 0.75.

| # | Ticket (rút gọn) | Nhãn đúng | (c) fine-tune | Nhận xét |
|---|---|---|---|---|
| 1 | `Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện. Cảm ơn shop nhiều.` | `hoan_tien / **thap** / bình giữ nhiệt / tich_cuc` | `hoan_tien / **trung_binh** / …` | ❌ **FT 3/4** — sai `urgency` |
| 2 | `Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. Khi nào tiện. Cho tôi hỏi.` | `san_pham_loi / **thap** / nồi chiên không dầu / trung_tinh` | `san_pham_loi / **trung_binh** / …` | ❌ **FT 3/4** — sai `urgency` |
| 3 | `Shop ơi, mình đặt áo khoác gió mã đơn VN613097. Bị lỗi. Khi nào tiện. Cảm ơn shop nhiều.` | `san_pham_loi / **thap** / áo khoác gió / tich_cuc` | `san_pham_loi / **trung_binh** / …` | ❌ **FT 3/4** — sai `urgency` |
| 4 | `Chào shop, mình đặt nồi chiên không dầu mã đơn VN949966. Hoàn tiền. Khi nào tiện. Quá tệ.` | `hoan_tien / **thap** / nồi chiên không dầu / tieu_cuc` | `hoan_tien / **trung_binh** / …` | ❌ **FT 3/4** — sai `urgency` |
| 5 | `Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại. Gấp. Shop hỗ trợ tốt.` | `doi_tra / cao / chuột không dây / tich_cuc` | `doi_tra / cao / chuột không dây / tich_cuc` | ✅ FT 4/4 |
| 6 | `Alo shop, mình đặt máy xay sinh tố mã đơn OD126693. Muốn đổi. Đã 3 ngày rồi. Bực mình.` | `doi_tra / trung_binh / máy xay sinh tố / tieu_cuc` | `doi_tra / trung_binh / máy xay sinh tố / tieu_cuc` | ✅ FT 4/4 |

*(4 ca sai đầu là 4 trong 6 ca yếu nhất; xem `results/qualitative.json` cho cả 50.)*

### 6.1 Mẫu chung ở các ca fine-tune sai — một phát hiện khớp hoàn hảo

Đây là phát hiện chắc chắn nhất tôi có được, và nó **khớp 6/6**:

- **Cả 6 ca sai đều sai đúng một trường: `urgency`.**
- **Cả 6 đều có nhãn đúng là `thap`.**
- **Cả 6 đều bị model đoán thành `trung_binh`.**
- **Cả 6 đều chứa cụm `"khi nào tiện"`** trong ticket.

Và kiểm tra ngược lại cũng khớp — chạy trên chính `results/qualitative.json`:

```
items with marker: 6 -> [3, 5, 12, 39, 41, 46]
of those, FT wrong on: [3, 5, 12, 39, 41, 46]
items with marker where FT was RIGHT: []
urgency distribution of the marker items: Counter({'thap': 6})
```

Nói cách khác: **`"khi nào tiện"` là marker của lớp `thap` trong `scripts/make_seed_data.py`
(`URGENCY_MARKERS["thap"] = ["khi nào tiện", "không vội", "hỏi cho biết thôi"]`), và model bỏ
sót nó 100% số lần.** Đây không phải nhiễu ngẫu nhiên — nó là một lỗ hổng hệ thống, định vị
được, với 6 bằng chứng độc lập.

**Giả thuyết:** `trung_binh` là lớp đa số trong phân phối train, nên khi tín hiệu yếu model
nghiêng về prior thay vì đọc marker. **Cách sửa cụ thể:** tăng tỉ lệ mẫu `urgency = thap` có
marker rõ trong train, hoặc thêm ví dụ few-shot cho `thap` vào prompt (b).

> **Giới hạn phải nói rõ.** Vì bundle **không chứa dự đoán per-item của baseline (b)** (chỉ
> có điểm tổng 0.765), tôi **không** xác định được ca nào fine-tune *thua (b)* theo từng mẫu
> — rubric 3.4 yêu cầu ≥2 ca như vậy. Cái tôi trình bày được là **6 ca fine-tune sai**, và
> chúng là bằng chứng thật, không phải ca thua được bịa ra. Tôi ghi rõ cột "(b) prompt" là
> không thu được thay vì để trống ngầm.

---

## 7. Kết luận & điều tôi học được

### Kết luận

**Tôi không nên deploy bản fine-tune này, và lần này lý do rất rõ ràng: nó quên.** Bản
fine-tune đạt target 0.97 — vượt baseline (b) 0.765 một khoảng +0.205, tương đương 194/200
trường đúng so với 153/200. Đó là một cải thiện thật trên tác vụ được huấn luyện. Nhưng nó
đồng thời làm general capability sụp từ 0.7911 xuống 0.5889, tức −0.202 — **gấp 10.1× ngưỡng
chịu đựng 0.02**. Cổng hồi quy FAILED, và nó đúng khi FAILED: một model trả lời ticket CSKH
giỏi nhưng quên thủ đô Việt Nam là gì thì không phải một model tốt hơn, chỉ là một model hẹp
hơn. Bản này cần **trộn 1–5% dữ liệu hội thoại phổ thông** (deck §6.3) rồi chạy lại, không
phải thêm training.

**Đòn bẩy thật sự trong lab này là learning rate, và ngay sau đó là dữ liệu — không phải rank
hay vị trí.** Nhìn biên độ đo được ở full eval, xếp hạng rõ ràng:

| Nút vặn | Thay đổi | Δ target | Ghi chú |
|---|---|---|---|
| **Learning rate** | 1e-4 → 1e-5 | **−0.97** (sụp hoàn toàn) | biên độ lớn nhất |
| **Prompt** (a→b) | naive → optimized | **+0.765** | và nhanh hơn 3.14× |
| **Precision** (qlora) | 16-bit → 4-bit | −0.03 | đổi lấy −56% VRAM |
| **Vị trí / rank** (attn_only) | text-linear r16 → q,v **r283** | **0.0** (hoà) | 194/200 vs 194/200 |
| **Dữ liệu** (không replay) | 0% replay | *(không đổi target)* | **−0.202 regression** |

Sai LR một thang 10× phá huỷ **toàn bộ** kết quả. Nâng rank gấp 17.7× để bù ngân sách tham
số **không mua được gì**. Prompt tử tế cho +0.765 — nhiều hơn mọi thứ LoRA làm được trong
lab này. Và thành phần dữ liệu — thứ không hề xuất hiện trong bảng target — là thứ duy nhất
**phá hỏng phán quyết**. Nói cách khác: **thí nghiệm trung tâm của lab phiên bản cũ (quét
rank r=8/16/64) là nút vặn ít ảnh hưởng nhất trong năm nút.**

Và điều này khép lại đúng luận điểm của lab: **điểm không nằm ở chỗ fine-tune thắng, mà ở
chỗ tôi biết nó thắng nhờ đâu và thắng chắc tới đâu.** Iteration 1 nói "PASSED, +0.250".
Iteration 2 nói "FAILED, quên mất 3/15 câu". Cùng một model. Điều khác biệt duy nhất là **tôi
đã đo trên đủ mẫu để thấy sự thật** — và đó là toàn bộ giá trị của việc đóng băng tập eval
trước khi train.

### Ba điều tôi học được

1. **Cỡ mẫu của tập eval quyết định kết luận, không chỉ độ chính xác của nó.** Cùng model,
   cùng seed, cùng 30 step, cùng corpus, cùng prompt — chỉ khác `n_regression` (8 vs 15) —
   và phán quyết lật từ PASSED sang FAILED. Ở n=8, bước nhảy nhỏ nhất là `1/8 = 0.125`, gấp
   **6.25×** ngưỡng chịu đựng `0.02`, nên cổng **không có khả năng** phát hiện vi phạm. Tôi
   đã suýt nộp một kết luận sai và gọi nó là thành công. Từ giờ, trước khi tin một cổng
   kiểm tra, tôi hỏi: *cỡ mẫu này có đủ để cổng báo động không?*

2. **`final_loss` xếp hạng đúng 1/4 run — và đó là bài học đắt nhất.** Bảng train loss nói
   `attn_only` (0.538) tốt nhất, trên `correct` (0.6255). Bảng target nói chúng **hoà**
   (194/200 cả hai). Hai thứ tự khác nhau ngay ở vị trí đầu. Nếu tôi kết luận bằng cột loss —
   như lab cũ làm, và như rubric gọi là "Lỗi #3" — tôi đã báo cáo một kết quả **ngược**. Chỉ
   `wrong_lr` được loss xếp đúng, vì sai 10× là sai đủ lớn để lộ ra kể cả trên chỉ số tồi.

3. **Một chỉ số không được thiết kế cho dữ liệu của bạn sẽ nói dối bằng số 0.**
   `valid_trace_rate = 0.0` trông như thảm hoạ "reasoning-trace collapse", nhưng dataset là
   JSON trần, template không bao giờ phát `think`, eval tắt thinking — nên 0.0 là **cấu
   trúc**, không phải hiện tượng. Ngược lại, `regression Δ = −0.202` là **quên thật**. Hai
   con số cùng trông "tệ" nhưng bản chất ngược nhau; gộp chúng lại là một kết luận sai. Giờ
   tôi kiểm tra: chỉ số này *có thể* khác 0 với dữ liệu này không, trước khi kết luận từ nó.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** (1) **trộn 1–5% replay data** (chính các câu hỏi phổ
thông, hoặc một tập tương tự) vào train rồi chạy lại — đây là cách sửa trực tiếp cho nguyên
nhân đã xác định, và là thí nghiệm duy nhất có khả năng biến FAILED thành PASSED một cách
trung thực; (2) export dự đoán per-item của (b) để §6 có đủ cột và xác định được ca *thua (b)*
mà rubric 3.4 yêu cầu; (3) tăng tỉ lệ mẫu `urgency = thap` — giả thuyết từ 6 ca sai, nơi model
bỏ sót marker "khi nào tiện" **6/6 lần**.

---

## 8. Phụ lục — nhật ký hai iteration

Hai lần chạy trên cùng base model, cùng corpus, cùng seed 42, cùng 30 optimizer step, cùng
`OPTIMIZED_PROMPT` (sha `719e74d3b6232053`). **Biến duy nhất là kích thước tập eval.**

| | **v1 — 2026-10-07** (iteration 1) | **v2 — 2026-10-07** (iteration 2) |
|---|---|---|
| `EVAL_LIMIT` | `8` (mặc định của notebook) | *(rỗng)* |
| `n_target` / `n_regression` | 8 / 8 | **50 / 15** |
| `smoke_mode` | `true` | `false` |
| (a) target / regression / format / ms | 0.0 / 0.75 / 0.0 / 3374.2 | 0.0 / 0.7911 / 0.0 / 3273.6 |
| (b) target / regression / format / ms | 0.6875 / 0.75 / 1.0 / 936.9 | 0.765 / 0.7911 / 1.0 / 1043.2 |
| (c) target / regression / format / ms | 0.9375 / 0.75 / 1.0 / 1476.4 | **0.97** / **0.5889** / 1.0 / 1373.8 |
| `correct` final_loss | 0.6274 | 0.6255 |
| `attn_only` target | 0.9375 | 0.97 |
| `qlora` target | 0.8438 | 0.94 |
| `wrong_lr` target | 0.0 | 0.0 |
| `target Δ` | +0.250 | +0.205 |
| `regression Δ` | **+0.000** | **−0.202** |
| **Phán quyết** | **PASSED** | **FAILED** |
| Gatekeeper | `[ FAIL ] full eval set used` | `[ ok ] full eval set used  50 target items` |
| Thời gian core | 2261 s (37.7 ph) | 1470 s (24.5 ph) |
| Bằng chứng | `docs/evidence/v1_2026-10-07_1145_cell3_pipeline.log` | `docs/evidence/v2_2026-10-07_1205_cell3_pipeline.log` |

**Ghi chú kỹ thuật về iteration 2.** NB4 báo `skip attn_only / skip wrong_lr / skip qlora:
adapters/<key>/ already trained` và chỉ train lại `correct` — vì ba adapter đối chứng đã có
sẵn trên VM từ iteration 1, còn `adapters/correct/` đã bị xoá để train lại. Vì
`report.append_row` **append chứ không upsert**, `runs.csv` có **hai dòng `correct`** (0.6274
của iteration 1 và 0.6255 của iteration 2). NB4 tự đọc **dòng cuối mỗi key**, và
`tools/report_tables.py` cũng vậy — nên bảng ở §4 hiển thị `correct` một lần với `final_loss
= 0.6255`, đúng adapter đang tồn tại trên đĩa. Xem `tools/report_tables.py::load_results`.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:

> **Chưa làm bonus nào.** Lý do cụ thể từng cái:
>
> - **B1** (merge + hot-swap, +3): bundle chỉ export adapter `correct`, và chỉ
>   `adapter_config.json` — không có trọng số. `attn_only`, `wrong_lr`, `qlora` không được
>   export khỏi VM. Cần một phiên Colab còn sống để lấy lại.
> - **B3** (reasoning-trace collapse, +4): **không thể làm trên corpus mặc định** —
>   `masked-think` / `response-only` là no-op ở đây (xem §2 và §5.4). Cần dataset có trace thật.
> - **B4** (quét rank, +3): ~1 giờ GPU; nên làm cùng lần chạy sửa replay data.
> - **B2** (dataset miền riêng, +3): ~2 giờ viết ≥200 mẫu + `data/CUSTOM_DATASET.md`.
> - **B5** (HF Hub, +2): cần `HF_TOKEN`; log cho thấy các lần tải đều
>   `unauthenticated requests to the HF Hub`.
