# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Trọng Minh  **MSSV**: 02496  **Ngày**: 2026-10-07
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB (14.6 GB khả dụng, sm_75)`

> **Phiên bản**: `v1_2026-10-07` — **iteration 1, smoke run** (`EVAL_LIMIT=8`, `n_target=8` không phải 50).
> Xem `docs/RUNS.md` để biết trạng thái từng bản chạy. Bản nộp cuối sẽ chạy lại ở chế độ full eval.
>
> Mọi con số dưới đây khớp với file trong `results/`. Grader kiểm tra chéo.
> Bảng số được **sinh tự động** bằng `python tools/report_tables.py` (nguồn: `results/*.json`),
> không chép tay — kiểm tra bằng `python tools/report_tables.py --check`.

> ### ⚠️ Đọc trước: hai giới hạn của bản v1
>
> **1. Đây là smoke run.** `results/baselines_frozen.json` ghi `"eval_limit": 8`,
> `"smoke_mode": true`, `"n_target": 8`. Mọi số target / regression / latency tính trên
> **8 mẫu**, nên **1 mẫu = 0.125** — thô gấp ~6 lần tập đầy đủ 50 mẫu (1 mẫu = 0.02).
> `scripts/verify.py` báo `[ FAIL ] full eval set used` đúng vì lý do này.
>
> **2. Không thu được dự đoán per-item của baseline (b).** Bundle không chứa chúng. Hệ quả
> trực tiếp: §6 không điền được cột "(b) prompt" và **không xác định được ca fine-tune thua**
> mà rubric 3.4 yêu cầu ≥2. Tôi khai báo thẳng giới hạn này ở §6 thay vì bịa ca thua.

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
không có (xem §5) — nhưng điều đó là do dữ liệu, không do template.

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
đúng vì lý do này. Đây là lý do §5 có `valid_trace_rate = 0.0`.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

Bảng sinh từ `results/verdict.json`:

| run | target | regression | format | latency_ms | n |
|---|---|---|---|---|---|
| (a) base + naive prompt | 0.0 | 0.75 | 0.0 | 3374.2 | 8 |
| (b) base + optimized prompt | 0.6875 | 0.75 | 1.0 | 936.9 | 8 |
| (c) LoRA fine-tune | 0.9375 | 0.75 | 1.0 | 1476.4 | 8 |

**(b) có thật sự mạnh hơn (a) không?** **Có** — `verify.py` xác nhận
`[ ok ] baseline (b) beats (a)  (a)=0.000 -> (b)=0.688`.

Và đây là điểm đáng chú ý nhất ở §3: **(b) vừa chính xác hơn vừa RẺ HƠN (a) ~3.6×**
(3374.2 ms → 936.9 ms). Prompt ngây thơ `"Phân loại ticket sau."` khiến model trả lời bằng
văn xuôi tiếng Việt — `format = 0.000`, không parse được JSON, và sinh dài nên chậm. Prompt
tối ưu có schema + ví dụ nên model đi thẳng vào JSON ngắn. Nghĩa là **prompt engineering
không phải là "chi phí để so sánh" mà tự nó đã là một cải thiện lớn** — đúng luận điểm
deck §21: bản fine-tune phải vượt *một baseline đã được prompt tử tế*, không phải một
baseline bị dựng lên cho yếu.

**Bạn có sửa `OPTIMIZED_PROMPT` không?** **Không.** `optimized_prompt_sha = 719e74d3b6232053`
khớp bản gốc của lab; `verify.py` xác nhận `[ ok ] baseline (b) prompt unmodified`. Tôi không
làm yếu (b) — đó là cách chính để gian lận lab này và rubric gọi thẳng là lỗi liêm chính.

---

## 4. Giải phẫu cấu hình sai (NB4)

Bảng sinh từ `results/runs.csv` và `results/autopsy.json`:

| run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32464896 | 0.0001 | 0.6274 | **0.9375** | 391.4 | 8.78 |
| `attn_only` | attn-only | 283 | 32456704 | 0.0001 | **0.538** | **0.9375** | 258.5 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32464896 | 1e-05 | 1.5702 | **0.0** | 396.9 | 8.78 |
| `qlora` | text-linear | 16 | 32464896 | 0.0001 | 0.7058 | 0.8438 | 468.7 | **3.86** |

Cả bốn run dùng **cùng 30 optimizer step** (`verify.py`: `[ ok ] all runs share ONE step budget
['attn_only','correct','qlora','wrong_lr'] at 30 steps`), nên khác biệt duy nhất giữa mỗi
contrast và `correct` là **đúng một biến**.

**Xếp hạng theo `target`:** `correct` = `attn_only` (0.9375) > `qlora` (0.8438) > `wrong_lr` (0.0).
**Xếp hạng theo `final_loss`:** `attn_only` (0.538) > `correct` (0.6274) > `qlora` (0.7058) > `wrong_lr` (1.5702).

**Hai thứ tự này KHÁC NHAU ở vị trí đầu tiên.** Đây là kết quả đáng giá nhất tôi đo được
trong lab, và nó được bàn ở 4.1.

### 4.1 — `attn_only` vs `correct`: rank có phải đòn bẩy không?

`attn_only` dùng **cùng ngân sách tham số** với `correct`: 32,456,704 vs 32,464,896, lệch
**0.025%** — thoải mái dưới ngưỡng 5% của rubric 2.1 (`verify.py`: `[ ok ] attn_only is a FAIR
contrast  32,456,704 vs 32,464,896 trainable params`). Để đạt ngân sách đó, rank phải nâng từ
**16 lên 283** (alpha 566) — tức gấp ~17.7×.

Trên tập target, `attn_only` **HOÀ** với `correct`: cùng 0.9375 và cùng format 1.0. Trên
`final_loss` thì `attn_only` **thắng** (0.538 < 0.6274). Thứ tự hai bảng **khác nhau**, và
`attn_only` còn **nhanh hơn 1.8×** khi suy luận (818.7 ms vs 1476.4 ms) vì chỉ có 2 module
được gắn adapter thay vì 12.

Điều đó nói gì về *rank* so với *vị trí*? **Trên tác vụ hẹp này, không cái nào là đòn bẩy
lớn.** Nâng rank từ 16 lên 283 để bù ngân sách **không mua được gì trên tập target** — nếu
rank là đòn bẩy thật, `attn_only` @ r=283 phải thắng rõ; nó không thắng. Và việc gắn adapter
vào toàn bộ linear của text decoder (`correct`) so với chỉ q,v (`attn_only`) cũng không tạo ra
khác biệt target đo được. Deck §11.2 gọi attention-only là "Lỗi #1"; số đo của tôi **không**
ủng hộ việc nó *thua* trên tác vụ này — nó hoà. Đó là một kết quả đúng và tôi báo cáo đúng
như nó là.

> **Giới hạn phải nói rõ:** n=8, 1 mẫu = 0.125. "Hoà" ở đây có nghĩa là *khác biệt nhỏ hơn
> độ phân giải của thang đo*, chứ không chắc là bằng nhau thật. Cần full eval 50 mẫu để phân
> biệt. Tôi không kết luận quá mức từ 8 mẫu.

### 4.2 — `wrong_lr`: chỉ khác đúng một con số

`wrong_lr` khác `correct` **duy nhất** ở `learning_rate`: 1e-5 thay vì 1e-4 — tức bằng đúng
thang full-fine-tune, sai 10× so với "vùng không hối tiếc" của deck §11.3.

Đường loss khác nhau rõ rệt. `correct`: 2.163 → 1.386 → 0.1425 → 0.02885 → 0.0176 → 0.02614.
`wrong_lr`: 2.163 → 2.066 → 1.606 → 1.326 → 1.141 → 1.119. Nó **giảm rồi phẳng** ở ~1.1,
không bao giờ đi xuống dưới 1.0; `final_loss` 1.5702. Trên tập target: **0.000 target, 0.000
format, và latency tệ nhất trong bốn run (4887.8 ms)** — model trả lời lan man, không parse
được JSON.

Nếu chỉ nhìn loss mà không biết LR, kết luận sai sẽ là: *"run này học kém vì dữ liệu quá ít
hoặc thiếu step"* — và cách sửa sai theo sẽ là tăng epoch hoặc tăng rank. Cả hai đều vô ích:
nguyên nhân là **LR sai thang 10×**, không phải thiếu capacity. Đây là run **duy nhất** mà
cột `final_loss` xếp hạng đúng (1.5702 rõ ràng tệ nhất) — vì sai 10× là sai đủ lớn để lộ ra
ngay cả trên một chỉ số tồi. Đúng như deck §11.3 dự đoán.

### 4.3 — `qlora`: tiết kiệm bao nhiêu, trả giá bằng gì?

`qlora` tiết kiệm **3.86 GB vs 8.78 GB = 56% VRAM**. Đó là con số lớn và có thật.

Trả giá bằng **bốn thứ**:

1. **Chất lượng thấp hơn**: target 0.8438 vs 0.9375 (−0.0938).
2. **Train chậm hơn**: 468.7 s vs 391.4 s (+20%) — lượng tử/giải lượng tử tốn thời gian.
3. **Suy luận chậm hơn**: 1791.3 ms vs 1476.4 ms (+21%).
4. **Cần vá precision**: log NB4 in `precision fix: recast 496/496 trainable tensors bf16 ->
   fp32 for the fp16 GradScaler`. Trên T4 (không có bf16 phần cứng), TRL trả về trọng số LoRA
   bf16 trong khi `fp16=True` đã bật GradScaler — và kernel
   `_amp_foreach_non_finite_check_and_unscale_cuda` **không có overload BFloat16**, nên run
   sẽ chết ở optimizer step đầu nếu không recast. Ở 16-bit thì `recast 0 of 496` (no-op).

**Số đo của tôi có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?** **Có,
một nửa.** Deck §13 nói sai số lượng tử hoá của dòng 2026 này cao hơn bình thường. Đo được:
VRAM tiết kiệm là **thật và lớn** (56%), nhưng chất lượng **mất** (−0.0938 target) và bạn trả
thêm thời gian ở cả train lẫn suy luận. Với tier `T4`, bf16/fp16 LoRA **vẫn vừa** (8.78 GB
trên 14.6 GB khả dụng), nên không có lý do phải đánh đổi. QLoRA chỉ đáng cân nhắc khi VRAM là
ràng buộc cứng — ví dụ tier `LAPTOP` 8 GB.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: **PASSED**
`target Δ = +0.250` · `regression Δ = +0.000` · `valid_trace_rate = 0.0`

`results/verdict.json`:
`"beat the optimized-prompt baseline by +0.250 with +0.000 general-capability change."`

### Diễn giải

Bản fine-tune **vượt baseline (b)** — mốc thật sự phải vượt — với target 0.9375 so với 0.6875,
và **không làm tụt** general capability (regression giữ nguyên 0.75 ở cả (a), (b) và (c)).
Cổng hồi quy đòi hai điều kiện: target tốt hơn (b) **và** regression không tụt quá
`REGRESSION_TOLERANCE = 0.02`. Cả hai đều thoả, nên PASSED.

Điều đáng nói là **regression không đổi chút nào** (0.750 ở cả ba). Trên một tập 15 câu hỏi
kiến thức phổ thông, fine-tune 30 step trên 225 mẫu hẹp **không** gây quên thảm hoạ — đúng
như deck §6.3 dự đoán về lợi thế của LoRA. Nhưng với n=8/15 mẫu, "không đổi" cũng có nghĩa
là *không đủ độ phân giải để phát hiện thay đổi nhỏ*. Một tụt 0.02 trên 15 mẫu là 0.3 mẫu —
không đo được ở kích thước này.

**Vì sao PASSED này chưa đủ để kết luận.** Hai lý do, cả hai đều là giới hạn của bản chạy chứ
không phải của kết quả:

1. **n = 8.** Mỗi mẫu trị giá 0.125. Chênh lệch +0.25 tương đương **đúng 2 mẫu** trong 8. Đó
   là tín hiệu thật nhưng mỏng; khoảng tin cậy rất rộng.
2. **Baseline (b) và (c) được sinh trên cùng 8 mẫu đó**, nên nhiễu giữa hai lần sinh không
   được trung bình hoá.

Kết luận đúng đắn: **PASSED ở bản smoke là dấu hiệu tích cực, không phải bằng chứng.** Cần
full eval 50 mẫu để khẳng định. `docs/MEASURED-T4-2026-08-20.md` của chính lab ghi rằng câu
hỏi đầu đề vẫn **chưa ai đo được** ở ngân sách đầy đủ, và bản v1 này cũng chưa trả lời nó.

### Vì sao `valid_trace_rate = 0.0` — và tại sao nó KHÔNG phải reasoning-trace collapse

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

Muốn đo reasoning-trace collapse **thật** thì phải có dữ liệu có trace — đó là bonus B3, và
nó **không thể** làm trên corpus mặc định. Ghi 0.0 vào bảng mà không giải thích sẽ là báo cáo
một hiện tượng không xảy ra.

---

## 6. Định tính

Nguồn: `results/qualitative.json` (8 mẫu của tập target, sắp theo `ft_score`).

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | `Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện. Cảm ơn shop nhiều.` | `hoan_tien / thap / bình giữ nhiệt / tich_cuc` | *(không thu được ở v1)* | `hoan_tien / trung_binh / bình giữ nhiệt / …` | ⚠️ **FT 3/4** — sai `urgency` |
| 2 | `Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. Khi nào tiện. Cho tôi hỏi.` | `san_pham_loi / thap / nồi chiên không dầu / trung_tinh` | *(không thu được ở v1)* | `san_pham_loi / trung_binh / nồi chiên không dầu / …` | ⚠️ **FT 3/4** — sai `urgency` |
| 3 | `Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại. Gấp. Shop hỗ trợ tốt.` | `doi_tra / cao / chuột không dây / tich_cuc` | *(không thu được ở v1)* | `doi_tra / cao / chuột không dây / tich_cuc` | ✅ FT 4/4 |
| 4 | `Shop ơi, mình đặt ốp lưng điện thoại mã đơn VN812931. Hoàn tiền. Sớm nhé. Bực mình.` | `hoan_tien / trung_binh / ốp lưng điện thoại / tieu_cuc` | *(không thu được ở v1)* | `hoan_tien / trung_binh / ốp lưng điện thoại / tieu_cuc` | ✅ FT 4/4 |
| 5 | `Xin chào, mình đặt đèn bàn LED mã đơn VN880807. Hoàn tiền. Quá hạn rồi. Cảm ơn shop nhiều.` | `hoan_tien / cao / đèn bàn LED / tich_cuc` | *(không thu được ở v1)* | `hoan_tien / cao / đèn bàn LED / tich_cuc` | ✅ FT 4/4 |

### ⚠️ Giới hạn của bản v1 — không có ca "fine-tune thua" để trình bày

Rubric 3.4 yêu cầu ≥5 ví dụ với **≥2 ca fine-tune THUA**. Bản chạy v1 **không tạo ra được**
những ca đó, và tôi nói rõ vì sao thay vì lấp chỗ trống bằng số bịa:

Bundle v1 **không chứa dự đoán per-item của baseline (b)** — chỉ có điểm tổng 0.6875. Muốn
biết fine-tune thua ở mẫu nào thì phải so từng mẫu với (b), mà dữ liệu đó không tồn tại. Trên
8 mẫu này, `ft_score` là `[0.75, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]` — **không mẫu nào dưới
0.75**, và trung bình 0.9375 > 0.6875 của (b). Không có ca thua rõ ràng nào để trình bày.

Nên tôi làm ba việc thay thế:

1. Trình bày **2 mẫu yếu nhất thật** (3/4 trường) — bằng chứng gần nhất với "ca thua" mà bản
   chạy này tạo ra.
2. **Không** cherry-pick chỉ ca thắng rồi bỏ qua phần còn lại: cả 8 mẫu đều được lưu trong
   `results/qualitative.json`, và bảng trên lấy từ 5 mẫu trải đều hai đầu.
3. Ghi rõ cột "(b) prompt" là **không thu được**, không để trống ngầm.

**Có mẫu chung nào ở các ca FT yếu không?** **Có, và nó rất rõ.** Cả hai ca 3/4 đều sai
**đúng một trường: `urgency`** — nhãn đúng là `thap`, model đoán `trung_binh`. Nhìn lại ticket:

- `"…Chưa thấy tiền. **Khi nào tiện.** Cảm ơn shop nhiều."` → nhãn `thap`
- `"…Thiếu phụ kiện. **Khi nào tiện.** Cho tôi hỏi."` → nhãn `thap`

Cả hai đều chứa cụm **"khi nào tiện"** — một trong các marker của `thap` trong
`scripts/make_seed_data.py` (`URGENCY_MARKERS["thap"] = ["khi nào tiện", "không vội", "hỏi cho
biết thôi"]`). Model **bỏ sót marker này** và mặc định về `trung_binh`. Giả thuyết hợp lý:
`trung_binh` là lớp đa số trong phân phối train, nên khi tín hiệu yếu model nghiêng về prior.
Điều này gợi ý một cải thiện cụ thể: tăng tỉ lệ mẫu `thap` có marker rõ, hoặc thêm ví dụ
few-shot cho `thap` trong prompt. **Đây là phát hiện hành động được, không phải nhận xét
chung** — và nó đến từ việc đọc 2 ca yếu nhất thay vì chỉ khoe 6 ca đúng.

---

## 7. Kết luận & điều tôi học được

### Kết luận

**Tôi chưa nên deploy bản fine-tune này, nhưng không phải vì nó thất bại.** Cổng hồi quy
PASSED: target 0.9375 vượt (b) 0.6875 một khoảng +0.250, và regression không tụt (0.750 ở cả
ba). Nếu đọc đúng như vậy thì đây là một thắng lợi rõ. Nhưng con số +0.250 đó được đo trên
**8 mẫu**, nơi một mẫu trị giá 0.125 — tức toàn bộ "chiến thắng" chỉ là **2 mẫu trong 8**.
Khoảng tin cậy quá rộng để ký một quyết định triển khai. Điều cần làm trước khi deploy là
chạy full eval 50 mẫu, không phải thêm training.

**Đòn bẩy thật sự trong lab này là learning rate, không phải rank hay vị trí.** Nhìn biên độ
đo được, xếp hạng rõ ràng:

| Nút vặn | Thay đổi | Δ target |
|---|---|---|
| **Learning rate** | 1e-4 → 1e-5 | **−0.9375** (sụp hoàn toàn) |
| **Prompt** (a→b) | naive → optimized | **+0.6875** |
| **Precision** (qlora) | 16-bit → 4-bit | −0.0938 (−56% VRAM) |
| **Vị trí / rank** (attn_only) | text-linear r16 → q,v **r283** | **0.0** (hoà) |

Sai LR một thang 10× phá huỷ **toàn bộ** kết quả. Nâng rank gấp 17.7× để bù ngân sách tham
số **không mua được gì**. Prompt tử tế cho +0.6875 — nhiều hơn mọi thứ LoRA làm được trong
lab này. Nói cách khác: **thí nghiệm trung tâm của lab phiên bản cũ (quét rank r=8/16/64) là
nút vặn ít ảnh hưởng nhất trong bốn nút** — đúng như deck hiện tại gọi nó là Lỗi #1–#3.

Và điều này khép lại đúng luận điểm của lab: **điểm không nằm ở chỗ fine-tune thắng, mà ở
chỗ tôi biết nó thắng nhờ đâu và thắng chắc tới đâu.** Bản v1 biết cả hai: thắng nhờ dữ liệu
hẹp + LR đúng, và chưa chắc vì n=8.

### Ba điều tôi học được

1. **`final_loss` xếp hạng đúng 1/4 run — và đó là bài học đắt nhất.** Bảng train loss nói
   `attn_only` (0.538) tốt nhất, trên `correct` (0.6274). Bảng target nói chúng **hoà**. Hai
   thứ tự khác nhau ngay ở vị trí đầu. Nếu tôi kết luận bằng cột loss — như lab cũ làm, và
   như rubric gọi là "Lỗi #3" — tôi đã báo cáo một kết quả **ngược**. Chỉ `wrong_lr` được
   loss xếp đúng, vì sai 10× là sai đủ lớn để lộ ra kể cả trên chỉ số tồi.

2. **`supervised_fraction = 0.4149` là con số tôi phải hiểu, không chỉ đọc.** 39/94 token
   được tính loss. Nếu nó là 0.95+ thì model đang học viết lại ticket của khách. NB1 cho
   thấy `everything` = 94/94 và mask đúng = 39/94 **cạnh nhau** — nhìn thấy, không phải tin.
   Và điều bất ngờ: `masked-think` / `response-only` là **no-op** trên corpus này vì template
   đóng `think` rỗng *bên trong generation prompt*. Một cờ "an toàn hơn" hoá ra không làm gì
   cả — đúng loại thất bại im lặng mà lab này dạy tôi đi tìm.

3. **Một chỉ số tôi không hiểu rõ còn tệ hơn không có chỉ số.** `valid_trace_rate = 0.0` trông
   như thảm hoạ "reasoning-trace collapse". Nhưng dataset là JSON trần, template không bao
   giờ phát `think`, eval tắt thinking. 0.0 là **cấu trúc**, không phải hiện tượng. Nếu tôi
   báo cáo nó mà không giải thích, tôi đã bịa ra một phát hiện. Giờ tôi kiểm tra: chỉ số này
   *có thể* khác 0 với dữ liệu này không, trước khi kết luận từ nó.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** (1) chạy lại full eval 50 mẫu — câu hỏi chưa được trả
lời, và 2 mẫu/8 quá mỏng; (2) export dự đoán per-item của (b) để §6 có đủ cột và tìm được ca
thua thật; (3) thử tăng tỉ lệ mẫu `urgency = thap` — giả thuyết từ 2 ca yếu nhất là model mặc
định về `trung_binh` khi bỏ sót marker "khi nào tiện".

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:

> **Bản v1 chưa làm bonus nào.** Lý do cụ thể từng cái:
>
> - **B1** (merge + hot-swap, +3): bundle v1 chỉ export adapter `correct`, và chỉ
>   `adapter_config.json` — không có trọng số. `attn_only`, `wrong_lr`, `qlora` không được
>   export khỏi VM. Cần một phiên Colab còn sống để lấy lại.
> - **B3** (reasoning-trace collapse, +4): **không thể làm trên corpus mặc định** —
>   `masked-think` / `response-only` là no-op ở đây (xem §2 và §5). Cần dataset có trace thật.
> - **B4** (quét rank, +3): ~1 giờ GPU; nên làm cùng lần chạy full eval.
> - **B2** (dataset miền riêng, +3): ~2 giờ viết ≥200 mẫu + `data/CUSTOM_DATASET.md`.
> - **B5** (HF Hub, +2): cần `HF_TOKEN`; log cho thấy các lần tải đều
>   `unauthenticated requests to the HF Hub`.
