# Trạng thái các bản chạy

> Ghi lại **chính xác** bản chạy nào đã tạo ra `results/` trong repo, để không ai -
> kể cả tác giả - đọc nhầm một bản smoke là kết quả cuối.
>
> Version của report sống ở **ba chỗ**, không ở tên file (vì `scripts/verify.py`
> hard-code `submission/REPORT.md` trong `REQUIRED_ARTIFACTS`):
> header của `submission/REPORT.md`, commit message, và file này.

## v1 - 2026-10-07 (smoke) * "iteration 1"

| | |
|---|---|
| Ngày | 2026-10-07 |
| Tier | `T4` (Colab Free) |
| GPU | Tesla T4, sm_75, 14.6 GB khả dụng |
| Base model | `unsloth/Qwen3.5-4B` |
| Precision | `fp16` (T4 không có bfloat16) |
| MASK_MODE | `assistant-only` |
| Epochs / max_steps | 2.0 / 30 |
| **`EVAL_LIMIT`** | **`8` - SMOKE MODE** |
| **`n_target` / `n_regression`** | **`8` / `8`** (tập đầy đủ: 50 / 15) |
| Thời gian core NB1-NB5 | 2261 s (~37.7 phút) |
| Kết quả cổng | PASSED - target delta `+0.250`, regression delta `+0.000` |

## Cảnh báo - bản này KHÔNG phải bản nộp cuối

1. **Smoke mode.** `results/baselines_frozen.json` ghi `"smoke_mode": true`,
   `"eval_limit": 8`. Mọi số target / regression / latency tính trên **8 mẫu**, nên
   **1 mẫu = 0.125** - thô gấp ~6 lần tập 50 mẫu (1 mẫu = 0.02).
   `scripts/verify.py` báo `[ FAIL ] full eval set used` đúng vì lý do này.
2. **Thiếu dự đoán per-item của baseline (b).** Bundle không chứa chúng, nên mục §6 của
   `submission/REPORT.md` không có cột "(b) prompt" và không xác định được ca "fine-tune
   thua" mà rubric 3.4 yêu cầu >=2.
3. **Chỉ có adapter `correct` (và chỉ `adapter_config.json`, không có trọng số).**
   `attn_only`, `wrong_lr`, `qlora` không được export khỏi VM. Số liệu train của chúng nằm
   trong `results/runs.csv` và `results/autopsy.json`, nhưng trọng số thì không còn => bonus
   B1 (merge + hot-swap) chưa làm được ở v1.
4. **`valid_trace_rate = 0.0`** là hệ quả cấu trúc của dữ liệu (250 câu trả lời là JSON
   trần, không có khối `think`), **không phải** reasoning-trace collapse. Xem §5 của report.

## Cách chạy lại cho bản nộp cuối

```bash
# Trong Colab, cell 3: để EVAL_LIMIT = ""  (rỗng = full set 50/15)
python scripts/colab_run.py nb1 nb2 nb3 nb4 nb5
```

`verify.py` chỉ xanh khi `EVAL_LIMIT` rỗng. Chọn `""` trong dropdown của cell 3 -
**không** chọn 25: mọi giá trị số đều đặt `smoke_mode = bool(EVAL_LIMIT) = true`.

## Bằng chứng nguồn

- `docs/evidence/v1_2026-10-07_1145_cell3_pipeline.log` - toàn bộ output NB1->NB5 (421 dòng)
- `docs/evidence/v1_2026-10-07_1145_cell4_gatekeeper.log` - output `verify.py` trên Colab (90 dòng)
- `results/` - artefact thô của bản chạy này

---

## v2 - 2026-10-07 (full eval) * "iteration 2" * **BẢN HIỆN HÀNH**

| | |
|---|---|
| Ngày | 2026-10-07 |
| Tier | `T4` (Colab Free) |
| GPU | Tesla T4, sm_75, 14.6 GB khả dụng |
| Base model | `unsloth/Qwen3.5-4B` |
| Precision | `fp16` (T4 không có bfloat16) |
| MASK_MODE | `assistant-only` |
| Epochs / max_steps | 2.0 / 30 |
| **`EVAL_LIMIT`** | *(rỗng - full set)* |
| **`n_target` / `n_regression`** | **`50` / `15`** (`smoke_mode: false`) |
| Thời gian core NB1-NB5 | 1470 s (~24.5 phút) |
| **Kết quả cổng** | **FAILED** - target delta `+0.205`, regression delta **`-0.202`** |

`results/` hiện tại là artefact của **bản v2**. Report ở HEAD là `v2_2026-10-07`.

### 🎯 Phán quyết đảo chiều so với v1 - và đây là kết quả đáng giá nhất của lab

| | v1 (smoke) | v2 (full eval) |
|---|---|---|
| `n_target` / `n_regression` | 8 / 8 | **50 / 15** |
| (c) target / regression | 0.9375 / 0.75 | 0.97 / **0.5889** |
| `target delta` | +0.250 | +0.205 |
| `regression delta` | **+0.000** | **-0.202** |
| **Phán quyết** | **PASSED** | **FAILED** |

Cùng base model, cùng corpus (sha khớp), cùng seed 42, cùng 30 step, cùng `OPTIMIZED_PROMPT`
(sha `719e74d3b6232053`). **Biến duy nhất là kích thước tập eval.**

**Vì sao lật ngược:** ở `n_regression = 8`, bước nhảy nhỏ nhất là `1/8 = 0.125` - gấp
**6.25x** ngưỡng chịu đựng `0.02`, nên cổng **không có khả năng** phát hiện vi phạm. Nó báo
`+0.000` không phải vì model không quên, mà vì 8 mẫu không đủ độ phân giải để thấy.
Trên 15 mẫu, `1/15 = 0.067` - vẫn thô hơn ngưỡng ~3.3x, nhưng đủ để thấy mất mát 3 câu.

**Nguyên nhân đã xác định:** 225 mẫu train **100% là ticket CSKH -> JSON**, trộn **0%** dữ liệu
phổ thông. Deck §6.3 đề xuất 1-5% replay data. Đây là cách sửa trực tiếp cho v3.

### Ghi chú kỹ thuật

- NB4 báo `skip attn_only / skip wrong_lr / skip qlora: adapters/<key>/ already trained` và chỉ
  train lại `correct` - ba adapter đối chứng đã có sẵn trên VM từ v1.
- Vì `report.append_row` **append chứ không upsert**, `runs.csv` có **hai dòng `correct`**
  (0.6274 của v1, 0.6255 của v2). Cả NB4 lẫn `tools/report_tables.py` đọc **dòng cuối mỗi key**,
  nên bảng report hiển thị `correct` một lần với `final_loss = 0.6255`.
- `valid_trace_rate = 0.0` vẫn là hệ quả cấu trúc (dataset không có trace), **không phải**
  reasoning-trace collapse. Xem §5.4 của report.
- **Vẫn thiếu** dự đoán per-item của baseline (b) => rubric 3.4 (>=2 ca FT *thua (b)*) chưa thoả.
  Report trình bày 6 ca fine-tune *sai* làm bằng chứng thật thay thế.

### Bằng chứng nguồn

- `docs/evidence/v2_2026-10-07_1205_cell3_pipeline.log` - toàn bộ output NB1->NB5 (372 dòng)
- `docs/evidence/v2_2026-10-07_1205_cell4_gatekeeper.log` - output `verify.py` trên Colab (92 dòng)
- `results/` - artefact thô của bản v2

---

## v3 - (dự kiến) replay data

> Chưa chạy. Mục tiêu: sửa nguyên nhân đã xác định ở v2 bằng cách trộn 1-5% dữ liệu phổ thông
> vào train, giữ nguyên mọi thứ khác.

| | |
|---|---|
| Thay đổi duy nhất | trộn 1-5% replay data vào `data/train_seed.jsonl` |
| Kỳ vọng | `regression` về gần 0.79 **mà không mất target** |
| Điều kiện hợp lệ | corpus đổi => phải cập nhật `data/checksums.json` **và** thêm `data/CUSTOM_DATASET.md`, nếu không `verify.py` báo FAIL `eval sets unmodified` |

```bash
# Trong Colab, cell 3: để EVAL_LIMIT = ""  (rỗng = full set 50/15)
python scripts/colab_run.py nb1 nb2 nb3 nb4 nb5
```

**Quan trọng:** tập **eval** phải giữ nguyên (checksum khớp) để so sánh được với v2. Chỉ đổi
tập **train**. Nếu đổi eval sau khi thấy kết quả, phép so sánh mất giá trị hoàn toàn.

