# Trạng thái các bản chạy

> Ghi lại **chính xác** bản chạy nào đã tạo ra `results/` trong repo, để không ai —
> kể cả tác giả — đọc nhầm một bản smoke là kết quả cuối.
>
> Version của report sống ở **ba chỗ**, không ở tên file (vì `scripts/verify.py`
> hard-code `submission/REPORT.md` trong `REQUIRED_ARTIFACTS`):
> header của `submission/REPORT.md`, commit message, và file này.

## v1 — 2026-10-07 (smoke) · "iteration 1"

| | |
|---|---|
| Ngày | 2026-10-07 |
| Tier | `T4` (Colab Free) |
| GPU | Tesla T4, sm_75, 14.6 GB khả dụng |
| Base model | `unsloth/Qwen3.5-4B` |
| Precision | `fp16` (T4 không có bfloat16) |
| MASK_MODE | `assistant-only` |
| Epochs / max_steps | 2.0 / 30 |
| **`EVAL_LIMIT`** | **`8` — SMOKE MODE** |
| **`n_target` / `n_regression`** | **`8` / `8`** (tập đầy đủ: 50 / 15) |
| Thời gian core NB1–NB5 | 2261 s (~37.7 phút) |
| Kết quả cổng | PASSED — target Δ `+0.250`, regression Δ `+0.000` |

## Cảnh báo — bản này KHÔNG phải bản nộp cuối

1. **Smoke mode.** `results/baselines_frozen.json` ghi `"smoke_mode": true`,
   `"eval_limit": 8`. Mọi số target / regression / latency tính trên **8 mẫu**, nên
   **1 mẫu = 0.125** — thô gấp ~6 lần tập 50 mẫu (1 mẫu = 0.02).
   `scripts/verify.py` báo `[ FAIL ] full eval set used` đúng vì lý do này.
2. **Thiếu dự đoán per-item của baseline (b).** Bundle không chứa chúng, nên mục §6 của
   `submission/REPORT.md` không có cột "(b) prompt" và không xác định được ca "fine-tune
   thua" mà rubric 3.4 yêu cầu ≥2.
3. **Chỉ có adapter `correct` (và chỉ `adapter_config.json`, không có trọng số).**
   `attn_only`, `wrong_lr`, `qlora` không được export khỏi VM. Số liệu train của chúng nằm
   trong `results/runs.csv` và `results/autopsy.json`, nhưng trọng số thì không còn ⇒ bonus
   B1 (merge + hot-swap) chưa làm được ở v1.
4. **`valid_trace_rate = 0.0`** là hệ quả cấu trúc của dữ liệu (250 câu trả lời là JSON
   trần, không có khối `think`), **không phải** reasoning-trace collapse. Xem §5 của report.

## Cách chạy lại cho bản nộp cuối

```bash
# Trong Colab, cell 3: để EVAL_LIMIT = ""  (rỗng = full set 50/15)
python scripts/colab_run.py nb1 nb2 nb3 nb4 nb5
```

`verify.py` chỉ xanh khi `EVAL_LIMIT` rỗng. Chọn `""` trong dropdown của cell 3 —
**không** chọn 25: mọi giá trị số đều đặt `smoke_mode = bool(EVAL_LIMIT) = true`.

## Bằng chứng nguồn

- `docs/evidence/v1_2026-10-07_1145_cell3_pipeline.log` — toàn bộ output NB1→NB5 (421 dòng)
- `docs/evidence/v1_2026-10-07_1145_cell4_gatekeeper.log` — output `verify.py` trên Colab (90 dòng)
- `results/` — artefact thô của bản chạy này

---

## v2 — (dự kiến) full eval

> Mục này chưa có số liệu. Điền sau khi chạy xong.

| | |
|---|---|
| `EVAL_LIMIT` | *(rỗng — full set)* |
| `n_target` / `n_regression` | *50 / 15 (dự kiến)* |
| Trạng thái | *(chờ chạy)* |

Khi chạy xong, thay mục này bằng số thật và cập nhật dòng `**Phiên bản**` trong
`submission/REPORT.md` thành `v2_<ngày>`, đồng thời viết lại các bảng số trong report
bằng `python tools/report_tables.py`.

```bash
# Trong Colab, cell 3: để EVAL_LIMIT = ""  (rỗng = full set 50/15)
python scripts/colab_run.py nb1 nb2 nb3 nb4 nb5
```
