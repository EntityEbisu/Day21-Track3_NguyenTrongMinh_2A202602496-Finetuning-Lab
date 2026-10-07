# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**

Hai điều, cả hai đều ngược với kỳ vọng của tôi.

Thứ nhất: **`attn_only` hoà với `correct` trên tập target** (cùng 0.9375), trong khi deck §11.2
gọi attention-only là "Lỗi #1". Tôi đã chuẩn bị sẵn tâm thế viết "đúng như deck dự đoán, nó
thua" — và số đo không cho tôi viết câu đó. Điều làm tôi ngạc nhiên hơn nữa là **`attn_only`
thắng trên `final_loss`** (0.538 < 0.6274) và **nhanh hơn 1.8×** khi suy luận. Tức là nếu tôi
chấm bằng cột loss, tôi sẽ kết luận ngược hoàn toàn: "chỉ gắn q,v là đủ, còn gắn 12 module là
lãng phí". Kết luận đó có thể đúng trên tác vụ hẹp này — nhưng tôi chỉ biết nó *không sai* nhờ
đọc cột target, không nhờ cột loss.

Thứ hai: **`masked-think` và `response-only` là no-op trên corpus này.** Tôi đọc bảng bốn
`MASK_MODE` trong NB1 và mặc định rằng chọn chế độ "nghiêm ngặt hơn" là an toàn hơn. Hoá ra
chat template của Qwen3.5 đóng khối `think` rỗng **bên trong generation prompt**, nên không
còn gì để bỏ qua và cả ba chế độ sinh ra mask y hệt. Một cờ trông như đang bảo vệ tôi lại
không làm gì cả — và chỉ `labkit` cảnh báo `no-op on this corpus`. Đây đúng là loại thất bại
im lặng mà cả lab này dạy tôi đi tìm, và tôi vẫn suýt bỏ qua nó.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

NB4 — **1289 s cho 3 run đối chứng**, so với 450 s của NB3 (1 run) và 255 s của NB5. Tổng cả
core là 2261 s (~37.7 phút), và NB4 chiếm **57%** thời gian đó. Đúng như tôi dự đoán: đọc
trước `HARDWARE-GUIDE.md` cho biết NB4 mất 45–60 phút, tôi đã tính toán và nó là notebook dài
nhất.

Chỗ tôi **không** dự đoán được là **NB2 mất 245 s trong đó 139 s chỉ để tải 9.32 GB trọng số**
— tức 57% thời gian của NB2 là tải file, không phải tính toán. Và bất ngờ hơn: `wrong_lr` lại
là run **chậm thứ hai** (396.9 s) dù nó học ít nhất, còn `qlora` chậm nhất (468.7 s) vì phải
lượng tử hoá/giải lượng tử hoá. Tôi tưởng run "học kém" thì phải nhanh hơn. Không: thời gian
train phụ thuộc vào chi phí mỗi bước, không phụ thuộc vào việc model có học được gì hay không.

Điều tôi thực sự mất thời gian mà không tính trước: **quên rằng `EVAL_LIMIT` có mặc định là
`8` trong notebook**. Tôi chạy xong toàn bộ pipeline mới phát hiện `smoke_mode: true` và
`verify.py` FAIL. Bài học: đọc giá trị mặc định của `# @param`, đừng chỉ đọc chú thích cạnh nó.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Tôi từng tin **rank là nút vặn chính** — rằng "quét r=8/16/64" là thí nghiệm trung tâm của
fine-tuning LoRA, và chọn rank đúng là quyết định lớn nhất. Lab cũ (theo `notebooks/04` mô tả)
đúng là lấy đó làm thí nghiệm trung tâm.

Giờ tôi không còn tin. Đo được: nâng rank từ **16 lên 283** (gấp 17.7×, để bù đúng ngân sách
tham số) **không tạo ra khác biệt target nào đo được** — hoà với `correct`. Còn đổi
**learning rate một thang 10×** (1e-4 → 1e-5) thì target sụp từ 0.9375 xuống **0.000**, format
về 0.000, latency tệ nhất. Deck §11 nói rank là "năng lực so với lượng thông tin trong dữ
liệu" — với 250 mẫu, rõ ràng không có đủ thông tin để r=64 (chứ đừng nói r=283) dùng hết.

Tôi cũng từng tin **`final_loss` là chỉ số để chọn cấu hình tốt nhất** — nó rẻ, có sẵn, và
"loss thấp hơn = tốt hơn" nghe hiển nhiên. Giờ tôi biết nó xếp hạng đúng **1 trong 4 run**,
và sai ở đúng vị trí đầu tiên. Loss thấp có thể chỉ là ghi nhớ.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Tôi dùng AI assistant để: đọc hiểu codebase `labkit` (9 module, tôi cần biết `matched_rank()`
làm gì và tại sao mask được xây bằng offset ký tự chứ không phải diff token), dựng
`tools/report_tables.py` để sinh bảng số từ `results/*.json` thay vì chép tay ~60 con số, và
truy vết bẫy CRLF làm `verify.py` FAIL giả trên Windows.

**Chỗ nó sai, cụ thể:**

1. **Nó không phát hiện `EVAL_LIMIT=8` ngay từ đầu.** Tôi phải tự đọc
   `baselines_frozen.json` rồi mới thấy `smoke_mode: true`. Assistant đã đọc file đó nhưng chỉ
   tóm tắt "cổng PASSED, target +0.250" — bỏ qua dòng `smoke_mode`, tức là dòng quyết định
   xem kết quả có nộp được hay không. Một bản tóm tắt bỏ sót đúng thứ quan trọng nhất còn tệ
   hơn không tóm tắt.

2. **Nó đề xuất một cách version hoá sai.** Ban đầu nó định đổi tên file thành
   `REPORT_v1_<timestamp>.md` — nhưng `scripts/verify.py` hard-code `submission/REPORT.md`
   trong `REQUIRED_ARTIFACTS`, nên đổi tên sẽ làm gatekeeper báo thiếu artefact. Chính tôi
   phải chỉ ra điều đó. Sau khi kiểm tra thì assistant đồng ý và chuyển sang ghi version ở
   commit message + header. **Đây là bài học tôi giữ lại: kiểm tra ràng buộc của hệ thống
   trước khi tin một đề xuất nghe hợp lý.**

3. **Nó suýt viết một heredoc sai.** Bản nháp `docs/RUNS.md` dùng `<<'EOF'` (cần thiết để
   backtick trong Markdown không bị shell diễn giải) nhưng lại đặt `${TS}` **bên trong** khối
   đó — nên timestamp sẽ không được thay và file sẽ ghi literal `<TS>`. Phải sửa bằng `printf`.

Tôi kiểm tra chéo mọi con số assistant đưa bằng cách tự chạy `sha256sum`, `git diff`, và đọc
`results/*.json` trực tiếp. Không có con số nào trong report này mà tôi không tự xác minh được
bằng một lệnh chạy lại được.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

**Đo baseline đã được prompt tử tế trước khi cam kết fine-tune.** Không phải bước thứ hai —
bước thứ nhất.

Lý do đến thẳng từ số đo của lab này: baseline (b) đạt **0.6875** trong khi baseline (a) đạt
**0.000**, và (b) còn **rẻ hơn 3.6×** (936.9 ms vs 3374.2 ms). Nghĩa là phần lớn giá trị nằm ở
chỗ viết prompt đúng, không ở chỗ train. Nếu tôi bắt đầu bằng fine-tune, tôi đã đốt GPU và
thời gian để đạt được thứ mà một prompt tốt đã cho sẵn.

Cụ thể tôi sẽ làm, theo thứ tự:

1. **Viết prompt tối ưu với schema + ví dụ, đo trên tập eval thật** (không phải tập tự chọn).
   Nếu nó đã đủ tốt cho khách hàng → dừng. Đây là kết luận hợp lệ và tiết kiệm nhất.
2. Nếu chưa đủ, **đo tại sao** — tách lỗi theo từng trường như `triage_field_accuracy` làm.
   Trong lab này, 2 ca yếu nhất đều sai đúng một trường (`urgency`: `thap` → `trung_binh`,
   bỏ sót marker "khi nào tiện"). Nếu lỗi tập trung ở một trường, tăng dữ liệu cho trường đó
   rẻ hơn nhiều so với train lại.
3. Chỉ khi đó mới fine-tune — và **đóng băng tập eval trước khi train**, vì sau khi thấy kết
   quả thì mọi sửa đổi eval đều là tự lừa mình.
4. Và **luôn chạy eval đủ lớn**. Bài học đắt nhất của lab này: kết quả trên 8 mẫu (1 mẫu =
   0.125) trông rất đẹp — +0.250, PASSED — nhưng thực chất chỉ là **2 mẫu trong 8**. Một
   khách hàng thật không nên nhận một quyết định triển khai dựa trên 2 mẫu.
