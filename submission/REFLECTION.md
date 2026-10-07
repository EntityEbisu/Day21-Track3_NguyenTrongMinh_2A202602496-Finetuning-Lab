# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

> Phiên bản `v2_2026-10-07` (iteration 2, full eval). Xem `docs/RUNS.md`.

**1. Điều gì làm bạn ngạc nhiên nhất?**

Điều làm tôi ngạc nhiên nhất không phải một con số — mà là **kết luận của chính tôi bị lật
ngược khi tôi đo trên nhiều mẫu hơn.**

Ở iteration 1 (smoke, `n_target=8`), cổng hồi quy báo **PASSED**: target +0.250, regression
không đổi (`+0.000`). Tôi đã viết xong một report nói "thắng lợi rõ, nhưng cần full eval để
chắc". Ở iteration 2 (full eval, `n_target=50`, `n_regression=15`) — **cùng base model, cùng
seed 42, cùng 30 step, cùng corpus, cùng prompt** — cổng báo **FAILED**: `regression Δ =
−0.202`, gấp **10.1×** ngưỡng chịu đựng `0.02`. Bản fine-tune đạt target 0.97 (rất tốt) nhưng
**quên mất 3/15 câu hỏi kiến thức phổ thông**.

Điều duy nhất tôi thay đổi giữa hai lần chạy là `EVAL_LIMIT`. Không có biến gây nhiễu nào:
corpus sha khớp (`verify.py` xác nhận `eval sets unmodified`), prompt (b) sha khớp
(`719e74d3b6232053`), seed và số step khớp. Vậy mà kết luận đảo chiều hoàn toàn.

Tôi đã suýt nộp bản iteration 1 và gọi nó là thành công. Cái ngăn tôi lại là một dòng trong
`baselines_frozen.json`: `"smoke_mode": true`.

Ngạc nhiên thứ hai, cụ thể hơn: **cả 6 ca fine-tune sai đều sai đúng một trường `urgency`,
đều có nhãn đúng là `thap`, đều bị đoán thành `trung_binh`, và cả 6 đều chứa cụm "khi nào
tiện"**. Kiểm tra ngược: đúng 6 câu trong tập eval chứa cụm đó, và model sai **6/6**. Đó không
phải nhiễu — đó là một lỗ hổng hệ thống với 6 bằng chứng độc lập.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Ở iteration 1: NB4 mất **1289 s** cho 3 run đối chứng, chiếm 57% của 2261 s core. Đúng như dự
đoán từ `HARDWARE-GUIDE.md`.

Ở iteration 2, bất ngờ đảo lại: **NB5 mất 644 s** (10.7 phút) — nhiều nhất, vì nó phải sinh
văn bản cho **4 adapter × 50 mẫu target** (cộng 3 lượt regression), chứ không chỉ 8 mẫu như
iteration 1. NB4 chỉ mất **50 s** vì ba adapter đối chứng đã có sẵn trên VM và bị bỏ qua
(`skip attn_only: adapters/attn_only/ already trained`). Tổng core giảm từ 2261 s xuống
**1470 s**.

Chỗ tôi **không** dự đoán được, và là chỗ tốn thời gian thật: **không phải GPU, mà là hiểu ra
rằng cỡ mẫu của tập eval quyết định kết luận.** Tôi đã tiêu phần lớn thời gian không phải để
train, mà để trả lời câu hỏi "tại sao cùng một model lại cho hai phán quyết ngược nhau?".
Câu trả lời nằm ở số học: ở `n_regression=8`, bước nhảy nhỏ nhất là `1/8 = 0.125`, gấp
**6.25×** ngưỡng `0.02` — nên cổng **không có khả năng** phát hiện vi phạm. Nó báo `+0.000`
không phải vì model không quên, mà vì 8 mẫu không đủ độ phân giải để thấy.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

**Tôi từng tin rằng một cổng kiểm tra báo "PASSED" nghĩa là kết quả đã đúng.** Giờ tôi biết
một cổng chỉ đúng trong phạm vi độ phân giải của nó. Cùng một cổng, cùng ngưỡng `0.02`, cùng
model — chỉ khác số mẫu — và nó cho hai câu trả lời trái ngược. Từ giờ trước khi tin một cổng,
tôi hỏi: *cỡ mẫu này có đủ để cổng báo động không?* Nếu không, "PASSED" chỉ có nghĩa là
"chưa phát hiện được vấn đề", không phải "không có vấn đề".

**Tôi từng tin rank là nút vặn chính** — rằng "quét r=8/16/64" là thí nghiệm trung tâm của
fine-tuning LoRA. Giờ tôi không còn tin: nâng rank từ **16 lên 283** (gấp 17.7×, để bù đúng
ngân sách tham số) **không tạo ra khác biệt target nào đo được** — 194/200 trường đúng cho cả
hai. Còn đổi **learning rate một thang 10×** thì target sụp xuống **0.000** hoàn toàn.

**Tôi từng tin `final_loss` là chỉ số để chọn cấu hình tốt nhất.** Giờ tôi biết nó xếp hạng
đúng **1 trong 4 run**, và sai ở đúng vị trí đầu tiên: nó nói `attn_only` (0.538) tốt nhất,
trên `correct` (0.6255) — trong khi trên tập target hai run **hoà**.

Và điều tôi không còn tin nữa, quan trọng nhất: **"fine-tune thắng trên target" là một kết
luận đủ để deploy.** Bản của tôi thắng target rất rõ (+0.205, 194/200) và vẫn **FAILED**.
Target cao không bù được việc quên những thứ khác.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Tôi dùng AI assistant để: đọc hiểu codebase `labkit` (9 module — tôi cần biết `matched_rank()`
làm gì và tại sao mask xây bằng offset ký tự chứ không phải diff token), dựng
`tools/report_tables.py` để sinh bảng số từ `results/*.json` thay vì chép tay ~60 con số, truy
vết bẫy CRLF làm `verify.py` FAIL giả trên Windows, và phát hiện một bug thật: `runs.csv` có
**hai dòng `correct`** vì iteration 2 train lại `correct` trong khi NB4 bỏ qua ba adapter đối
chứng đã có sẵn — nên bảng report phải lấy **dòng cuối mỗi key**, đúng như NB4 tự làm.

**Chỗ nó sai, cụ thể:**

1. **Nó không phát hiện `EVAL_LIMIT=8` ngay từ đầu.** Assistant đã đọc
   `baselines_frozen.json` nhưng chỉ tóm tắt "cổng PASSED, target +0.250" — bỏ qua dòng
   `smoke_mode: true`, tức dòng quyết định xem kết quả có nộp được hay không. Tôi phải tự
   đọc file mới thấy. Một bản tóm tắt bỏ sót đúng thứ quan trọng nhất còn tệ hơn không tóm tắt.

2. **Nó đề xuất một cách version hoá sai.** Ban đầu nó định đổi tên file thành
   `REPORT_v1_<timestamp>.md` — nhưng `scripts/verify.py` hard-code `submission/REPORT.md`
   trong `REQUIRED_ARTIFACTS`, nên đổi tên sẽ làm gatekeeper báo thiếu artefact. Chính tôi
   phải chỉ ra điều đó. **Bài học: kiểm tra ràng buộc của hệ thống trước khi tin một đề xuất
   nghe hợp lý.**

3. **Nó suýt viết một heredoc sai.** Bản nháp `docs/RUNS.md` dùng `<<'EOF'` (cần thiết để
   backtick trong Markdown không bị shell diễn giải) nhưng lại đặt `${TS}` **bên trong** khối
   đó — nên timestamp sẽ không được thay và file ghi literal `<TS>`. Phải sửa bằng `printf`.

4. **Nó suýt ghi một con số sai vào report.** Ở iteration 1, nó viết tỉ lệ tốc độ (b)/(a) là
   "~3.4×"; số thật là 3374.2/936.9 = **3.6×**. Tôi bắt được vì tôi tự chia lại. Từ đó tôi
   kiểm tra mọi tỉ lệ và mọi chênh lệch bằng một lệnh chạy lại được, không tin con số đọc sẵn.

Tôi kiểm tra chéo mọi con số assistant đưa bằng `sha256sum`, `git diff`, và đọc `results/*.json`
trực tiếp. Không con số nào trong report này mà tôi không tự xác minh được.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

**Chốt cỡ mẫu của tập đánh giá trước, rồi mới đo baseline.** Không phải bước thứ hai — bước
thứ nhất.

Lý do đến thẳng từ thất bại của chính tôi: tôi đã có một phán quyết PASSED sai, và nó sai
không phải vì tôi cấu hình sai mà vì **tập eval quá nhỏ để cổng có thể báo động**. Nếu tôi
mang 8 mẫu đó đến một khách hàng thật, tôi đã ký một quyết định triển khai dựa trên 2 mẫu.

Cụ thể tôi sẽ làm, theo thứ tự:

1. **Chốt cỡ mẫu eval trước khi train**, và kiểm tra nó có đủ độ phân giải không: nếu ngưỡng
   chấp nhận là `T`, tôi cần `1/n` nhỏ hơn `T` một cách thoải mái. Ở đây `T = 0.02` cần
   `n ≥ 50`; tập regression 15 mẫu vẫn còn quá thô (`1/15 = 0.067`, gấp 3.3× ngưỡng).
2. **Đo baseline đã được prompt tử tế.** Trong lab này (b) đạt 0.765 so với (a) 0.000, và (b)
   còn **rẻ hơn 3.14×** (1043 ms vs 3274 ms). Phần lớn giá trị nằm ở việc viết prompt đúng,
   không ở chỗ train. Nếu prompt tốt đã đủ cho khách hàng → dừng.
3. **Trộn replay data trước khi train.** Bài học trực tiếp từ `regression Δ = −0.202`: 225 mẫu
   train 100% là ticket CSKH, trộn **0%** dữ liệu phổ thông, và model quên mất 3/15 câu kiến
   thức. Deck §6.3 đề xuất 1–5%; con số 0 của tôi là sai lầm cụ thể và đã đo được hậu quả.
4. **Đóng băng tập eval trước khi train**, vì sau khi thấy kết quả thì mọi sửa đổi eval đều là
   tự lừa mình.
5. Và **đọc cột theo trường, không chỉ điểm tổng.** 6 ca sai của tôi đều tập trung ở `urgency`
   với cùng một marker bị bỏ sót — một lỗi định vị được, sửa bằng dữ liệu cho một trường, rẻ
   hơn nhiều so với train lại toàn bộ.
