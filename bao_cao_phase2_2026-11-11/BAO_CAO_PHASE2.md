# PHASE 2 – COORDINATE DIAGNOSTICS

Ngày: **11/11/2026**, múi giờ người dùng **Asia/Saigon**. Timestamp CSV dùng UTC có offset `+00:00`.

**Kết quả nghiệm thu:** **98 PASS, 0 FAIL, 0 ERROR, 0 SKIP**: 9 test gốc, 31 test bổ sung được cập nhật thành regression, 58 test mới Phase 2. Đã tạo CSV tổng hợp và ảnh preview diagnostic. **Chưa mở camera, chưa kết nối controller/robot, chưa chạy Phase 3.**

## 1. Phạm vi và trạng thái trước sửa

Đã đọc báo cáo Phase 1, JSON bằng chứng, ba log kết quả và hai script tương ứng trong `bao_cao_offline_2026-11-11/` trước khi triển khai. Chỉ dùng mã và dữ liệu kiểm thử của workspace `hand_pose-master`; không truy cập `robotstudio_bridge` hoặc dự án khác.

Thư mục cục bộ vẫn không có `.git`; `git status --short` báo `fatal: not a git repository`. Không thể tạo Git backup/commit từ metadata cục bộ. Trước sửa đã sao lưu các tệp liên quan vào `truoc_thay_doi/` và ghi `truoc_thay_doi_sha256.json`; không tự khởi tạo Git hoặc thay thế workspace bằng bản clone. Tại thời điểm nghiệm thu, Phase 2 thực hiện cục bộ và chưa đăng các thay đổi này lên GitHub; việc xuất bản là bước riêng theo yêu cầu tiếp theo của người dùng.

`robot/HandFollow.mod` giữ nguyên SHA-256:

```text
1d55047a1288024b937950f2e5d274ce073c06a1ad3d8e01260825a98a13b583
```

Ba tệp test gốc trong `kiem_tra/` cũng giữ nguyên mã băm Phase 1. README, requirements, adapter MediaPipe và model không bị sửa. Các báo cáo, JSON và log Phase 1 giữ nguyên; chỉ script 31 test được cập nhật công khai theo hành vi mới, với bản trước sửa đã sao lưu.

## 2. Danh sách tệp thay đổi

| Tệp | Thay đổi |
|---|---|
| `run.py` | Thêm `process_wrist_frame`, điều kiện raw/filtered ROI, metadata frame, CLI diagnostic, nối logger với frame/socket, kiểm finite FPS, bảo vệ cảnh báo logging và cleanup |
| `tien_ich/theo_doi_co_tay_2d.py` | Reset EMA khi đổi kích thước; validation đầu vào/đầu ra; helper ROI trả trạng thái không hợp lệ thay vì lỗi số học |
| `tien_ich/ket_noi_yumi.py` | Target sai → HOLD; snapshot target/context; observer sau sendall; kiểm request và giới hạn buffer; giữ wire format |
| `bao_cao_offline_2026-11-11/kiem_thu_bo_sung.py` | Giữ 31 test, cập nhật tám nhóm kỳ vọng cũ thành regression; không xóa test |
| `tien_ich/chan_doan_toa_do.py` **mới** | Logger CSV bất đồng bộ và hàm vẽ panel diagnostic |
| `kiem_tra/phase2/kiem_tra_diagnostic.py` **mới** | 11 test logger/UI |
| `kiem_tra/phase2/kiem_tra_main_phase2.py` **mới** | 10 test CLI/vòng chính với toàn bộ thiết bị, model, link, cửa sổ được mock |
| `kiem_tra/phase2/kiem_tra_pipeline.py` **mới** | 8 test validation, công thức, resize và ý nghĩa bản ghi |
| `kiem_tra/phase2/kiem_tra_tcp_phase2.py` **mới** | 13 test formatter, observer, context, framing và lỗi socket bằng mock |
| `kiem_tra/phase2/kiem_tra_tich_hop_xy.py` **mới** | 16 test tích hợp từ MediaPipe mock đến TCP loopback |
| `bao_cao_phase2_2026-11-11/chay_nghiem_thu.py` **mới** | Runner có guard camera/socket, lưu kết quả, CSV và ảnh mẫu |

Tệp JSON nghiệm thu chứa SHA-256 sau sửa của mã Python/test liên quan. Các bản sao và tệp báo cáo trong thư mục này là bằng chứng, không được import vào ứng dụng.

## 3. Nhiệm vụ A – CSV diagnostic

CLI mới, mặc định đều tắt:

- `--diagnostic-log DUONG_DAN.csv`: bật CSV.
- `--diagnostic`: bật thêm cửa sổ `Coordinate Diagnostic`; độc lập với CSV.

Đã có đầy đủ 19 trường yêu cầu:

```text
timestamp, frame_id, arm,
raw_x_pixel, raw_y_pixel, filtered_x_pixel, filtered_y_pixel,
x_workspace_mm, y_workspace_mm,
raw_inside_roi, filtered_inside_roi, armed, target_valid,
x_target_mm, y_target_mm, tcp_connected,
x_sent_mm, y_sent_mm, send_timestamp
```

Thêm `event_type`, `wrist`, `sent_valid`, `robot_feedback`, `data_source`, `frame_resized` để giải thích từng bản ghi:

| Trường/loại | Ý nghĩa |
|---|---|
| `FRAME` | Quyết định của `run.py` tại frame; tọa độ target chưa làm tròn; các trường sent là `NOT_AVAILABLE` |
| `SEND` | Snapshot đúng target/context dùng để tạo thông điệp đã `sendall` thành công; tọa độ sent đọc từ chính payload đã làm tròn |
| `timestamp` | Thời điểm xử lý frame trước inference, UTC; giữ nguyên trong SEND tương ứng. SEND chưa có frame context dùng thời điểm send làm fallback |
| `frame_id` | Số frame tăng trong phiên; giúp ghép FRAME và SEND. Không tự gán lần gửi cho frame hiện tại nếu socket thực tế dùng frame cũ |
| `arm` / `wrist` | Tay robot `ROB_L/ROB_R` / nhãn cổ tay `LEFT/RIGHT`; swap chỉ đổi tay đích |
| `target_valid` | Quyết định mục tiêu được tạo ở frame |
| `sent_valid` | Valid thực có trong payload. Có thể False dù target_valid=True nếu target đã stale hoặc request sai |
| `x_sent_mm/y_sent_mm` | Trong SEND của HOLD là 0,0 theo payload; phải đọc cùng sent_valid=False, không hiểu là lệnh đi về gốc |
| `robot_feedback` | Luôn `NOT_AVAILABLE`: giao thức hiện không trả tọa độ thực hiện của robot |
| `data_source` | `CAMERA_UNVERIFIED` trong ứng dụng; `SYNTHETIC` cho CSV mẫu/tích hợp offline |

`sendall` thành công nghĩa dữ liệu đã được giao cho socket cục bộ, **không chứng minh controller nhận/parse hoặc robot đã di chuyển**. `R` chỉ là yêu cầu mục tiêu, không phải phản hồi vị trí. Trạng thái `tcp_connected` là snapshot socket, không phải heartbeat hay xác nhận robot sẵn sàng.

Logger dùng queue giới hạn 4096 bản ghi và `put_nowait`; mọi open/write/flush CSV chạy ở writer thread. Queue đầy thì bỏ bản ghi mới và tăng `dropped_rows`. Lỗi I/O ghi vào `error`, ngừng logging và không lan sang camera/TCP. Khi đóng, drain có timeout 2 giây; nếu chưa xong báo `drained=False`, không cam kết lưu được mọi hàng khi ổ đĩa treo. Có test disk lỗi/disk chậm/queue đầy; đây là kiểm tra thiết kế không chờ I/O, không phải phép đo FPS hoặc bảo đảm thời gian thực của camera.

Thông báo lỗi diagnostic chịu được console Windows không in được tiếng Việt; cleanup camera/detector/windows chạy trước thông báo cuối. Không thêm phụ thuộc ngoài các thư viện hiện có.

## 4. Nhiệm vụ B – ROI trước và sau EMA, D01

Luồng tọa độ thực trong vòng main được dùng chung với test:

```text
MediaPipe adapter → raw pixel
    → tracker.update(..., image_size=...)
    → raw_inside_roi và filtered_inside_roi tính riêng
    → target_valid = armed AND có raw/filtered hợp lệ
                     AND raw_inside_roi AND filtered_inside_roi
    → target hoặc None/HOLD → YumiArmLink
```

Đã tái hiện D01 bằng dữ liệu tổng hợp: lịch sử `(640,360)`, raw mới `(1100,360)`, alpha mặc định 0,25 ⇒ EMA `(755,360)`, workspace X≈59,895833, raw_inside=False, filtered_inside=True. **Target nay không hợp lệ và TCP trả `0,0.0,0.0\n`.** Có test cả chiều raw đã vào ROI nhưng filtered còn ngoài: vẫn HOLD.

Không đổi phương trình EMA, smoothing mặc định, công thức ROI/mapping, dấu X/Y, flip, gốc hoặc ma trận. Một số test dùng alpha=1 để kiểm riêng dấu/góc; đó chỉ là fixture, không đổi mặc định ứng dụng. Mất điểm/visibility thấp/dữ liệu lỗi xóa lịch sử nhãn; nhận lại dùng quan sát mới ngay, giữ hành vi đã được yêu cầu. Không thêm ramp hoặc thuật toán dự đoán.

## 5. Nhiệm vụ C – Đổi kích thước frame, D08

`WristXYTracker.update` nhận thêm `image_size` tùy chọn, tương thích lời gọi cũ. Với kích thước hợp lệ khác lần trước, reset lịch sử **trước** khi EMA; không dùng trực tiếp tọa độ pixel của frame cũ. Vòng main luôn truyền kích thước `frame.shape` thực.

Test 1280×720/tâm `(640,360)` → 640×360/tâm `(320,180)` cho workspace **X=0,Y=0** ngay frame đầu ở độ phân giải mới. Có test đổi qua lại nhiều lần, test pipeline, tích hợp TCP và vòng main mock. `frame_resized` ghi nhận sự kiện để chẩn đoán.

## 6. Nhiệm vụ D – Tích hợp X/Y offline

16 test tích hợp dùng `MediaPipeWristDetector.__new__` với kết quả model giả lập, nhưng chạy adapter normalized→pixel thật; sau đó chạy pipeline, tracker, mapping, target, `YumiArmLink` và `FakeRobotTask` hiện có.

Bao phủ X±, Y±, bốn góc/biên ROI, ngoài từng biên, hai điều kiện ROI độc lập, LEFT/RIGHT, swap, ARMED/HOLD, visibility thấp, mất/nhận lại, resize, stale, timeout nhận và reconnect. Kiểm chính xác chuỗi TCP để xác nhận thứ tự/dấu và làm tròn. Thêm test main CLI để xác nhận `--swap-arms` thực sự nối đúng link sau SPACE, thay vì chỉ test một hàm định tuyến riêng.

Runner chặn `cv2.VideoCapture`; các test main mock `open_camera`, constructor model/link và toàn bộ cửa sổ OpenCV. Những lời gọi main trong test không chạy chương trình điều khiển thiết bị thật. TCP thật chỉ tới literal `127.0.0.1`, cổng server giả lập do hệ điều hành cấp; cả `socket.connect`, `connect_ex`, `bind` đều được guard trong toàn bộ lượt nghiệm thu. JSON cuối lưu `network_destinations` của lượt tái tạo mẫu `--artifacts-only`, không còn là danh sách kết nối của full-suite; số kết nối ngoài loopback trong danh sách đó = **0**. Guard áp dụng cho cả full-suite và lượt tái tạo, không cho phép kết nối tới IP robot.

Các mock connect-timeout kiểm nhánh retry và tham số, không được trình bày như phép đo timeout mạng/controller. Không có test nghiệm thu controller ABB trong Phase 2.

## 7. Nhiệm vụ E – Giao diện diagnostic

Panel tùy chọn hiển thị raw pixel, EMA pixel, workspace mm, hai trạng thái ROI, ARMED/HOLD, mục tiêu TCP dự kiến, kết nối, nhãn tay và thông tin logging. Header ghi rõ **EXPECTED TARGETS, NOT ROBOT POSITION** và **ROBOT FEEDBACK: NOT_AVAILABLE**. Trường sent trên hàng FRAME vẫn NOT_AVAILABLE; không gán target dự kiến thành một lần gửi/robot đã thực hiện.

Mặc định vẫn chỉ có `Webcam` và `Wrist XY`; CSV không được tạo và panel mới không được mở. Test xác minh các cửa sổ mặc định và `--no-robot` không tạo link. Ảnh `diagnostic_SYNTHETIC.png` được vẽ trực tiếp từ CSV tổng hợp, đã kiểm tra trực quan, không chụp webcam hoặc robot. Ghi PNG dùng `imencode` + `pathlib` để tương thích đường dẫn tiếng Việt Windows; không cần mở cửa sổ.

## 8. Nhiệm vụ F – Validation và giao thức

| Phát hiện Phase 1 | Xử lý Phase 2 |
|---|---|
| D04 – phi số/sai shape tracker | Chuyển đổi có kiểm tra, bỏ observation, xóa lịch sử nhãn; không ValueError làm dừng vòng |
| D05 – formatter/worker target sai | Normalize/copy tuple hữu hạn khi publish; phi số/sai chiều dài/NaN/∞ thành HOLD, target hợp lệ kế tiếp vẫn gửi được |
| D06 – overflow sau EMA | Giữ phương trình, kiểm finite đầu ra; nếu overflow thì bỏ điểm và lịch sử, không đưa inf/NaN vào overlay |
| D07 – ROI không hợp lệ | Bounds/workspace/point không hợp lệ trả `(0,0,False)` từ helper; pipeline HOLD và workspace diagnostic `NOT_AVAILABLE`, không trình bày sentinel 0 như tọa độ hợp lệ |
| D09 – request khác R | Chỉ `R\n`/`R\r\n` hợp lệ; request khác nhận HOLD cùng định dạng; request sau hợp lệ vẫn hoạt động |
| Buffer Python không giới hạn | Giới hạn request 64 byte, quá dài đưa vào nhánh lỗi socket/retry, không tích lũy vô hạn |
| D10 – RAPID bỏ phần dư nhiều dòng | **Chưa sửa** vì cấm sửa HandFollow.mod. Đã rà soát, giữ thành giới hạn cần kiểm VC; Python vẫn xử lý request chia/gộp dòng |
| D11 – FPS NaN/∞ | Parser từ chối mọi FPS không hữu hạn trước khi mở camera/logger/link |

Wire format giữ nguyên: `valid,x,y\n`, một chữ số thập phân; HOLD `0,0.0,0.0\n`. Không thêm trường vào TCP, không thêm feedback giả. Metadata mới chỉ ở CSV và bộ nhớ Python. Callback chỉ chạy sau sendall thành công; lỗi send không tạo SEND thành công; lỗi observer không giết worker. Snapshot context/target giữ đúng frame dù có target mới publish trong lúc trao đổi TCP.

Giữ connect timeout, recv timeout và retry mặc định; recv timeout 0,5 giây vẫn chỉ tiếp tục chờ, không tự reconnect khi peer còn mở nhưng im lặng. Kích thước queue/request là giới hạn kỹ thuật phần mềm, không phải thông số hiệu chuẩn hoặc giới hạn chuyển động robot.

## 9. Kết quả test và thay đổi test cũ

Lệnh nghiệm thu đã thực thi, không chạy ứng dụng thiết bị:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
& '.venv/Scripts/python.exe' 'bao_cao_phase2_2026-11-11/chay_nghiem_thu.py'
```

| Nhóm | PASS | FAIL | ERROR | SKIP | Thời gian theo JSON runner, xấp xỉ |
|---|---:|---:|---:|---:|---:|
| 9 test gốc không sửa | 9 | 0 | 0 | 0 | 2,980 s |
| 31 test bổ sung cập nhật regression | 31 | 0 | 0 | 0 | 3,592 s |
| Test mới Phase 2 | 58 | 0 | 0 | 0 | 14,048 s |
| **Tổng test phân biệt** | **98** | **0** | **0** | **0** | **≈20,620 s** |

Môi trường thực tế: Python 3.11.9, NumPy 2.4.6, opencv-contrib-python 5.0.0.93, MediaPipe 1.2.2; đọc metadata cài đặt. Không kết luận model thật tương thích chỉ từ test dùng mock.

Tám nhóm kỳ vọng trong script 31 test được cập nhật, không âm thầm xóa test:

| Test trước | Regression hiện tại |
|---|---|
| ROI rộng 0 phải ném lỗi | Trạng thái không hợp lệ |
| NaN/∞ mapping lan NaN hoặc clamp | Không hợp lệ `(0,0,False)` |
| Phi số tracker ném ValueError | Bỏ điểm |
| Raw ngoài nhưng EMA trong | Giữ chứng minh EMA không đổi, thêm xác nhận pipeline không tạo target |
| EMA cực lớn trả nonfinite | Bỏ kết quả không hữu hạn |
| Resize còn lịch sử cũ | Truyền image_size và kiểm reset/tâm 0 |
| Malformed target ném ValueError | HOLD |
| Request NOT_R trả target | HOLD |

Lượt đầu Phase 2 có **55 PASS/1 FAIL/0 ERROR trên 56 test mới**; 9 và 31 test đều PASS. Failure là assert tuple số thực tuyệt đối: `-49.999999999999986 != -50.0`. Đã sửa test dùng `assertAlmostEqual(..., places=10)` cho target chưa làm tròn, không sửa công thức. Kiểm chuỗi TCP vẫn so sánh chính xác. Lưu traceback tại `test_phase2_lan_dau_fail.txt` và kết quả tại `ket_qua_lan_dau.json`. Sau đó thêm hai test về lỗi console/cleanup, tổng mới thành 58 test Phase 2. Không có FAIL/ERROR trong bản cuối.

Sau lần chạy toàn bộ cuối, tái tạo riêng CSV/PNG bằng `--artifacts-only` để sửa lỗi xuất ảnh của OpenCV trên đường dẫn Unicode. Chế độ này kiểm SHA-256 của toàn bộ mã/test trong bằng chứng phải còn trùng trước khi dùng lại kết quả test; không sửa production/test sau lần nghiệm thu. JSON đánh dấu `test_results_reused_after_hash_check=true`. CSV cuối hợp lệ, PNG xuất thành công, mã băm RAPID vẫn trùng.

## 10. Ví dụ CSV từ dữ liệu tổng hợp

`diagnostic_SYNTHETIC.csv` có **9 hàng dữ liệu**, 6 FRAME cho hai nhãn ở ba frame, 3 SEND của link ROB_L giả lập. Tất cả `data_source=SYNTHETIC`, `robot_feedback=NOT_AVAILABLE`. Timestamp là thời gian thực của bài kiểm thử offline, **tọa độ là dữ liệu tổng hợp**, không phải số đo người/camera/robot.

Trích các cột chính; file CSV đầy đủ có đủ 25 cột:

| Frame | Loại | Arm | Raw X px | EMA X px | Workspace X mm | Raw/EMA trong ROI | target_valid | x_target_mm | sent_valid | x_sent_mm |
|---|---|---|---:|---:|---:|---|---|---|---|---|
| 1 | FRAME | ROB_L | 755 | 755 | 59,895833… | True/True | True | 59,895833… | NOT_AVAILABLE | NOT_AVAILABLE |
| 1 | SEND | ROB_L | 755 | 755 | 59,895833… | True/True | True | 59,895833… | True | 59,9 |
| 2 | FRAME | ROB_L | 1100 | 841,25 | 104,817708… | False/True | False | NOT_AVAILABLE | NOT_AVAILABLE | NOT_AVAILABLE |
| 2 | SEND | ROB_L | 1100 | 841,25 | 104,817708… | False/True | False | NOT_AVAILABLE | False | 0,0 |
| 3 | SEND | ROB_L | NOT_AVAILABLE | NOT_AVAILABLE | NOT_AVAILABLE | False/False | False | NOT_AVAILABLE | False | 0,0 |

Đây là ví dụ bắt đầu EMA tại x=755, khác fixture D01 bắt đầu tại640. TCP giả lập đã nhận lần lượt `1,59.9,0.0\n`, `0,0.0,0.0\n`, `0,0.0,0.0\n`. Không có tọa độ phản hồi robot. Một FRAME có thể có tcp_connected=False rồi SEND tương ứng True vì snapshot kết nối thay đổi giữa hai thời điểm; không suy diễn đó là dữ liệu robot đã thực hiện.

## 11. Những điểm còn tồn tại và mức sẵn sàng VC

**Sẵn sàng về phần kiểm tra Python/diagnostic để chuẩn bị một đợt kiểm thử VC có kiểm soát. Chưa xác minh hoạt động với Virtual Controller hoặc robot.** Phase 2 dừng ở đây, không mở RobotStudio/VC và không thực hiện Phase 3.

Các giới hạn còn lại:

- RAPID `ReadLine`/`ParseTarget`, giới hạn string/số, lỗi parser, timeout từng chunk và phần dư nhiều dòng chưa được kiểm bằng môi trường ABB; D10 còn nguyên theo phạm vi cấm sửa RAPID.
- HOLD chỉ không phát mục tiêu MoveL mới; chưa có cơ chế hủy chuyển động đã nhận hoặc xác nhận dừng. Không dùng diagnostic/CSV để chứng nhận dừng tức thì.
- Chưa hiệu chuẩn camera, đo sai số X/Y thật, FPS, độ trễ, độ lặp lại, quỹ đạo, va chạm hoặc khả năng tới đích.
- Chưa chạy model MediaPipe thật hoặc kiểm nhãn giải phẫu trên ảnh phản chiếu; test model là mock. Không thay nhãn/dấu để đoán kết quả trên người thật.
- Workspace CLI vẫn có thể khác giới hạn RAPID cố định ±200/±150; chưa thay vì không được tự chọn thông số controller. Cần đối chiếu cấu hình thực trước vận hành.
- README gốc còn sai khác `--mm-per-pixel` và tên WorkObject phải đã nêu trong Phase 1; Phase 2 không sửa nội dung đó. Tài liệu CLI diagnostic mới là mục 3 và runner ở báo cáo này.
- Logger có thể mất hàng khi queue đầy, I/O lỗi hoặc drain timeout; có counter/trạng thái để nhận biết. Không cam kết log bền vững khi mất điện hoặc disk treo.
- Ngay cả target hữu hạn nhưng rất lớn từ caller trực tiếp vẫn chưa chứng minh nằm trong miền số/string ABB; các mục tiêu sản xuất còn phụ thuộc workspace cấu hình và clamp RAPID. Không tự đặt thêm giới hạn vật lý.

Những thông số/dữ liệu bắt buộc lấy từ controller hoặc VC tương ứng cho giai đoạn sau:

| Nhóm | Dữ liệu cần xác nhận |
|---|---|
| Hệ thống ABB | Phiên bản RobotWare, model/controller, option socket/PC Interface và khả năng load/compile module |
| Task/socket | Tên task thực, IP bind, cổng trái/phải và quy tắc mạng; không suy ra hoạt động từ IP mặc định trong mã |
| Tool | Tool đang được `CTool()` chọn, tooldata/TCP/load, trạng thái hiệu chuẩn |
| WorkObject | Tồn tại và giá trị đầy đủ `wobjPlaneXY`/`wobjPlaneXY1`, user/object frames, hướng trục và quan hệ với base/tool |
| Tâm | Kết quả `CRobT` trong đúng Tool/WObj khi khởi động, START_IS_CENTER, Z/orientation/config/extaxes; tác động khi reconnect |
| Chuyển động | Giới hạn offset được duyệt, vùng làm việc thực, speed/zone/deadband, cấu hình khớp, singularity và không gian hai tay |
| Phản hồi và dừng | Cách đo TCP/pose thực và timestamp; hành vi khi mất tay/mạng/timeout và khi MoveL đang chạy |

Các dữ liệu này chưa có trong bài kiểm tra, không tự tạo giá trị thay thế.

## 12. Bằng chứng và kết luận

- `test_goc_9.txt`, `test_bo_sung_31.txt`, `test_phase2.txt`: log bản cuối.
- `ket_qua_nghiem_thu.json`: kết quả, môi trường, đích TCP, kiểm CSV/PNG và SHA-256 sau sửa.
- `test_phase2_lan_dau_fail.txt`, `ket_qua_lan_dau.json`: failure ban đầu và traceback giữ lại.
- `diagnostic_SYNTHETIC.csv`, `diagnostic_SYNTHETIC.png`: dữ liệu/preview tổng hợp.
- `truoc_thay_doi/`, `truoc_thay_doi_sha256.json`: bản sao và mã băm trước triển khai.

Phase 2 đã bổ sung diagnostic CSV/UI, sửa điều kiện ROI D01, reset theo độ phân giải D08 và bảo vệ dữ liệu/TCP Python theo yêu cầu. Công thức, dấu/gốc tọa độ và smoothing mặc định giữ nguyên. **98 test PASS xác nhận phạm vi offline đã kiểm; không phải chứng nhận độ chính xác tọa độ hoặc an toàn chuyển động thực tế. RAPID không đổi, camera/controller chưa được kích hoạt, Phase 3 chưa triển khai.**
