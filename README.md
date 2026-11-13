# Điều khiển YuMi bằng cổ tay và cử chỉ bàn tay

## Chạy

```powershell
.\.venv-gripper\Scripts\python.exe run.py
```

Chương trình mở webcam, kết nối robot tại `192.168.125.1:1025` (ROB_L) và `:1026` (ROB_R), bắt đầu ở HOLD. Nhấn `SPACE` để bật/dừng điều khiển; `q` hoặc `Esc` để thoát.

```powershell
.\.venv-gripper\Scripts\python.exe run.py --help
```

## Cấu trúc chương trình

`run.py` chỉ xử lý tham số, khởi tạo và điều phối vòng lặp. Các chức năng dùng chung nằm trong `tien_ich`:

| File | Chức năng |
|---|---|
| `nhan_dien_ban_tay.py` | Mở camera, nhận landmark/cử chỉ bàn tay, khởi tạo MediaPipe dùng chung |
| `nhan_dien_co_tay.py` | Nhận cổ tay trái/phải bằng MediaPipe Pose |
| `theo_doi_co_tay_2d.py` | Lọc EMA, ánh xạ mm, vùng WORKSPACE, ngưỡng chuyển động và tạo target |
| `dieu_khien_gripper.py` | Phân loại mở/nắm, ghép đúng cổ tay và xác nhận cử chỉ ổn định |
| `ket_noi_yumi.py` | Kết nối TCP từng tay robot, bản tin mới nhất, hết hạn target và timestamp |
| `chan_doan_toa_do.py` | Hiển thị webcam/dashboard, trạng thái điều khiển và ghi CSV chẩn đoán tùy chọn |

`mo_hinh` chứa các model đang được hỗ trợ; `robot/HandFollow.mod` là chương trình RAPID cho hai tay YuMi.

## Chuyển động cổ tay

- WORKSPACE mặc định `400 × 300 mm`, nằm giữa ảnh; X dương sang phải, Y dương lên trên.
- Sau lọc EMA (`0.25`), so sánh vị trí hiện tại với điểm đã chấp nhận gần nhất.
- Chỉ khi `abs(delta_x) > 10` **hoặc** `abs(delta_y) > 10 mm`, cập nhật cả X và Y và hiển thị `MOVE`. Đúng 10 mm hoặc nhỏ hơn giữ điểm, hiển thị `HOLD`.
- Điểm mới trở thành mốc tiếp theo. Hai tay có mốc độc lập; mất điểm/ra ngoài WORKSPACE/đổi kích thước ánh xạ sẽ đặt lại mốc.
- `INIT` đánh dấu lần đặt mốc, `NO POINT` là không có quan sát hợp lệ. Đây là trạng thái tọa độ, độc lập với ARMED/HOLD cho phép robot chạy.

Mm là đơn vị quy đổi theo WORKSPACE, cần hiệu chuẩn riêng để đo dịch chuyển vật lý.

```powershell
.\.venv-gripper\Scripts\python.exe run.py --motion-threshold-mm 10 --work-width-mm 400 --work-height-mm 300
```

## Gripper

| Cử chỉ | Lệnh |
|---|---|
| Mở bàn tay | OPEN (`1`) |
| Nắm bàn tay | CLOSE (`0`) |
| Chưa rõ, mất tay hoặc chưa cho phép điều khiển | HOLD (`-1`), không phát lệnh mới |

Tay trái điều khiển ROB_L, tay phải điều khiển ROB_R. `--swap-arms` đảo cả XY và gripper. Gripper vẫn hoạt động khi keypoint cổ tay đang giữ nguyên dưới ngưỡng chuyển động.

Mặc định kết hợp model Gesture Recognizer với hình học ngón tay. Model và hình học mâu thuẫn hoặc cử chỉ khác như chỉ ngón/Victory thì giữ UNKNOWN. Cử chỉ cần ổn định ít nhất `250 ms`, bàn tay/cổ tay trong WORKSPACE và đủ confidence/visibility. Khi hai tay ghép mơ hồ, chương trình không phát lệnh gripper.

```powershell
.\.venv-gripper\Scripts\python.exe run.py --gripper-hand right --gripper-stable-ms 250 --gripper-confidence 0.7
.\.venv-gripper\Scripts\python.exe run.py --no-gripper
```

Model mặc định: `pose_landmarker_lite.task` và `gesture_recognizer.task` trong `mo_hinh`. Chọn model bằng `--pose-model`, `--gesture-model`; `--gesture-model=` dùng riêng hình học với `hand_landmarker.task` hoặc `--hand-model`. `pose_landmarker_full.task` được giữ để có thể chọn thay model lite. Khởi tạo MediaPipe dùng model buffer và thư mục ASCII để hỗ trợ đường dẫn tiếng Việt trên Windows.

## Robot và chẩn đoán

Nạp `robot/HandFollow.mod` vào cả `T_ROB_L` và `T_ROB_R`. Cần PC Interface cho socket và SmartGripper add-in đã khởi tạo/hiệu chuẩn. Work object mặc định: `wobjPlaneXY` ở trái, `wobjPlaneXY1` ở phải. Chỉnh `GRIP_OPEN_MM` (24 mm) và `GRIP_FORCE_N` (5 N) theo bộ kẹp/vật gắp, kiểm tra chương trình và vị trí tâm trong RobotStudio/controller trước khi chạy.

Giao thức yêu cầu `R,G` nhận `valid,x_mm,y_mm,grip`; client vẫn hỗ trợ bản tin ba trường của RAPID cũ yêu cầu `R`. Target quá 300 ms chuyển HOLD. HOLD không tự nhả vật và không hủy lệnh robot đã nhận; TCP gửi thành công không xác nhận gripper đã hoàn thành.

```powershell
.\.venv-gripper\Scripts\python.exe run.py --diagnostic --diagnostic-log logs/coordinates.csv
```

Chẩn đoán hiển thị riêng tọa độ thô, EMA, điểm chấp nhận, delta và target. CSV ghi sự kiện FRAME/SEND, nhãn/điểm model; `robot_feedback` luôn là `NOT_AVAILABLE`. Dùng tên CSV mới khi cấu trúc cột thay đổi.

Các gói cần thiết nằm trong `requirements.txt`; môi trường đang dùng là `.venv-gripper`.
