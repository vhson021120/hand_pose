# MediaPipe Wrist XY Tracker

Ứng dụng webcam chỉ theo dõi cổ tay bằng MediaPipe Pose và các điểm bàn tay bằng MediaPipe Hand Landmarker. Không chạy KAPAO, IK, mô phỏng robot hoặc dựng mô hình 3D.

## Hiển thị

- Cửa sổ `Webcam` vẽ hai điểm cổ tay khi MediaPipe Pose xác nhận đủ visibility và vẽ 21 landmark của mỗi bàn tay được phát hiện.
- Cửa sổ `Wrist XY` đặt gốc `O` tại tâm ảnh. Trục X dương hướng sang phải; Y dương hướng lên trên.
- Tọa độ cổ tay được quy đổi theo `mm = (pixel - tâm ảnh) x mm-per-pixel`; khi thiếu phát hiện, điểm bị bỏ ngay, không giữ hoặc dự đoán tọa độ.
- Bộ lọc EMA mặc định `0.25` chỉ làm mượt các quan sát liên tiếp; khi mất điểm, trạng thái bộ lọc được xóa.

`--mm-per-pixel` mặc định là `0.35`, đây là scale hiển thị quy ước, không phải hiệu chuẩn kích thước vật lý. Muốn đo mm thực, cần hiệu chuẩn theo vật chuẩn trong cùng mặt phẳng với tay và nhập hệ số tương ứng.

## Chạy

```powershell
python run.py
```

Ví dụ chọn camera, độ phân giải và vùng làm việc:

```powershell
python run.py --pipe 0 --camera-width 1280 --camera-height 720 --work-width-mm 400 --work-height-mm 300
```

## Điều khiển YuMi

1. Nạp `robot/HandFollow.mod` vào cả `T_ROB_L` và `T_ROB_R` (cần option 616-1 PC Interface và `wobjPlaneXY` trong mỗi task).
2. Jog hai tay robot tới điểm tâm, đặt PP to routine `rHandFollow` ở cả hai task, chạy.
3. `python run.py` — laptop tự kết nối `192.168.125.1:1025` (ROB_L) và `:1026` (ROB_R).
4. Nhấn `SPACE` để robot bắt đầu bám tay; nhấn lại để giữ yên. Mất tay hoặc ra ngoài WORKSPACE thì robot giữ yên.

Chỉ theo dõi, không kết nối robot: `python run.py --no-robot`.

Nhấn `q` hoặc `Esc` để thoát. Tham số đầy đủ:

```powershell
python run.py --help
```

Mặc định dùng `mo_hinh/pose_landmarker_lite.task` và `mo_hinh/hand_landmarker.task`. Có thể chọn model khác bằng `--pose-model` và `--hand-model`.

## Kiểm tra

```powershell
python -m unittest discover -s kiem_tra -p "kiem_tra_*.py"
```
