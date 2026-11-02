# Hand Pose Webcam

Ứng dụng webcam dùng MediaPipe để phát hiện người, cánh tay và bàn tay. Phần lấy mẫu, chống rung, mô hình XY và tính góc YuMi được tích hợp trực tiếp vào chương trình chính, chạy bằng một lệnh `python run.py`. Chương trình hiển thị ba cửa sổ: `Webcam`, `3D Model` và `Wrist XY`; không cần bộ trung gian, trình duyệt hay chương trình nhận dữ liệu riêng. KAPAO vẫn là detector tùy chọn.

Cửa sổ `Wrist XY` theo dõi riêng cổ tay trái/phải, hiển thị X/Y theo pixel của ảnh camera và vị trí tương ứng trên lưới. Tọa độ ảnh có gốc `(0,0)` ở góc trên bên trái; đây không phải tọa độ robot hay khoảng cách vật lý. Cửa sổ báo trạng thái chờ/tạm dừng, mất người hoặc keypoint cổ tay không đủ tin cậy.

## Chế độ XY tích hợp mặc định

- Mặt phẳng điều khiển có **Z camera đặt trước 1.000 mm**; đây là giả thiết cấu hình, không phải phép đo chiều sâu.
- Camera và nhận diện tiếp tục chạy theo nhịp nguồn. Bộ tính chọn tối đa **5 mẫu keypoint mới/giây**, trước hiệu chỉnh, lọc, IK và cập nhật hình mô hình.
- Lọc trung vị 3 mẫu, EMA 0,35, vùng giữ 2 mm và giới hạn tốc độ XY 30 mm/s. Cửa sổ 3 mẫu trải trên 400 ms ở 5 Hz; lọc làm chậm đáp ứng, nên cần thử trên webcam thật.
- Trung điểm hai vai bù dịch thân người. Mô hình vai–khuỷu–cổ tay và bàn tay được lọc trên mặt phẳng camera; Z của các điểm điều khiển giữ cố định.
- Cửa sổ `3D Model` hiển thị mô hình đã lọc, tọa độ camera/robot riêng biệt và hai bảng **J1–J7**, vận tốc góc, dự báo +200 ms.
- Góc khớp được giải bằng IK của IRB 14000. Z flange trong hệ gốc robot giữ theo tư thế tham chiếu, riêng với Z camera 1.000 mm. Đây là kết quả tính toán để quan sát; chưa có TCP dụng cụ, hiệu chuẩn camera–robot hoặc đầu ra RobotStudio.

Sau khi bắt đầu nhận diện, giữ tư thế trung lập trong 20 mẫu hợp lệ, khoảng 4 giây ở 5 Hz. Nhấn `c` hiệu chỉnh lại; tạm dừng giữ góc lịch sử và dừng dự báo. Tiếp tục sẽ thu lại mốc trung lập, giữ nguyên tư thế robot tính toán cuối cùng.

```powershell
python run.py
# Các giá trị mặc định, có thể thay đổi trực tiếp:
python run.py --xy-depth-mm 1000 --xy-analysis-hz 5 --xy-calibration-frames 20 --xy-median-window 3 --xy-ema-alpha 0.35 --xy-deadband-mm 2 --xy-speed-mm-s 30
```

Khi chưa nhập `--xy-focal-px FX FY` và `--xy-principal-point-px CX CY`, XY dùng gain quy ước `--xy-scale-m 0.2` theo chiều cao ảnh, chưa phải thang đo vật lý đã hiệu chuẩn. Thông số nội tại phải khớp kích thước, ảnh lật và biến dạng ống kính của luồng đang dùng. Đầu vào dự đoán hoặc thiếu độ tin cậy không được dùng như phép đo mới; mất dữ liệu quá 0,5 giây thì giữ giá trị lịch sử.

Hình bàn tay dùng kết quả Hand Landmarker gần nhất còn trong 0,5 giây và được gắn vào cổ tay đã lọc. Nhờ đó `--hand-stride 2` không làm bàn tay bật/tắt giữa hai lần nhận diện. Các góc robot chỉ dùng phép đo vai/cổ tay mới từ detector cơ thể.

Để xem lại mô hình Pose world và các công cụ ánh xạ 3D cũ:

```powershell
python run.py --control-mode pose-world
```

## Chạy

Môi trường Python phải có các gói trong `requirements.txt` và MediaPipe. Chạy webcam mặc định:

```powershell
python run.py
```

Hoặc chọn detector KAPAO:

```powershell
python run.py --detector kapao
```

Các tham số webcam được truyền trực tiếp, ví dụ:

```powershell
python run.py --help
python run.py --camera-width 1280 --camera-height 720
```

## Cấu trúc chương trình

- `run.py` là file khởi chạy và điều phối duy nhất: camera, nhận diện, lấy mẫu XY, tính góc khớp và hiển thị.
- `tien_ich/` chứa các bộ lọc keypoint, nhận diện tay/tư thế, tính khoảng cách và ánh xạ tọa độ.
- `tien_ich/dieu_khien_xy_yumi.py` và `dong_hoc_yumi.py` là các mô-đun nội bộ, cùng dữ liệu hình học/giấy phép trong `du_lieu_yumi/`; không có launcher hay tiến trình phụ.
- `mo_hinh/` chứa mã kiến trúc KAPAO cùng model MediaPipe; `weights/` chứa trọng số detector.
- `danh_gia.py` cung cấp hậu xử lý cần thiết cho detector KAPAO.
- `kiem_tra/` chứa các file kiểm tra chức năng quan trọng; chạy bằng `python -m unittest discover -s kiem_tra -p "kiem_tra_*.py" -v`.

Mặc định (`--start-mode zone-thumbs-up`) chương trình chỉ bắt đầu khi đúng một người đứng ổn định trong vùng làm việc 80–120 cm (5 lần đo liên tiếp) **và** giơ thumbs-up 1 giây. Khi đang chạy, giữ **lòng bàn tay mở** 3 giây để tạm dừng; giữ **nắm tay** 0,4 giây để tiếp tục; giữ ký hiệu **V** 2 giây để kết thúc. `--start-mode zone` tự bắt đầu khi vào vùng; `--start-mode thumbs-up` bỏ qua khoảng cách. Nhấn `q` hoặc đóng một trong ba cửa sổ để thoát. Nhấn `c` để hiệu chuẩn lại tư thế trung tính cho bản xem trước cổ tay.

## Keypoint và mô phỏng 3D

Màn hình camera vẽ các điểm mặt tham chiếu, vai, khuỷu, cổ tay và 21 keypoint mỗi bàn tay. Bộ theo dõi có thể dự đoán vị trí trong tối đa hai khung hình mất điểm, nhưng bộ tính XY loại các điểm dự đoán đó. Chế độ XY mặc định chỉ vẽ mô hình từ snapshot đã lọc ở nhịp phân tích, nên khung camera xen giữa không làm mô hình cập nhật theo nhiễu thô. Với `--control-mode pose-world`, cửa sổ trở lại bản xem trước landmark world và vector cổ tay như đường xử lý cũ.

Bản xem trước cổ tay trong `pose-world` hiển thị thay đổi ΔX/ΔY/ΔZ của vector từ trung điểm hai vai đến cổ tay so với tư thế trung tính, yêu cầu đúng một người và confidence landmark đủ cao. Giá trị là **ước lượng tương đối của mô hình**, không phải tọa độ camera đã hiệu chuẩn hay tọa độ TCP robot. Các chế độ hiện tại đều không gửi lệnh robot.

## Khoảng cách vùng làm việc

Khoảng cách được đo bằng **độ rộng hai vai** trên ảnh theo mô hình camera phối cảnh: `Z_ngực = scale · fx · W_vai · cosθ / độ_rộng_vai_px`, trong đó `θ` là góc xoay thân lấy từ world landmarks (từ chối nếu > 25°). Gốc tọa độ là trung điểm hai vai. Khoảng cách cổ tay = `Z_ngực` + chênh lệch Z (world landmarks) giữa cổ tay và trung điểm vai; màn hình hiển thị khoảng cách và XYZ camera (m) của từng cổ tay. Giá trị làm mượt bằng median theo track. Không còn dùng Depth Anything vì sai số hệ thống lớn khi đo thực tế (~+0,74 m tại 0,8 m).

**Cần hiệu chuẩn để dùng kiểm tra vùng làm việc có cơ sở.** Khi chưa xác nhận hiệu chuẩn, chương trình hiện thông báo và chuyển sang bắt đầu bằng thumbs-up, không kiểm tra vùng. `--allow-uncalibrated-range` cho phép dùng ước lượng chưa xác nhận. Khoảng cách vùng làm việc này độc lập với Z mặt phẳng XY đặt trước. Đo bằng thước từ camera tới ngực:

**Thứ tự hiệu chuẩn khuyến nghị** (đều dành cho webcam laptop, chạy lần lượt):

```powershell
# 1. Tiêu cự và điểm chính bằng bàn cờ in sẵn (cần >= 12 góc nhìn, sai số tái chiếu <= 1 px mới lưu)
python run.py --focal-calibrate --chessboard-size 9 6 --chessboard-square-mm 25
# 2. Khớp scale chiều sâu ngực (đo bằng thước)
python run.py --range-calibrate --range-distances 0.8 1.0 1.2 --range-shoulder-width-m 0.38
# 3. Kiểm chứng độc lập; chỉ khi đạt mới đặt confirmed = true
python run.py --range-validate --range-distances 0.9 1.1 1.15
# 4. Đánh giá sai số do xoay thân (đứng cùng một khoảng cách, xoay thân ~0/15/30 độ)
python run.py --yaw-test --range-distances 1.0 --yaw-test-angles 0 15 30 --max-yaw-deg 25
# 5. Đánh giá sai số độ sâu cổ tay (duỗi cổ tay ra trước mặt phẳng ngực 0/0,2/0,4 m, đo bằng thước)
python run.py --wrist-validate --wrist-side RIGHT --wrist-offsets 0 0.2 0.4
```

- `--focal-calibrate` lưu `fx`, `fy`, `cx`, `cy` và sai số tái chiếu vào `data/hieu_chuan_khoang_cach.json`, đồng thời **đặt lại** `scale` và trạng thái xác nhận, vì tiêu cự đã đổi. Ảnh webcam được lật gương như khi chạy ứng dụng nên thông số khớp với khung hình thực dùng. Chưa hiệu chỉnh méo ống kính.
- `--range-calibrate` in thêm độ rộng vai ngầm định (`scale × độ rộng vai nhập`); nếu lệch nhiều so với số đo bằng thước thì kiểm tra lại độ rộng vai hoặc cách đứng.
- `--yaw-test` in sai số có và không có hiệu chỉnh `cos(góc xoay)`, rồi lưu vào `yaw_metrics`. Khi chạy thật, thân xoay quá `--max-yaw-deg` (mặc định 25°) thì không đo.
- `--wrist-validate` so sánh phần duỗi tay thật với `dz` của MediaPipe, in MAE/RMSE/sai số lớn nhất, hệ số `gain` (lệch tỉ lệ hệ thống) và kiểm tra dấu. Đạt khi MAE ≤ 5 cm, sai số lớn nhất ≤ 8 cm, đúng dấu. Chưa đạt thì nhãn cổ tay trên màn hình hiện `DEPTH UNVERIFIED`.

Các bước này chỉ làm webcam laptop khả thi hơn để tiếp tục phát triển, **không** chứng minh phương pháp đạt độ chính xác cần cho robot thật; cổ tay đặc biệt phụ thuộc độ sâu tương đối do MediaPipe suy luận.

```powershell
python run.py --working-distance-m 1.0 --working-distance-tolerance-m 0.2
```

Đây là ước lượng từ một camera, phụ thuộc độ rộng vai thật, góc xoay thân và tư thế; khoảng cách cổ tay còn phụ thuộc độ sâu tương đối do MediaPipe suy luận nên kém chính xác hơn khoảng cách ngực. Không phải cảm biến an toàn và không gửi lệnh/tọa độ tới robot; chỉ dùng làm đầu vào dry-run sau khi đã kiểm chứng bằng thước.

Gốc XYZ trong preview 3D và vector cổ tay là trung điểm 3D của hai vai MediaPipe (landmark 11 và 12). Nếu một trong hai vai thiếu hoặc confidence thấp thì vector không được xem là hợp lệ. Gốc này đi cùng người, không phải gốc sàn/camera/robot; không thay thế hiệu chuẩn camera–robot.

## Phase 1 — xem trước ánh xạ tọa độ (dry-run)

Có thể kiểm tra phép ánh xạ từ `delta_cm` đã trừ tư thế trung tính sang một điểm đích giả lập bằng mm trong hệ robot đã chọn. Mapper **không trừ neutral lần nữa**, không kết nối TCP và không gửi tọa độ robot. Nếu không có cấu hình hiệu chuẩn, UI báo `MAPPING NOT CONFIGURED`; không có hệ số/gốc robot mặc định.

Để bật preview sau khi đã đo và xác nhận hiệu chuẩn bên ngoài chương trình, tạo một JSON riêng và chạy:

```powershell
python run.py --control-mode pose-world --mapping-config C:\path\to\confirmed_mapping.json --mapping-log C:\path\to\mapping.csv
```

JSON phải khai báo đầy đủ:

- `calibration_confirmed`: boolean `true`, chỉ đặt sau khi kiểm tra hiệu chuẩn thực tế;
- `robot_frame`: tên hệ tham chiếu robot đã xác nhận;
- `tracked_side`: `RIGHT` hoặc `LEFT`;
- `home_robot_mm`: tọa độ XYZ home do người vận hành xác nhận, tính bằng mm;
- `mm_per_cm_matrix`: ma trận 3×3 đã hiệu chuẩn, đổi vectơ `delta_cm` sang offset mm. Ma trận được áp dụng cho **vectơ tịnh tiến**, không mô tả hướng TCP;
- `max_offset_mm`: giới hạn offset tuyệt đối cho từng trục;
- `max_speed_mm_s`, `max_sample_interval_s`, `max_delta_step_cm`, `min_confidence`: giới hạn chuyển động giả lập và chất lượng dữ liệu.

Không dùng ma trận identity, home giả định hoặc chiều trục suy ra từ cửa sổ 3D làm hiệu chuẩn robot. Trước khi xác nhận cấu hình, thu ít nhất 30 frame neutral hợp lệ, kiểm tra các chuyển động theo từng trục, xác minh dấu/trục bằng dữ liệu đo và so sánh với các dịch chuyển đã đo thực tế. Độ sâu từ camera RGB đơn mắt vẫn có thể sai đáng kể.

Mapper từ chối confidence thấp, mất tracking, delta NaN/Inf, bước nhảy, offset vượt giới hạn và frame cũ/`dt` không hợp lệ; các trường hợp đó không tạo target XYZ mới. Tốc độ target giả lập được giới hạn theo `max_speed_mm_s`. CSV có timestamp, landmark vai/cổ tay thô (đơn vị m), delta đã lọc (cm), offset/target (mm), frame và trạng thái. UI luôn ghi `DRY-RUN TARGET`; đây **không phải lệnh robot** và không được truyền vào bất kỳ kênh gửi nào.

Việc nhận diện người hiện chỉ dùng tính liên tục của track hình học; nếu track đổi hoặc không còn đúng một người, neutral và mapping được khởi tạo lại. Đây chưa phải nhận dạng danh tính: thay người ngay tại cùng vị trí mà detector giữ nguyên track có thể không bị phát hiện. Chỉ thử với một người điều khiển duy nhất trong khung hình.

## Ổn định cổ tay (`tien_ich/on_dinh_co_tay.py`)

Bật mặc định trong `--control-mode pose-world`, chạy giữa kết quả Pose world và `RelativeWristPreview`. Bộ lọc làm việc trên vector cổ tay so với trung điểm hai vai. Hand Landmarker chỉ cung cấp pixel ảnh để xác nhận cổ tay và tái dùng kết quả cử chỉ có sẵn. Chế độ XY dùng bộ lọc tích hợp riêng với các tùy chọn `--xy-*`, không chạy thêm đường lọc world này.

Trạng thái: `TRACKED` (dữ liệu đi tiếp) → `UNCERTAIN` (mất/ngoại lai ngắn, mapper giữ nguyên, không reset) → `LOST` (quá `--wrist-uncertain-frames` hoặc quá 0,4 s, mapper reset) → `REACQUIRING` (cần `--wrist-reacquire-frames` mẫu liên tiếp nhất quán mới trở lại `TRACKED`). Mẫu bị loại khi: không hữu hạn, visibility thấp, cổ tay ngoài khung, cánh tay quá dài, độ dài xương lệch >35% so với tham chiếu, tốc độ > `--wrist-max-speed`.

| Cờ | Mặc định | Ý nghĩa |
|---|---|---|
| `--no-wrist-stabilizer` | bật | hoàn nguyên hoàn toàn về đường cũ (EMA 0,35 trong preview) |
| `--wrist-min-cutoff` | 1.5 | One Euro cutoff tối thiểu (Hz); thấp = mượt hơn, trễ hơn |
| `--wrist-beta` | 6.0 | tăng thì giảm trễ khi tay di chuyển nhanh |
| `--wrist-max-speed` | 6.0 | m/s, loại bước nhảy |
| `--wrist-min-visibility` | 0.5 | ngưỡng tin cậy cổ tay/vai |
| `--wrist-uncertain-frames` | 6 | số frame được giữ ở `UNCERTAIN` |
| `--wrist-reacquire-frames` | 4 | số mẫu nhất quán để lấy lại |
| `--wrist-hand-refine` | tắt | cho phép cổ tay bàn tay (pixel) gần cổ tay Pose cứu mẫu visibility 0,3–0,5 |
| `--wrist-keep-preview-smoothing` | tắt | giữ thêm EMA cũ (lọc kép, trễ hơn) |
| `--wrist-benchmark-csv` | — | ghi CSV từng frame; in tóm tắt jitter/dropout/độ trễ khi thoát |
| `--camera-fps`, `--camera-fourcc` | driver | ví dụ `--camera-fourcc MJPG --camera-fps 30`; ở ánh sáng yếu driver thường tự hạ FPS |

So sánh mô phỏng trước/sau: `python danh_gia_on_dinh_co_tay.py --seed 0`. Với 3 seed, so với đường cũ: RMSE ~3,1 → ~1,9 cm, ngoại lai >5 cm ~38 → 1–2 frame, trễ 67 → 33 ms. **Jitter tăng nhẹ** (~1,58 → ~1,98 cm RMS gia tốc) vì ưu tiên giảm trễ; muốn mượt hơn đặt `--wrist-min-cutoff 1.0 --wrist-beta 3` (jitter ~1,70, trễ ~67 ms). Số liệu này chỉ từ chuỗi mô phỏng, **chưa phải đo với người thật dưới ánh sáng thật**.

Giới hạn còn lại: z/độ sâu cổ tay của MediaPipe vẫn là ước lượng của model; chưa tinh chỉnh bằng Hand world (khác hệ tọa độ, chưa đối chiếu được an toàn); chưa có đo thực tế về ánh sáng thay đổi; độ trễ camera/driver không được stabilizer đo; model vẫn là `lite` trong chế độ VIDEO đồng bộ (có thể thử `pose_landmarker_full.task` nhưng chậm hơn); mọi đầu ra chỉ là dry-run, không gửi lệnh tới ABB.

## Mô hình và giới hạn

Các model MediaPipe nằm trong `mo_hinh/`. Detector KAPAO yêu cầu weight tương ứng trong `weights/`; nếu thiếu, dùng `python data/scripts/download_models.py` để tải. Ảnh, video, dataset cục bộ và kết quả trong `runs/` không bị xóa bởi ứng dụng.

`danh_gia.py` được giữ vì webcam dùng bộ giải mã/phối hợp keypoint KAPAO từ module này. Các chức năng huấn luyện, demo ảnh/video và giao diện gán nhãn riêng đã được loại khỏi launcher webcam.

## Nguồn gốc KAPAO

KAPAO là phương pháp ước lượng tư thế nhiều người được công bố tại ECCV 2022: [Rethinking Keypoint Representations: Modeling Keypoints and Poses as Objects for Multi-Person Human Pose Estimation](https://arxiv.org/abs/2111.08557). Mã nền được fork từ [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) tại commit `5487451`. Xem `LICENSE` để biết điều khoản sử dụng.
