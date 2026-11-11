# BÁO CÁO RÀ SOÁT VÀ KIỂM THỬ OFFLINE HAND_POSE

Ngày lập: **11/11/2026**, múi giờ **Asia/Saigon**.

**Kết quả:** bộ test hiện có **9 PASS, 0 FAIL, 0 ERROR**; bộ kiểm tra bổ sung **31 PASS, 0 FAIL, 0 ERROR**. Các công thức và mẫu tổng hợp đã kiểm tra không cho thấy đảo thứ tự X/Y hoặc đảo dấu giữa Python và RAPID. Có sai khác tài liệu, độ trễ kiểm tra ROI do EMA và các thiếu sót xử lý đầu vào/giao thức được nêu dưới đây. Kết quả này chưa chứng nhận chuyển động thực tế của robot.

## 1. Phạm vi, tệp và phiên bản

Chỉ rà soát mã cục bộ tại `C:\Users\lenovo\Điều khiển ABB\hand_pose-master`. Không truy cập GitHub, `robotstudio_bridge` hay mã/config/dữ liệu của dự án khác. Không chạy `run.main()`, không mở camera, không khởi tạo model nhận diện, không kết nối controller và không phát lệnh robot. TCP thực nghiệm chỉ dùng `127.0.0.1` với cổng do hệ điều hành cấp. Không sử dụng dữ liệu đo camera/robot.

Đã đọc đầy đủ:

| Tệp | Nội dung kiểm tra |
|---|---|
| `run.py` | Tham số, flip, ROI, routing hai tay, ARM/HOLD, vòng xử lý, cleanup |
| `tien_ich/nhan_dien_co_tay.py` | Landmark cổ tay, visibility, normalized → pixel |
| `tien_ich/theo_doi_co_tay_2d.py` | EMA, mất điểm, ánh xạ ROI và helper cũ |
| `tien_ich/ket_noi_yumi.py` | Định dạng, freshness, TCP, reconnect |
| `robot/HandFollow.mod` | Phân tích tĩnh RAPID, parser, tọa độ và tham chiếu |
| `kiem_tra/kiem_tra_nhan_dien_co_tay.py` | 1 test detector dùng mock |
| `kiem_tra/kiem_tra_theo_doi_co_tay_2d.py` | 3 test tracker/helper |
| `kiem_tra/kiem_tra_ket_noi_yumi.py` | 5 test formatter/mapping/TCP, `FakeRobotTask` |
| `tien_ich/nhan_dien_ban_tay.py` | Đối chiếu vai trò HandLandmarker; chỉ dữ liệu vẽ overlay trong luồng này |
| `README.md`, `requirements.txt` | Hướng dẫn, cấu hình và phụ thuộc |

**Git:** không có `.git` trong thư mục hiện tại. Cả `git status --short`, `git branch --show-current`, `git rev-parse HEAD` đều trả:

```text
fatal: not a git repository (or any of the parent directories): .git
```

Vì vậy **không xác định được nhánh, commit hoặc diff chưa commit** của bản cục bộ. Không thể kết luận “working tree sạch” hay đồng nhất với `vhson021120/hand_pose`. Bản được kiểm tra được định danh bằng SHA-256 ở phụ lục, thay cho một commit chưa thể xác minh.

Các mã băm trước/sau trong `bang_chung_kiem_thu.json` giống nhau cho cả 11 tệp gốc được theo dõi. Đối chiếu thêm với mã băm đầu phiên cho các tệp bắt buộc cũng trùng khớp. **Không sửa mã nguồn chính, RAPID, tài liệu gốc, requirements hoặc các test có sẵn.** Chỉ thêm tệp kiểm thử và báo cáo trong thư mục này; tắt ghi bytecode khi chạy.

Môi trường thực thi thực tế:

| Thành phần | Phiên bản đọc từ môi trường |
|---|---|
| Python | 3.11.9, 64 bit, MSC v.1938 |
| Interpreter | `.venv/Scripts/python.exe` của workspace |
| NumPy | 2.4.6 |
| opencv-contrib-python | 5.0.0.93 |
| MediaPipe | 1.2.2 |
| coverage.py | Không có; dùng tracing chuẩn Python để đo dòng |

Đây là metadata cài đặt, không phải xác nhận toàn bộ API/model MediaPipe hoạt động. Detector trong test được mock, không nạp model. Lệnh `python` không có trong PATH của shell; `.venv` ban đầu không truy cập được Python nền do sandbox. Sau quyền thực thi đã được chấp thuận, interpreter chạy được. Một lần in đường dẫn tiếng Việt qua console CP1252 gặp `UnicodeEncodeError`; script báo cáo dùng UTF-8 và hoàn tất. Đây là lỗi ghi thông tin môi trường, không phải FAIL/ERROR của unit test.

## 2. Truy vết toàn bộ luồng tọa độ

```text
frame camera [KHÔNG chạy trong đợt kiểm tra này]
  → cv2.flip(frame, 1)
  → PoseLandmarker: landmark 15 LEFT, 16 RIGHT
  → (u × width, v × height, visibility)
  → kiểm tra dữ liệu và EMA riêng từng nhãn
  → ROI theo kích thước frame thực và workspace cấu hình
  → X/Y mm quy ước, inside của điểm đã lọc
  → armed + có điểm + inside: target; còn lại None/HOLD
  → LEFT/RIGHT chọn ROB_L/ROB_R hoặc đổi link bằng --swap-arms
  → TCP: valid,X,Y\n; tối đa một chữ số thập phân
  → RAPID ReadLine → ParseTarget → Clamp → DEADBAND
  → pTarget = pCenter, cộng X/Y trong wFollow
  → MoveL bằng tFollow và wFollow [CHỈ phân tích tĩnh]
```

Nguồn: `run.py:232–277`, `tien_ich/nhan_dien_co_tay.py:75–86`, `tien_ich/theo_doi_co_tay_2d.py:25–68`, `tien_ich/ket_noi_yumi.py:17–24,54–101`, `robot/HandFollow.mod:98–110`.

`MediaPipeHandDetector` tạo landmark bàn tay để vẽ (`run.py:262,266`, `nhan_dien_ban_tay.py:118–120`). Nhãn `hand.is_left` không quyết định tay robot; routing robot dựa trên nhãn PoseLandmarker. Không sử dụng world-landmark hay độ sâu để tính mục tiêu robot.

### Công thức từ mã thực tế

Gọi kích thước frame thực là `W,H`, workspace là `A,B` mm, fill là `f`:

```text
s = min(W*f/A, H*f/B)                  [pixel/mm]
roi_width = A*s; roi_height = B*s
L = (W - A*s)/2; T = (H - B*s)/2
R = L + A*s; D = T + B*s
```

Theo `run.py:50–55`. Tỷ lệ ROI là `A:B`; `fill` không có nghĩa chiếm cùng tỷ lệ ở cả chiều ngang lẫn chiều dọc.

Với điểm đã lọc `x,y`, theo `theo_doi_co_tay_2d.py:63–68`:

```text
u = clamp((x-L)/(R-L), 0, 1)
v = clamp((y-T)/(D-T), 0, 1)
X = (u-0.5)*A
Y = (0.5-v)*B
inside = (L <= x <= R) and (T <= y <= D)
```

Trong ROI, tương đương `X=(x-W/2)/s`, `Y=(H/2-y)/s`. Hai điểm đối xứng qua tâm ROI cho tọa độ đối dấu. X dương sang phải, Y dương hướng lên **trên ảnh sau flip**. Giá trị ngoài ROI được clamp để hiển thị nhưng `run.py:274–277` gửi HOLD khi `inside=False`.

`screen_pixel_to_centered_mm` tại `theo_doi_co_tay_2d.py:71–87` là helper khác, dùng `mm_per_pixel`; **không được `run.py` gọi**. Không áp mặc định 0,35 của helper/test cũ cho đường điều khiển hiện tại.

### Flip và LEFT/RIGHT

`run.py:257` lật ngang **trước nhận diện**. Với pixel rời rạc, `x'=W-1-x`, Y giữ nguyên; không có phép đổi dấu X bổ sung sau đó. Tâm toán học của mapping là `W/2,H/2`, còn phép flip pixel phản xạ quanh `(W-1)/2`; không nên giả định từng cặp pixel trước/sau flip cho X đối dấu tuyệt đối quanh `W/2`.

Mã gán landmark 15 thành LEFT, 16 thành RIGHT (`nhan_dien_co_tay.py:77`). Không có bước tráo nhãn thủ công sau flip. Không thể suy ra chỉ từ phép flip rằng MediaPipe chắc chắn tráo nhãn giải phẫu. Cần ảnh/video đã biết tay trái/phải để kiểm tra ý nghĩa thực tế của đầu ra mô hình trên ảnh phản chiếu.

Mặc định `LEFT→ROB_L`, `RIGHT→ROB_R`; `--swap-arms` đổi hai **link đích**, không đổi X/Y hay dấu (`run.py:235–238`). Ý nghĩa “tay người trái điều khiển tay robot nào” vẫn cần được thống nhất với cách bố trí camera/người vận hành.

## 3. Bảng đầu vào và đầu ra quy đổi

**Toàn bộ đầu vào sau là dữ liệu tổng hợp theo cấu hình mặc định, không phải đo thực nghiệm camera.** Frame 1280×720, workspace 400×300 mm, fill 0,8 ⇒ ROI `(256,72,1024,648)` = 768×576 pixel. ROI chiếm 60% chiều ngang, 80% chiều dọc. Scale là `25/48 ≈ 0,5208333333 mm/pixel`, không phải hiệu chuẩn vật lý.

Các ca dưới đã được đối chiếu bằng `MappingChecks`. Cột TCP giả định đang armed, điểm này là điểm sau lọc, dữ liệu còn mới và robot yêu cầu thông điệp.

| Pixel sau lọc `(x,y)` | X kỳ vọng mm | Y kỳ vọng mm | inside | Thông điệp kỳ vọng |
|---|---:|---:|---|---|
| (640,360) | 0 | 0 | True | `1,0.0,0.0\n` |
| (256,72) | −200 | +150 | True | `1,-200.0,150.0\n` |
| (1024,72) | +200 | +150 | True | `1,200.0,150.0\n` |
| (256,648) | −200 | −150 | True | `1,-200.0,-150.0\n` |
| (1024,648) | +200 | −150 | True | `1,200.0,-150.0\n` |
| (736,360) | +50 | 0 | True | `1,50.0,0.0\n` |
| (544,360) | −50 | 0 | True | `1,-50.0,0.0\n` |
| (640,264) | 0 | +50 | True | `1,0.0,50.0\n` |
| (640,456) | 0 | −50 | True | `1,0.0,-50.0\n` |
| (1025,360) | +200, clamp | 0 | False | `0,0.0,0.0\n` |
| (255,360) | −200, clamp | 0 | False | `0,0.0,0.0\n` |
| (640,71) | 0 | +150, clamp | False | `0,0.0,0.0\n` |
| (640,649) | 0 | −150, clamp | False | `0,0.0,0.0\n` |

Đổi frame thành 640×480 với cùng workspace/fill ⇒ ROI `(64,48,576,432)`, scale 0,78125 mm/pixel; `(384,240)` ⇒ `(50,0,True)`. Cùng số pixel dịch chuyển không luôn tương đương cùng mm khi độ phân giải/tỷ lệ ảnh đổi. Mã dùng kích thước `frame.shape` thực tế (`run.py:258,265`), không giả định camera đã đáp ứng độ phân giải yêu cầu.

## 4. EMA, mất điểm và dữ liệu không hợp lệ

EMA tại `theo_doi_co_tay_2d.py:38–44` áp lên cả `(x,y,visibility)`:

```text
q_t = p_t                                nếu chưa có lịch sử
q_t = q_(t-1) + alpha*(p_t - q_(t-1))     nếu quan sát liên tục
```

Với bước nhảy Δ và alpha=0,25, sai số sau n cập nhật là `0,75^n * Δ`; lần đầu còn 75% sai số, sau 11 cập nhật đạt ít nhất 95% bước nhảy. Với chuyển động đều, độ trễ ổn định tương đương ba khoảng cập nhật. Đây là hệ quả toán học; chưa có số đo FPS/độ trễ nên không quy đổi thành mili giây.

**Phản ví dụ ROI đã chạy:** khởi tạo LEFT `(640,360,0.9)`, rồi quan sát thô `(1100,360,0.9)` ngoài ROI. EMA cho `(755,360,0.9)`, mapping cho `(59.895833…,0,True)`. Khi armed, luồng hiện tại vẫn tạo target, formatter cho X≈59,9 mm. Nếu giữ quan sát thô đó, điểm lọc còn trong ROI sáu cập nhật và ra ngoài ở cập nhật thứ bảy. Đây là độ trễ có thể tái hiện, không phải số liệu từ camera. Test xác nhận cả hai trạng thái raw-outside/filtered-inside; chưa chạy vòng camera.

Thiếu điểm, visibility dưới ngưỡng, NaN/∞ hoặc sai shape được bỏ và lịch sử của nhãn đó bị xóa (`theo_doi_co_tay_2d.py:29–37,46`). Nhận diện lại dùng ngay điểm mới, không nội suy qua quãng mất điểm. Vì vậy có thể xuất hiện bước nhảy vị trí khi nhận lại; không thấy cơ chế giới hạn bước nhảy trong Python.

| Loại dữ liệu | Hành vi xác minh |
|---|---|
| NaN/±∞ tại từng thành phần tracker | Bị bỏ, không xuất điểm |
| visibility thấp | Bị bỏ trước EMA; mất lịch sử nhãn tương ứng |
| shape khác `(3,)` nhưng chuyển được thành mảng số | Bị bỏ |
| Chuỗi `'bad'` trong tracker | `ValueError` tại chuyển mảng, chưa được xử lý thành bỏ điểm |
| Hai tọa độ hữu hạn cực lớn trái dấu | EMA có thể overflow, đầu ra không hữu hạn; đã tái hiện với ±1e308 |
| NaN/±∞ ở formatter | Thành HOLD |
| Tuple sai chiều dài/chuỗi không phải số ở formatter | `ValueError`, không thành HOLD |
| NaN đưa thẳng vào helper ROI | X có thể NaN, `inside=False` |
| ±∞ đưa thẳng vào helper ROI | Clamp về biên, `inside=False` |
| ROI có chiều rộng 0 | `ZeroDivisionError` |

Các ca cực lớn/phi số/ROI suy biến là phép thử độ bền API. Chưa có bằng chứng MediaPipe thật tạo ra chúng. Luồng bình thường đã có tracker và kiểm tra workspace CLI bảo vệ nhiều trường hợp. Visibility NaN không qua điều kiện detector; visibility +∞ có thể qua detector nhưng tracker loại sau đó.

## 5. TCP giả lập

Tái sử dụng `FakeRobotTask` trong `kiem_tra/kiem_tra_ket_noi_yumi.py:9–34`. Server bind `127.0.0.1:0`; client chỉ được kết nối literal `127.0.0.1`. Runner bổ sung chặn camera và chặn đích TCP khác. Không dùng địa chỉ controller trong bất kỳ phép thử kết nối nào.

| Hạng mục | Kết quả và căn cứ |
|---|---|
| Thứ tự/trục/dấu | `format_target` luôn xuất `valid,x,y` (`ket_noi_yumi.py:21–24`). Tổng hợp `(12.34,-56.78)` nhận `1,12.3,-56.8\n`; đổi dấu hai trục vẫn đúng. RAPID gán x/y tương ứng, không đổi dấu. |
| Làm tròn | Một chữ số thập phân ⇒ sai số lượng tử tối đa khoảng 0,05 mm mỗi trục, khoảng 0,07071 mm nếu xét độ lớn sai số 2D; không gồm sai số nhận diện/EMA/robot. Đã test số dương/âm và `-0.0`; `-0.0` có giá trị số bằng 0. Không giả định mọi trường hợp đúng nửa bước đều làm tròn ra xa 0. |
| Ảnh hưởng DEADBAND | Do so sánh hai giá trị đã làm tròn, độ chênh mỗi trục có thể khác dữ liệu chưa làm tròn đến khoảng 0,1 mm, ảnh hưởng quyết định gần ngưỡng 2 mm. Không thấy sai số đảo trục từ làm tròn. |
| Mục tiêu mới nhất | `set_target` ghi đè một slot dưới lock; chỉ lấy khi nhận dòng request (`54–66,99–101`). Hai lần publish trước request trả lần sau. Không có hàng đợi target Python. |
| Freshness | Mặc định `max_age=0.3`; tuổi **lớn hơn** ngưỡng thành HOLD (`33,64–66`). Test thêm kiểm đúng tại ngưỡng vẫn hợp lệ và sau ngưỡng thành HOLD bằng đồng hồ mock. Chỉ gửi khi có request, không phải watchdog chủ động cắt chuyển động. |
| Mất dữ liệu | None/disarmed/mất điểm/điểm đã lọc ngoài ROI ⇒ HOLD. Test None và stale đều đạt. |
| Connect timeout/retry | `connect_timeout=2.0`, `retry_delay=1.0` mặc định. Test mock `socket.timeout` xác nhận `_run` đi vào retry và truyền đúng timeout/delay; không đo thời gian timeout mạng thật. |
| Recv timeout Python | 0,5 giây chỉ tiếp tục chờ, không tự reconnect (`80,94–95`). Server im lặng 0,65 giây rồi gửi request vẫn nhận đúng target. |
| Mất kết nối/reconnect | EOF/OSError ra khỏi phiên, `connected=False`, chờ retry rồi kết nối lại (`73–87,96–97`). Test đóng client và accept lại đạt, cả sau khoảng im lặng. |
| Request bị chia gói | Server gửi `R`, chưa có LF ⇒ chưa có phản hồi; gửi LF sau ⇒ đúng một phản hồi chứa target mới nhất. `pending` giữ phần chưa đủ dòng (`90–101`). |
| Nhiều request một gói | `R\nR\nR\n` ⇒ ba phản hồi đúng thứ tự; vòng while rút hết dòng hoàn chỉnh. |
| Nội dung request sai | `NOT_R\n` vẫn nhận target. Đây là hành vi đã tái hiện: code bỏ nội dung trước LF thay vì kiểm tra `R`. |

**Giới hạn:** test chia gói trên Python kiểm chiều robot-giả-lập → Python. Chiều Python → RAPID chỉ phân tích tĩnh `ReadLine`; không dùng việc FakeRobotTask nhận được dòng để khẳng định ABB parser đã chạy đúng. Peer im lặng nhưng socket còn mở có thể giữ `connected=True` vô hạn; cờ này không phải heartbeat hoạt động RAPID. Buffer Python chưa giới hạn độ dài khi không có LF.

## 6. Rà soát tĩnh RAPID

| Thành phần | Kết luận từ `robot/HandFollow.mod` |
|---|---|
| `START_IS_CENTER` | `TRUE` ở dòng 33. `pCenter` lấy tại lúc bắt đầu routine bằng `CRobT` (68), không phải mặc định gốc WorkObject. |
| Chế độ FALSE | Chỉ đổi X/Y tâm thành 0 (69–72); Z và các phần pose khác lấy từ pose bắt đầu. |
| `WOBJ_NAME_L/R` | Trái: `wobjPlaneXY`; phải: `wobjPlaneXY1` (25–26). `GetDataVal` đọc vào `wPlane`, gán sang `wFollow` (55–62). Mọi task không có tên `T_ROB_L` đi nhánh phải. |
| `wFollow` | WorkObject sao chép từ dữ liệu task tại lúc khởi động, không phải phép biến đổi hiệu chuẩn từ ảnh camera. |
| Tool | `tFollow := CTool()` ở 66, ghi đè giá trị khai báo ban đầu ở 41. Phải xác nhận TCP/load/tool đang hoạt động trên controller. |
| `pCenter` qua reconnect | Chụp một lần trước vòng accept (68,80–88). Reconnect reset `lastX/lastY` thành 9999, không đọc lại tâm hay TCP hiện tại. |
| Phép cộng | `pTarget := pCenter`; `pTarget.trans.x := pCenter.trans.x + x`, Y tương tự (105–108). Không hoán đổi hay đổi dấu. |
| Các thành phần giữ lại | Copy toàn bộ pCenter, chỉ sửa X/Y; giữ Z, quaternion, configuration và external axes của pose lưu. |
| `LIMIT_X/Y` | Clamp offset X ±200, Y ±150 (29–30,102–103,153–156). Không phải giới hạn tọa độ tuyệt đối của WorkObject hoặc chứng minh vùng an toàn. |
| `DEADBAND` | 2 mm (35). Move nếu **một trong hai trục** lệch ≥2 so với target đã phát gần nhất (104,109–110), không phải khoảng cách Euclid hay sai số TCP đo thực. |
| HOLD | `valid!=1`/parse không hợp lệ chỉ `WaitTime 0.02` (101–116). Không thấy `StopMove`/hủy MoveL đã phát. Chưa đủ căn cứ gọi đây là dừng tức thì. |
| `ConfL`, tốc độ, zone | `ConfL \Off` (67), vFollow `[150,200,5000,1000]`, zone `z5` (38–39). Chưa kiểm thời gian/quỹ đạo/blending hay tính tới đích vật lý. |

Ví dụ **tính toán**, không đo: nếu tâm trong WorkObject là `(300,-100,Z)` và TCP gửi offset `(50,-20)`, target là `(350,-120,Z)`. Nếu `wFollow` xoay so với base, X/Y này là hướng của WorkObject, không tự động trùng hướng màn hình hay base robot.

### ParseTarget, ReadLine và timeout

`ParseTarget` tìm hai dấu phẩy rồi dùng `StrToVal` cho valid/x/y (`142–150`). Parser không ép valid thuộc `{0,1}`, nhưng điều kiện chuyển động chỉ chấp nhận `valid=1` (`101`). Không nên kết luận có lỗi LF-vs-CRLF chỉ từ `StrFind(...,"\0D\0A")`; biểu thức này tìm ký tự thuộc tập CR/LF trong RAPID, không nhất thiết đòi cặp CRLF liên tiếp. Không chạy RAPID compiler hoặc parser trong đợt này.

`ReadLine` nối nhiều chunk tới khi thấy LF (`126–135`), nên cấu trúc có hỗ trợ một dòng chia nhiều lần nhận. Tuy nhiên buffer là biến cục bộ, trả cả buffer và không giữ phần sau LF cho lần sau. Khi một lần nhận chứa hai dòng, dòng sau không được đưa vào lần `ReadLine` kế tiếp. Đây là thiếu sót framing tĩnh; giao thức chuẩn một request/một reply giảm khả năng gặp trường hợp này.

`RECV_TIMEOUT=3` áp cho từng `SocketReceive` (`37,131`), không phải deadline tổng để nhận xong một dòng. Các chunk đến rời rạc có thể kéo dài tổng thời gian quá 3 giây. Lỗi closed/timeout đóng client và quay lại accept (`118–123`); lỗi khác không có nhánh recovery rõ trong handler. Chưa xác minh trên controller giới hạn string, trường rỗng, số vượt miền, NaN/∞, lỗi `StrPart`/`StrToVal`, hoặc propagation lỗi hàm → routine.

## 7. Lỗi và thiếu sót được chứng minh

Các phát hiện sau chỉ rõ điều kiện tái hiện; không suy diễn rằng controller/camera thực đã gặp lỗi.

| ID | Phát hiện, vị trí | Bằng chứng và mức kết luận |
|---|---|---|
| D01 | ROI kiểm tra sau EMA: `run.py:261,274–276`; `theo_doi_co_tay_2d.py:42` | Test tổng hợp raw x=1100 vẫn tạo điểm filtered x=755, inside=True. Không đáp ứng cách hiểu “quan sát cổ tay ra ngoài ROI thì HOLD ngay” trong `README.md:31`/comment `run.py:275`. Nếu yêu cầu chỉ xét điểm lọc thì đây là hành vi cần ghi rõ, không phải lỗi công thức. |
| D02 | README sai tham số/scale: `README.md:9,12`; CLI thực ở `run.py:191–196` | README mô tả `--mm-per-pixel=0.35` nhưng parser không có tham số đó; đường điều khiển dùng ROI. Test cùng pixel x=740 cho helper cũ X=35, còn đường ROI X≈52,083333. Sai khác tài liệu chắc chắn. |
| D03 | README sai tên WorkObject phải: `README.md:28`; RAPID `26,60` | README chỉ yêu cầu `wobjPlaneXY` ở mỗi task; nhánh phải cần `wobjPlaneXY1`. Chưa biết controller hiện có đủ symbol hay không. |
| D04 | Tracker không bắt lỗi chuyển đổi phi số: `theo_doi_co_tay_2d.py:31` | Test `'bad'` ném ValueError. Đây là lỗ xử lý dữ liệu không hợp lệ của API, chưa thấy ở đầu ra MediaPipe thật. |
| D05 | Formatter có thể làm chết worker khi target sai hợp đồng: `ket_noi_yumi.py:21,54–58,83` | Test tuple sai số phần tử/phi số ném ValueError. Theo control flow, lỗi này thoát `_run` vì chỉ bắt OSError; kết luận chết worker dựa phân tích tĩnh, không tạo lỗi thread thật trong test. Pipeline chuẩn hiện xuất tuple số/None. |
| D06 | Không kiểm finite sau EMA: `theo_doi_co_tay_2d.py:42–49` | Test hai giá trị hữu hạn ±1e308 cho đầu ra không hữu hạn. Overlay `run.py:31` không đổi được inf/NaN thành int. Đây là ca tổng hợp cực đoan, không phải sai lệch camera đã quan sát. |
| D07 | Helper ROI không validate: `theo_doi_co_tay_2d.py:63–68` | Test ROI rộng 0 ném ZeroDivisionError, NaN lan ra output. Luồng chính đã bảo vệ phần lớn qua tracker và kiểm tra CLI. |
| D08 | Lịch sử pixel không đổi theo kích thước frame: `run.py:258,261,265`; tracker `38–46` | Test tâm 1280×720 rồi tâm 640×360, cùng normalized position, vẫn EMA với pixel cũ → target tạm thời lệch/ngoài ROI. Thiếu xử lý khi đổi kích thước được tái hiện; chưa xác nhận camera thực đổi kích thước giữa phiên. |
| D09 | Python phản hồi dòng không phải R: `ket_noi_yumi.py:99–101` | TCP giả lập gửi `NOT_R\n` vẫn nhận tọa độ. Thiếu validation request; chưa thấy robot chuẩn gửi sai dòng. |
| D10 | RAPID không giữ phần dư nhiều dòng: `HandFollow.mod:126–135` | Phân tích buffer cục bộ chứng minh thiếu bộ đệm cho dòng sau; chưa chạy parser ABB. Không đồng nghĩa TCP chia một dòng thành nhiều gói bị lỗi. |
| D11 | `--camera-fps` nhận NaN/∞ qua validation: `run.py:209–210` | Check chỉ `<0`; cả NaN và +∞ đều không bị loại. Phát hiện tĩnh; chưa gọi camera với các giá trị này. |

Không có bằng chứng lỗi đổi dấu/hoán đổi hai trục trong các phép biến đổi đã kiểm. Không có bằng chứng camera được hiệu chuẩn, robot đang sai WorkObject, hay target thực sự không tới được.

## 8. Nguy cơ X/Y chưa xác nhận trên hệ thực

| Nguy cơ | Điều chưa biết/cách xác minh |
|---|---|
| mm hiển thị khác mm tay thật | Không có hiệu chuẩn vật lý, homography, khoảng cách camera, lens distortion hay mặt phẳng tay được đo. Scale cấu hình chỉ là ánh xạ quy ước. Có thể dùng ảnh/video và vật chuẩn được cung cấp để hiệu chuẩn offline. |
| Nhãn giải phẫu sau mirror không khớp ý định | Cần ảnh/video có nhãn trái/phải để kiểm MediaPipe; không cần bật camera mới hoặc robot cho bước này. |
| Hướng WorkObject không khớp màn hình | Cần dữ liệu WorkObject/Tool thực. Có thể kiểm trước bằng backup/Virtual Controller; hướng chuyển động thật cần controller và quy trình kiểm tra được phê duyệt. |
| Giới hạn GUI khác giới hạn RAPID | Workspace CLI được phép đổi tùy ý nhưng RAPID cố định ±200/±150. Ví dụ GUI 600×300 cho X=300 ở biên, RAPID sẽ clamp còn200. Sự không khớp với cấu hình tùy biến là suy ra trực tiếp; chưa biết cấu hình vận hành dự kiến. |
| Reacquire/re-arm/reconnect tạo bước nhảy | Điểm nhận lại không được ramp; pCenter giữ cố định; target hợp lệ đầu tiên sau reconnect vượt deadband. Độ lớn và tác động vật lý chưa đo. |
| HOLD không dừng ngay | Mã chỉ ngừng phát MoveL mới; chuyển động đã được nhận hoặc đang thực hiện có thể tiếp tục. Thời gian dừng/quỹ đạo thực không suy ra từ unit test. |
| Tốc độ vòng lặp không ổn định | EMA không tính theo thời gian; cả pose và hand inference nằm trước publish target. Chưa có phép đo FPS/latency để kết luận tốc độ bám. |
| Hai tay cùng không gian | Không thấy đồng bộ hai task hoặc kiểm va chạm trong module; chưa biết cấu hình/giới hạn controller. Không thể kết luận an toàn chuyển động hai tay từ clamp X/Y. |

## 9. Kết quả unit test thực tế

Lệnh yêu cầu đã chạy với interpreter của chính workspace vì PATH không có `python`:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
& '.venv/Scripts/python.exe' -m unittest discover -s kiem_tra -p 'kiem_tra_*.py' -v
```

| Lần chạy | Test | PASS | FAIL | ERROR | SKIP | Thời gian unittest |
|---|---:|---:|---:|---:|---:|---:|
| Bộ có sẵn, lần thành công đầu tiên | 9 | 9 | 0 | 0 | 0 | 3,020 s |
| Bộ có sẵn trong lần đo coverage cuối | 9 | 9 | 0 | 0 | 0 | 2,445 s |
| Bộ bổ sung cuối | 31 | 31 | 0 | 0 | 0 | 3,086 s |

Tổng **40 test phân biệt** ở bản bổ sung cuối, không cộng lần chạy lại thành test mới. Một lượt thử bổ sung trước đó gồm 30 test cũng PASS (3,664 s); lượt đo coverage tương ứng của 9 test gốc PASS (2,960 s). Chạy lại để thêm mock connect-timeout và loại dòng bytecode tổng hợp số 0 khỏi mẫu số coverage. Các log tự sinh và JSON lưu kết quả lượt cuối; bản chép lần thành công đầu tiên lưu riêng.

Không có traceback FAIL/ERROR trong các lần unit test. Các exception trong test `assertRaises` là hành vi được dự kiến để tái hiện lỗ xử lý đầu vào, **không có nghĩa lỗi đã được sửa**.

Tệp bằng chứng:

- `unit_test_co_san_lan_dau.txt`: lệnh, bản chép stdout/stderr của lần chạy yêu cầu thành công đầu tiên.
- `unit_test_co_san_co_coverage.txt`: output tự ghi của 9 test trong lượt đo cuối.
- `unit_test_bo_sung.txt`: output tự ghi của 31 test bổ sung.
- `bang_chung_kiem_thu.json`: môi trường, kết quả, dòng chưa chạy và SHA-256 trước/sau.
- `kiem_thu_bo_sung.py`: 31 test mới nằm riêng, không sửa `kiem_tra/`.
- `chay_kiem_thu_offline.py`: runner giữ guard camera/loopback, ghi log và đo dòng.

### Độ bao phủ của bộ test gốc

Đo bằng `sys.settrace` và `threading.settrace`, tập dòng thực thi do thư viện chuẩn `trace` tìm, bỏ dòng bytecode số 0. **Bao gồm import/định nghĩa; không phải branch coverage hoặc coverage phần cứng.** Chỉ đo 9 test gốc; chưa cộng các test bổ sung. Không cài thêm package.

| Tệp | Dòng đã chạy / dòng có thể chạy theo phép đo | Tỷ lệ |
|---|---:|---:|
| `run.py` | 0 / 257 | 0,00% |
| `tien_ich/nhan_dien_co_tay.py` | 37 / 76 | 48,68% |
| `tien_ich/theo_doi_co_tay_2d.py` | 57 / 63 | 90,48% |
| `tien_ich/ket_noi_yumi.py` | 76 / 78 | 97,44% |
| `tien_ich/nhan_dien_ban_tay.py` | 0 / 102 | 0,00% |
| RAPID | Không có phép đo thực thi | Không áp dụng |

Tỷ lệ connector cao không chứng minh đủ ca lỗi: bộ test gốc chưa chạy nhánh OSError ở `ket_noi_yumi.py:83–84`, và không có branch coverage. Test detector dùng `__new__` và mock, không chạy constructor/model. Ba test tracker có một test `screen_pixel_to_centered_mm` là helper cũ; mapping ROI thực được test riêng tại `kiem_tra_ket_noi_yumi.py:44–51`, không được bỏ sót test này khi đánh giá.

Bộ gốc chưa kiểm flip/nhãn/swap/ARM tích hợp, ROI theo tỷ lệ ảnh, post-EMA raw-outside, sai dữ liệu phi số, chia/gộp request TCP hoặc timeout connect. Bộ bổ sung đã lấp một phần bằng các ca tổng hợp và test hành vi hiện tại; chưa chạy vòng chính `run.main()` hay RAPID.

## 10. Việc xử lý được offline và việc cần controller/robot

### Hoàn toàn xử lý hoặc chuẩn bị được offline

- Đồng bộ README với CLI/ROI và WorkObject tay phải; mô tả rõ HOLD và tâm cố định.
- Chọn quy tắc raw-outside có HOLD ngay hay theo điểm lọc; đưa vào hàm thuần để kiểm thử, rồi bổ sung regression test sau khi được duyệt sửa.
- Chặn dữ liệu phi số, ROI suy biến, post-EMA nonfinite; reset/rescale lịch sử khi kích thước frame đổi; kiểm finite FPS.
- Đồng bộ/kiểm tra giới hạn workspace Python với RAPID và mô tả rõ vùng offset quanh tâm.
- Giới hạn độ dài dòng TCP, xác thực R, deadline/heartbeat nếu yêu cầu, buffer phần dư RAPID; kiểm thử lỗi chia/gộp gói và retry bằng server giả.
- Viết harness mock toàn bộ camera, detector, GUI và link để chạy routing LEFT/RIGHT, swap, ARM/HOLD, camera read failure/cleanup mà không bật thiết bị.
- Chạy MediaPipe trên ảnh/video được người dùng cung cấp/cho phép; kiểm nhãn trên ảnh phản chiếu và timestamp mà không mở webcam.
- Kiểm chứng cú pháp, `ParseTarget`, socket và symbol bằng RobotStudio/Virtual Controller phù hợp nếu được cung cấp trong phạm vi được phép. Đây là kiểm tra ảo, không đòi nối robot thật; hiện chưa thực hiện.

### Cần dữ liệu/controller ABB hoặc robot thật để xác nhận cuối cùng

- Phiên bản RobotWare, option socket, khả năng compile/load module, symbol/tool/WorkObject thực trong hai task: cần controller hoặc Virtual Controller tương ứng/backup đáng tin cậy.
- `CTool`, TCP/load đã hiệu chuẩn, vị trí tâm vật lý, hướng trục WorkObject so với không gian thực và độ chính xác mm tại TCP: cần hệ robot thực cho xác nhận vật lý.
- Reachability, singularity, cấu hình tay, sai số bám, blending z5, va chạm, không gian hai tay, giới hạn cơ khí và hành vi HOLD/mất mạng khi đang MoveL: mô phỏng giúp chuẩn bị, nhưng xác nhận vận hành thực cần robot/controller và kế hoạch đã phê duyệt.
- Thời gian đáp ứng end-to-end camera → robot, sai số đo và độ lặp lại thực: cần đo trên hệ được cho phép; không thể điền con số từ test tổng hợp.

Chỉ riêng công thức ánh xạ, dấu/trục trên dữ liệu tổng hợp và robustness TCP Python không bắt buộc nối robot. Không đưa mọi vấn đề parser/cú pháp ABB vào nhóm “phải có robot thật”, vì Virtual Controller phù hợp có thể kiểm một phần.

## 11. Đề xuất theo thứ tự ưu tiên

| Mức | Đề xuất cụ thể | Tiêu chí kiểm tra trước khi coi xong |
|---|---|---|
| **P0** | Xác định và thực hiện đúng ý nghĩa HOLD/STOP khi mất tay, raw ngoài ROI, stale, disarm hoặc mất mạng. Nếu yêu cầu là HOLD ngay khi raw ra ngoài, kiểm raw ROI trước EMA. Không dùng thông điệp hiện tại như cam kết dừng khẩn cấp. | Regression raw x=1100 phải đúng chính sách đã chốt; mô tả rõ không phát mục tiêu mới hay hủy chuyển động; xác nhận chuyển động đang chạy bằng môi trường ABB thích hợp và robot khi được duyệt. |
| **P0** | Trước bất kỳ vận hành robot nào: xác nhận Tool, WorkObject của từng tay, pCenter, hướng trục, giới hạn offset và không gian chuyển động. | Kiểm bằng cấu hình thực/VC; sau đó bài thử vật lý có giới hạn được người dùng phê duyệt. Không coi ±200/±150 tự động là vùng an toàn. |
| **P1** | Sửa README cho đúng CLI/ROI và `wobjPlaneXY1`; đồng bộ giới hạn workspace ứng dụng/RAPID. | Hướng dẫn chỉ dùng tham số thực có; cấu hình không cho giao diện và robot hai miền X/Y khác nhau mà không báo. |
| **P1** | Validation đầu vào và đầu ra EMA; formatter chuyển dữ liệu không hợp lệ theo chính sách HOLD; worker không chết do dữ liệu sai hợp đồng; xử lý đổi frame size. | Các ca phi số/sai shape/NaN/∞/overflow/resize có kết quả đã định nghĩa và không làm rơi worker. Chuyển test characterization thành regression kỳ vọng mới sau khi sửa. |
| **P1** | Bổ sung test tích hợp main bằng mock: LEFT/RIGHT, swap, armed/disarmed, raw/filtered ROI, mất/nhận lại, camera failure/cleanup; kiểm parser RAPID trong VC. | Không mở camera/socket ngoài loopback khi chạy test; xác minh cả hai chiều chia/gộp dòng, trường thiếu/sai, số vượt miền. |
| **P2** | Siết framing TCP: xác thực R, giới hạn buffer, giữ phần dư, deadline toàn dòng; làm rõ connected chỉ là trạng thái socket và chính sách reconnect khi peer im lặng. | Test dòng sai/quá dài/không LF, nhiều dòng, packet split, silent peer đạt chính sách đã chọn; không replay target stale. |
| **P2** | Validation FPS hữu hạn, tài liệu deadband từng trục và tâm cố định qua reconnect; đo latency/hiệu chuẩn khi có dữ liệu cho phép. | NaN/∞ bị chặn; kết quả hiệu chuẩn/độ trễ có nguồn số đo thật, điều kiện đo và sai số. |

Không áp dụng bất kỳ đề xuất sửa mã nào trong đợt này. Các mục P0 là điều kiện ưu tiên trước vận hành, không phải tuyên bố đã phát hiện tai nạn hay va chạm thực tế.

## Phụ lục: SHA-256 bản cục bộ

| Tệp | SHA-256 |
|---|---|
| `run.py` | `4626a45bcef0e2ab7f5e5cdf09db6c73f9cd597d0c2a721262e8e9c66d46fc54` |
| `tien_ich/nhan_dien_co_tay.py` | `2d76cdfb275f0b395f9bb36df6d6cd1c333c44faa693e1b3b45250b9d0d1c634` |
| `tien_ich/theo_doi_co_tay_2d.py` | `27e6e46f78492cf390f6d2c5f0e9f3717d19d24111dbe22405cf92ad33541e60` |
| `tien_ich/ket_noi_yumi.py` | `d4bd00b848835d645a1818f094e45a26ce17e953ef87a945d38464be2f466a44` |
| `robot/HandFollow.mod` | `1d55047a1288024b937950f2e5d274ce073c06a1ad3d8e01260825a98a13b583` |
| `tien_ich/nhan_dien_ban_tay.py` | `640d09070859adf28f4f5ee61c08ae93e44505542f50380dc84978e179afc4c7` |
| `kiem_tra/kiem_tra_nhan_dien_co_tay.py` | `8ddd4c7b253f7a629f5f62c64c11e933136ea5e93160b622069331e5a0a12fb9` |
| `kiem_tra/kiem_tra_theo_doi_co_tay_2d.py` | `fe6f7367cce6f8f9865c50e3922238316c98b078803d5f56ec2fd505f449d1c2` |
| `kiem_tra/kiem_tra_ket_noi_yumi.py` | `76940b88410ad4ed09644e8998b8efb873383dec5d889084467489b0578e3cf2` |
| `README.md` | `96159298fe82f537894953697fb821a03537f0f36802c5544a163a8b96dad5d0` |
| `requirements.txt` | `c0debb5cc21468144cc7879f025765ec0a7773426e865938e1fabb963a769d91` |

## KẾT LUẬN VÀ ĐỀ XUẤT CHỈNH SỬA

Bản cục bộ vượt qua **9 test gốc và 31 test bổ sung**, với công thức X/Y đối xứng quanh tâm ROI và thứ tự/dấu được giữ qua định dạng TCP đến phép cộng RAPID. Đây là xác nhận offline trong phạm vi đã kiểm, chưa xác nhận độ chính xác vật lý hay an toàn chuyển động ABB.

Ưu tiên làm rõ/sửa chính sách HOLD và kiểm ROI trước EMA nếu yêu cầu dừng nhận mục tiêu ngay khi cổ tay thô ra ngoài; tiếp theo đồng bộ README/WorkObject/giới hạn, tăng validation và test tích hợp, rồi hoàn thiện framing TCP. Tool, WorkObject, tâm, hướng trục và hành vi khi đang MoveL cần kiểm chứng với môi trường ABB phù hợp trước khi vận hành. **Chưa sửa mã nguồn chính, chưa bật camera và chưa kết nối robot; mọi chỉnh sửa chương trình hoặc kích hoạt điều khiển phải chờ người dùng phê duyệt.**
