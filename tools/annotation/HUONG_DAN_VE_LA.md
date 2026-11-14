# Hướng dẫn hiệu chỉnh mask lá cho tập train

Tool này dùng để hiệu chỉnh mask host cho các ảnh train mà proposal tự động chưa đạt yêu cầu. Mask ban đầu chỉ là gợi ý; người gán nhãn phải kiểm tra toàn bộ biên lá.

## Chạy tool

Yêu cầu Python 3.10 trở lên. Mở PowerShell tại thư mục gói annotation rồi chạy:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_host_train.ps1
```

Lần đầu script tạo `.venv-host-gold` và cài Pillow/NumPy. Trình duyệt mở tại địa chỉ ghi trong `host_train_annotation_config.json`. Dừng bằng `Ctrl+C`; tiến độ được lưu sau mỗi ảnh.

## Quy tắc vẽ

Vẽ toàn bộ chiếc lá hoặc quả chứa vùng bệnh, kể cả phần mô còn xanh.

- Không vẽ nền, đất, tay, cành hoặc lá không có vùng bệnh.
- Không suy diễn phần bị che hay nằm ngoài ảnh.
- Nếu nhiều lá có bệnh, vẽ tất cả các lá đó.
- Nếu nhãn triệu chứng nằm ngoài host thật, đánh dấu nghi ngờ nhãn và ghi chú.

Phím `B` để tô, `E` để xóa, `Ctrl+Z` để hoàn tác. Chọn “Duyệt và lưu” khi hoàn tất; ảnh “chưa chắc chắn” không được đưa vào manifest cuối.

## Kiểm tra tiến độ

```powershell
powershell -ExecutionPolicy Bypass -File .\status_host_train.ps1
```

## Ghép hai gói đã hoàn tất

Chạy từ thư mục `tools/annotation`, truyền các đường dẫn thật trên máy của bạn:

```powershell
python .\merge_host_labels.py --manifest C:\path\to\pseudo_host_manifest.csv C:\path\to\host_train_A C:\path\to\host_train_B --out C:\path\to\manual_host_masks.zip
```

Chỉ mask có `review_status=approved` được ghép. Kiểm tra thống kê theo lớp và SHA-256 mà lệnh in ra trước khi đưa bundle vào pipeline paper.
