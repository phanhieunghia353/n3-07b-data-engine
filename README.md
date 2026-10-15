# N3-07B · Local image search

Web tìm ảnh bằng câu tiếng Anh hoặc ảnh mẫu, dùng CLIP/SigLIP và FAISS. Có bộ lọc, tìm ảnh tương tự, lịch sử quay lại, so sánh ảnh nguyên/cắt ô, xuất CSV và giao diện sáng/tối.

Repository này chỉ chứa mã chạy web localhost và kiểm thử nhỏ bằng dữ liệu tổng hợp. Báo cáo, paper, notebook, code train/eval, nhãn chấm và kết quả benchmark được quản lý riêng.

## Chạy trên máy của bạn

Cần Python 3.12. Mở terminal tại thư mục repository:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item configs/paths.example.yaml configs/paths.yaml
```

Trên Linux/macOS, kích hoạt bằng `source .venv/bin/activate` và sao chép cấu hình bằng `cp configs/paths.example.yaml configs/paths.yaml`.

Sửa `configs/paths.yaml` để trỏ đến bộ tài nguyên tìm kiếm cục bộ của bạn. Sau đó:

```powershell
python tools/check_assets.py
python -m streamlit run app/app.py
```

Truy cập `http://localhost:8501`. Trên Windows cũng có thể mở `chay_app.bat` sau khi đã cài dependency và cấu hình tài nguyên; nếu dùng môi trường ảo, chạy lệnh Python ở trên trong môi trường đó.

Khi chưa có cấu hình hoặc chỉ mục, ứng dụng hiện hướng dẫn thiết lập. Clone code không tự cung cấp bộ ảnh hoặc model để tìm kiếm thật.

## Tài nguyên cần chuẩn bị riêng

Ảnh, metadata, chỉ mục và trọng số không được đưa vào Git. Sử dụng bộ tài nguyên tương thích do bạn chuẩn bị hoặc được cung cấp qua kênh riêng phù hợp với quyền sử dụng dữ liệu.

```text
local-assets/
  images/...                 # ảnh JPEG/PNG, đường dẫn do metadata chỉ tới
  meta/images.parquet
  indexes/
    siglip384__whole/
      index.faiss
      lookup.parquet
      meta.json
    siglip384__t3x2o20/
      index.faiss
      lookup.parquet
      meta.json
  weights/                   # tuỳ chọn; xem phần model bên dưới
```

- `images.parquet`: có các cột `image_id`, `image_path`, `weather`, `scene`, `timeofday`. `image_path` tương đối với `data_root`.
- `lookup.parquet`: ánh xạ hàng vector qua `row_id`, `image_id`, `tile_idx`.
- `meta.json`: thông tin model/index, gồm `model_id`, `backend`, `n_tiles`, `n_vectors`, `open_clip_name`, `pretrained`, `preprocess`.
- Model phải khớp với model đã tạo chỉ mục; cấu hình encoder ở `configs/clip_models.yaml`, cấu hình ô cắt ở `configs/tiling.yaml`.
- Chức năng So sánh cần cả `whole` và `t3x2o20` của cùng model. Các chỉ mục có thể nằm ngoài repository bằng cách đổi `index_root`.
- `vectors.npy` và các file train/eval không cần thiết để chạy tìm kiếm.

Model ưu tiên file `{open_clip_name}__{pretrained}.safetensors` trong `clip_weights_dir`, rồi cache Hugging Face; nếu chưa có, thư viện tải pretrained weights/tokenizer ở lần dùng đầu. Muốn chạy offline cần chuẩn bị cả weights và tokenizer/cache trước.

Runtime mặc định dùng PyTorch CPU. Nếu chỉ mục ghi backend OpenVINO nhưng máy chưa có OpenVINO, ứng dụng thông báo và dùng CPU cho ảnh truy vấn. Muốn dùng tăng tốc Intel, cài thêm:

```powershell
python -m pip install -r requirements-openvino.txt
```

Để chạy trên máy chủ CPU mà không thử khởi tạo OpenVINO, đặt biến môi trường
`N3_IMAGE_BACKEND=torch_cpu`. Metadata và vector của chỉ mục vẫn được giữ nguyên.

## Chạy bằng Docker và Google Cloud

Docker sử dụng CPU, chạy bằng user không phải root và đọc gói tài nguyên riêng ở chế độ read-only.
Xem [hướng dẫn triển khai](deploy/google-cloud/README.md) để chuẩn bị pack có SHA256,
kiểm tra container cục bộ, triển khai Compute Engine và mở HTTPS sau khi kiểm tra nội bộ.
Ảnh, weights, tokenizer, cấu hình máy và file `.env` vẫn nằm ngoài Git và Docker build context.

## Sử dụng

- **Tìm bằng câu:** nhập tối đa 5 câu tiếng Anh, mỗi dòng một câu.
- **Tìm bằng ảnh:** tải ảnh hoặc nhập mã ảnh đã có trong bộ dữ liệu.
- **Tìm ảnh giống:** tìm tiếp từ một kết quả; ảnh mẫu được loại khỏi danh sách mới.
- Nhấp ảnh để xem lớn, nhấp lại trên ảnh để thu nhỏ; nút Toàn màn hình và Esc dùng khi cần.
- **Chi tiết** chứa ID, điểm tương đồng và ô cắt. Thông tin thời tiết/thời điểm được hiển thị bằng nhãn tiếng Việt; CSV giữ giá trị gốc.
- Nút sáng/tối ở đầu trang nhớ lựa chọn trong trình duyệt.
- Đổi bộ lọc cần bấm Tìm để cập nhật. Reload có thể tạo phiên mới; lịch sử không phải kho lưu trữ lâu dài.
- Lượt tìm đầu có thể mất thời gian nạp model. Thanh trạng thái hiển thị khi xử lý và chặn gửi yêu cầu trùng.

## Kiểm tra và xuất bộ mã sạch

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
python tools/export_runtime.py C:/Temp/n3-web-runtime
```

Các kiểm thử dùng ảnh và vector tổng hợp, không cần tài liệu nghiên cứu hoặc dữ liệu BDD100K. `public-files.txt` là danh sách chính xác các file được phép xuất. `.gitignore` dùng danh sách cho phép để tránh vô tình thêm báo cáo, kết quả, cấu hình máy hoặc tài nguyên lớn.

Quy tắc này áp dụng cho bản mã hiện tại và các lần commit sau. Việc bỏ file khỏi phiên bản mới không tự xoá nội dung từng có trong lịch sử Git.
