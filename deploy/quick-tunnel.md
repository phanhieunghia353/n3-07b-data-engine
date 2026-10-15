# Demo HTTPS miễn phí từ máy cá nhân

Cloudflare Quick Tunnel tạo URL HTTPS tạm thời cho web tìm kiếm trên máy này.
Không cần tài khoản Cloudflare, domain, thẻ thanh toán hoặc Docker.
Máy tính là server: cần cắm nguồn, giữ kết nối mạng và tránh gập nắp laptop.
Script giữ Windows không tự sleep trong thời gian demo; màn hình vẫn có thể tắt.
Khi dừng demo, script trả lại cơ chế sleep thông thường, không sửa power plan.

1. Cài Python/dependency và cấu hình `configs/paths.yaml` theo README chính.
2. Tải `cloudflared-windows-amd64.exe` từ [nguồn chính thức](https://developers.cloudflare.com/tunnel/downloads/).
   Đổi tên thành `cloudflared.exe`, đặt trong `.demo-online/` tại thư mục repository.
3. Mở `chay_demo_online.bat`. File `.demo-online/link.txt` chứa link HTTPS khi kết nối xong.
4. Mở `dung_demo_online.bat` khi muốn ngừng truy cập. Chỉ các tiến trình do script tạo bị dừng.

Có thể dùng một pack riêng và kiểm tra nội bộ trước:

```powershell
python deploy/quick_tunnel.py start --paths C:/Temp/paths.demo.yaml --preview
# Mở http://localhost:8510 và kiểm tra ba tab tìm kiếm.
python deploy/quick_tunnel.py publish
python deploy/quick_tunnel.py status
```

Script chạy ứng dụng trên cổng loopback 8510; benchmark không được đưa vào tunnel.
Nếu đường dẫn weights trỏ vào pack do `tools/prepare_assets.py` tạo, script dùng cả tokenizer offline.
Logs, trạng thái, binary và link nằm trong `.demo-online/`, được Git bỏ qua.
Nếu dùng cấu hình khác lúc khởi động bằng dòng lệnh, dùng lại đúng lệnh đó khi cần chạy lại.

Link chỉ hoạt động khi ứng dụng và tunnel đang chạy. Khởi động lại tunnel tạo hostname mới.
Mọi người có link có thể truy cập web. Quick Tunnel không có cam kết uptime;
đây là bản trình diễn tạm thời, không phải hosting có SLA hoặc địa chỉ cố định.
Không có cơ chế tự khởi động lại sau khi Windows reboot.

[Tài liệu và giới hạn Cloudflare Quick Tunnel](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/).
