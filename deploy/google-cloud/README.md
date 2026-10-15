# Google Cloud Free Trial: bản demo CPU

Đích triển khai: web tìm kiếm công khai; benchmark và tài liệu nghiên cứu giữ riêng.
Đây là cấu hình chuẩn bị triển khai. Chỉ coi là đã deploy sau khi kiểm tra bằng URL HTTPS thật.
Trạng thái chuẩn bị: kiểm thử Python và kiểm tra cú pháp Compose đã chạy; build image/container
Linux và triển khai Google Cloud vẫn cần kiểm chứng khi Docker Engine/VM sẵn sàng.

## Cấu hình dự kiến và chi phí

- Compute Engine `e2-standard-2`: 2 vCPU, 8 GB RAM; Ubuntu 24.04 x86_64.
- Khu vực Singapore, zone `asia-southeast1-b`; ổ `pd-balanced` 50 GB.
- Một máy chạy Streamlit CPU và Caddy. Không GPU, database, load balancer hoặc autoscaling.
- Chi phí VM, disk, IP và băng thông trừ vào credit trial. Đây không phải gói VM miễn phí vĩnh viễn.
- Kiểm tra Billing đang là **Free Trial**, còn $300 credit và thời hạn; không nâng cấp tài khoản trả phí.
- Xem cost estimate theo đúng region trước khi tạo. Đặt budget alert $25 cho dự án demo;
  cảnh báo ngân sách không phải giới hạn chi tiêu cứng. Không mua tài nguyên ngoài trial.
- Dự kiến dùng một tuần. Có thể đặt lịch **STOP sau 192 giờ** để chừa thời gian cài đặt;
  việc dừng vẫn giữ disk và không chấm dứt mọi chi phí lưu trữ. Không tự động DELETE dữ liệu.

Nguồn: [Free Trial](https://cloud.google.com/signup-faqs),
[E2](https://docs.cloud.google.com/compute/docs/general-purpose-machines),
[lịch dừng VM](https://docs.cloud.google.com/compute/docs/instances/limit-vm-runtime).

## 1. Chuẩn bị bộ tài nguyên trên máy nguồn

Cài dependency theo README chính. Cấu hình `configs/paths.yaml` trỏ đúng dữ liệu nguồn.
Nếu dùng `--paths` từ thư mục code khác, dùng cấu hình có đường dẫn tuyệt đối.
Model SigLIP-384 và tokenizer phải có sẵn trong cache hoặc thư mục weights trước bước này.

```powershell
python tools/prepare_assets.py C:/Temp/n3-assets --index siglip384__whole --index siglip384__t3x2o20
python tools/prepare_assets.py C:/Temp/n3-assets --verify
tar -cf C:/Temp/n3-assets.tar -C C:/Temp/n3-assets .
```

Tool chỉ lấy ảnh xuất hiện trong hai lookup, metadata tương ứng, ba file cần thiết của mỗi
chỉ mục, weights và tokenizer. Không lấy nhãn, `vectors.npy`, kết quả benchmark, báo cáo hoặc
token đăng nhập Hugging Face. Không sửa backend, vector hoặc metadata nguồn của chỉ mục.
Mọi file được đối chiếu SHA256; destination đã tồn tại sẽ bị từ chối.

Cấu trúc pack:

```text
n3-assets/
  data/images/100k/train/*.jpg
  meta/images.parquet
  indexes/siglip384__whole/{index.faiss,lookup.parquet,meta.json}
  indexes/siglip384__t3x2o20/{index.faiss,lookup.parquet,meta.json}
  weights/ViT-B-16-SigLIP-384__webli.safetensors
  tokenizer-cache/hub/models--.../{refs,snapshots}/...
  asset-manifest.json
```

Giữ pack ngoài Git. Container đọc assets ở chế độ read-only và dùng model/tokenizer offline.
Việc lựa chọn bộ ảnh để công khai phải phù hợp quyền sử dụng dữ liệu của dự án.

## 2. Kiểm tra Docker trước khi đưa lên cloud

```powershell
$env:N3_ASSETS_DIR = 'C:/Temp/n3-assets'
docker compose config --quiet
docker compose build
docker compose up -d
docker compose ps
docker compose logs --tail 50 search
```

Mở `http://localhost:8503`, kiểm tra tìm bằng câu, tìm bằng ảnh, tìm ảnh giống, quay lại và So sánh.
Model và hai chỉ mục giữ đúng phiên bản đã dùng cục bộ; `N3_IMAGE_BACKEND=torch_cpu`
chỉ đổi backend của ảnh truy vấn. Không cần Intel GPU hoặc OpenVINO trên VM.
Healthcheck xác nhận server còn sống; không thay thế kiểm thử truy vấn thật.

## 3. Tạo máy sau khi trial đã kích hoạt

Chạy từng bước trong **Google Cloud Shell** của project dành riêng cho demo.
Thay `YOUR_PROJECT_ID` bằng Project ID thật. Không sao chép credential vào repository.
Lệnh bên dưới tạo tài nguyên cloud, vì vậy chỉ chạy sau khi đã xem Billing/cost estimate.

```bash
export PROJECT_ID=YOUR_PROJECT_ID
export ZONE=asia-southeast1-b
gcloud config set project "$PROJECT_ID"
gcloud services enable compute.googleapis.com iap.googleapis.com
gcloud compute networks create n3-demo-net --subnet-mode=custom
gcloud compute networks subnets create n3-demo-subnet --network=n3-demo-net \
  --region=asia-southeast1 --range=10.42.0.0/24
gcloud compute firewall-rules create n3-demo-iap-ssh --network=n3-demo-net \
  --allow=tcp:22 --source-ranges=35.235.240.0/20 --target-tags=n3-demo
gcloud compute instances create n3-demo --zone="$ZONE" --machine-type=e2-standard-2 \
  --image-family=ubuntu-2404-lts-amd64 --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB --boot-disk-type=pd-balanced --boot-disk-auto-delete \
  --subnet=n3-demo-subnet --tags=n3-demo --no-service-account --no-scopes \
  --max-run-duration=192h --instance-termination-action=STOP \
  --labels=app=n3-demo
```

SSH qua [IAP](https://docs.cloud.google.com/iap/docs/using-tcp-forwarding).
Tài khoản triển khai cần quyền Compute/SSH và IAP tunnel phù hợp; nếu Console báo thiếu quyền,
cấp đúng quyền cho chính tài khoản vận hành, không tạo key dịch vụ dài hạn.

```bash
gcloud compute ssh n3-demo --zone="$ZONE" --tunnel-through-iap
```

Trong VM:

```bash
sudo apt-get update
sudo apt-get install -y git python3
git clone https://github.com/phanhieunghia353/n3-07b-data-engine.git ~/n3-web
cd ~/n3-web
git rev-parse HEAD
sudo bash deploy/google-cloud/install-docker.sh
```

Ghi lại commit đã build và giữ nguyên commit đó trong tuần demo.
Script cài Docker từ [APT repository chính thức](https://docs.docker.com/engine/install/ubuntu/).
Không dùng tính năng `Deploy container`/`create-with-container` cũ của Compute Engine.

## 4. Chuyển pack qua SSH và kiểm tra nội bộ

Từ máy có gcloud CLI và file tar, hoặc dùng SCP client qua SSH được cấp quyền:

```powershell
gcloud compute scp C:/Temp/n3-assets.tar n3-demo:~/n3-assets.tar --project=YOUR_PROJECT_ID --zone=asia-southeast1-b --tunnel-through-iap
```

File tar chỉ chứa pack vừa tạo. Không upload toàn bộ workspace hoặc thư mục cache cá nhân.
Trong VM:

```bash
mkdir -p ~/n3-assets
tar -xf ~/n3-assets.tar -C ~/n3-assets
cd ~/n3-web
python3 tools/prepare_assets.py ~/n3-assets --verify
export N3_ASSETS_DIR="$HOME/n3-assets"
sudo env N3_ASSETS_DIR="$N3_ASSETS_DIR" docker compose build
sudo env N3_ASSETS_DIR="$N3_ASSETS_DIR" docker compose up -d
sudo docker compose ps
curl --fail http://127.0.0.1:8503/_stcore/health
```

Để xem qua trình duyệt trước khi công khai, tạo SSH port forward từ máy của bạn:

```powershell
gcloud compute ssh n3-demo --project=YOUR_PROJECT_ID --zone=asia-southeast1-b --tunnel-through-iap -- -N -L 8504:localhost:8503
```

Mở `http://localhost:8504`. Kiểm tra truy vấn thật và xem `sudo docker stats --no-stream`.
Chỉ tiếp tục nếu không có OOM, đủ dung lượng disk, cả ba tab hoạt động và kết quả phù hợp.

## 5. Công khai qua HTTPS sau khi kiểm tra

Dùng domain của bạn trỏ vào IP VM. Nếu chưa có domain, có thể thử hostname miễn phí
`<IP>.sslip.io` cho tuần demo. Đây là DNS do bên thứ ba cung cấp; không cam kết tên miền/SLA.
Kiểm tra DNS trước khi xin chứng chỉ. Cần giữ IP VM ổn định trong suốt đợt demo.
Không dừng/khởi động lại VM nếu dùng IP ephemeral và cần giữ nguyên URL.

Trong Cloud Shell:

```bash
gcloud compute instances describe n3-demo --zone="$ZONE" \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
gcloud compute firewall-rules create n3-demo-web --network=n3-demo-net \
  --allow=tcp:80,tcp:443 --source-ranges=0.0.0.0/0 --target-tags=n3-demo
```

Trong VM, thay hostname thật vào file cấu hình local:

```bash
cd ~/n3-web
printf 'N3_ASSETS_DIR=%s\nN3_PUBLIC_HOST=%s\n' "$HOME/n3-assets" 'YOUR_PUBLIC_HOST' > .env
sudo docker compose -f compose.yaml -f deploy/compose.cloud.yaml up -d
sudo docker compose -f compose.yaml -f deploy/compose.cloud.yaml logs --tail 50 gateway
```

Caddy cấp HTTPS và chuyển tiếp WebSocket cho Streamlit. Cổng 8501/8503 không mở trên firewall.
Mở URL trên mạng khác và kiểm tra lại tìm kiếm, ảnh lớn, upload ảnh, theme và CSV.
Model có thể nạp chậm ở lượt tìm đầu; cần chạy thử một lượt trước buổi trình diễn.
Đo lại độ trễ và mức RAM khi có 2–3 phiên; chưa cam kết tải nhiều người dùng.
Pin digest image Python/Caddy sau build đã kiểm chứng nếu cần tái tạo môi trường chính xác.

## 6. Kết thúc tuần demo

- Tải lại cấu hình cần giữ; pack gốc và code vẫn có bản trên máy.
- Chỉ STOP nếu muốn tiếp tục dùng lại disk. Disk và IP giữ lại vẫn có thể phát sinh chi phí.
- Khi quyết định gỡ hẳn, xóa VM và boot disk của **n3-demo** qua Console hoặc lệnh dưới đây.
  Đây là bước hủy dữ liệu cloud; không chạy khi vẫn cần dữ liệu trên VM.

```bash
gcloud compute instances delete n3-demo --zone="$ZONE" --delete-disks=boot
gcloud compute firewall-rules delete n3-demo-web n3-demo-iap-ssh
gcloud compute networks subnets delete n3-demo-subnet --region=asia-southeast1
gcloud compute networks delete n3-demo-net
```

Kiểm tra Disks, External IP addresses, Snapshots và Billing để bảo đảm không còn tài nguyên demo
ngoài ý muốn. Không xóa tài nguyên của project khác. Không chuyển sang paid account để duy trì demo.
