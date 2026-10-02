# Lab 16 — Cloud AI Infrastructure (GCP)

Bài làm CPU + LightGBM. Kết quả chính nằm trong `benchmark_result.json`, báo cáo ngắn trong `report.md`.

## Bằng chứng

| Checkpoint | Thực hiện | Bằng chứng |
|---|---|---|
| CP0 | Fork cá nhân, nhánh `lab16-work`, starter commit `55539f67d7c78b43afe334a2ec3271c4bfdbbe2d`; GCP Billing và ADC hoạt động | Source trong `infra/`, provider lockfile |
| CP1 | Terraform tạo 16 tài nguyên; SSH qua IAP + OS Login vào VM CPU riêng | `logs/vm_metadata.json`, `screenshots/benchmark-command.png` |
| CP2 | Cài Python trong `/opt/lab16-venv`; tải dataset Kaggle gốc trên VM | Số dòng, cột, fraud, missing values trong JSON |
| CP3 | Train/evaluate/benchmark trên `ai-gpu-node`, `e2-medium`, `us-central1-a` | `benchmark.py`, JSON, `logs/benchmark_output.txt`, ảnh command/results |
| CP4 | Lấy mẫu `top` mỗi giây trong benchmark; ghi `free -h` và `ip -s link` trước/sau | `logs/resource_*.txt`, `screenshots/cpu-ram-network.png`, ảnh Billing |
| CP5 | Copy kết quả về laptop, đối chiếu SHA256, xóa hạ tầng và kiểm tra lại | `logs/SHA256SUMS`, `logs/cleanup_verification.json`, `logs/terraform_destroy.txt` |

## Cách đo và giải thích

- Dataset: Kaggle `mlg-ulb/creditcardfraud`, 284.807 dòng, 31 cột; `Class` là target, 492 giao dịch gian lận. Không đưa dataset hoặc Kaggle key vào Git.
- Chia stratified với seed 16: 170.883 train, 56.962 validation, 56.962 test. Lần chia thứ hai dùng 0,25 của 80% còn lại để lấy 20% tổng dữ liệu làm validation.
- LightGBM: tối đa 300 cây, learning rate 0,05, 2 threads; early stopping 20 vòng trên validation. Test chỉ dùng cho đánh giá cuối, không chọn ngưỡng hay số vòng bằng test.
- AUC dùng xác suất `predict_proba`; Accuracy/F1/Precision/Recall dùng ngưỡng cố định 0,5. Accuracy cao cần đọc cùng Recall/F1 vì dữ liệu mất cân bằng.
- Chỉ đo `pd.read_csv` cho load time và `.fit()` cho training time. Không gộp thời gian tạo VM hay download dataset.
- Inference đo riêng `predict_proba` trên pandas input sau warm-up; median của 50 lần cho 1 dòng và 10 lần cho batch 1.000 dòng. Throughput = 1.000 / thời gian batch tính bằng giây; không phải tốc độ của HTTP endpoint.
- `run_benchmark.sh` chạy Python và theo dõi cùng PID bằng `top`; exit code của Python được giữ nguyên. Ảnh command chụp `bash -x run_benchmark.sh`; ảnh results hiển thị log của đúng lần chạy cuối lúc `2026-10-02T09:42:54Z` (16:42:54 giờ Việt Nam).
- Ảnh tài nguyên chụp lại log `top` trong lần chạy và `free -h`/`ip -s link` ngay sau chạy. CPU tiến trình có thể đạt 200% trên 2 vCPU. Network là bộ đếm bytes tích lũy, không phải tốc độ tức thời.
- Có một lần chạy kiểm tra trước phiên chụp ảnh. JSON, logs, ảnh và báo cáo sử dụng cùng phiên cuối; khác biệt thời gian giữa hai lần chạy là bình thường. Không chọn kết quả tốt nhất trong nhiều lần chạy.
- Thư viện LightGBM hiện dùng phát cảnh báo deprecation cho `eval_set`, nhưng chạy thành công. `requirements-freeze.txt` ghi phiên bản thực tế để tái hiện.

## Hạ tầng và chi phí

VM không có public IP. Laptop vào VM qua IAP trên TCP 22, OS Login kiểm soát đăng nhập. Cloud Router/NAT cho phép VM tải thư viện và dataset từ internet. Load Balancer và health check port 8000 giữ theo starter; CPU benchmark không chạy HTTP server nên backend unhealthy là trạng thái dự kiến.

Startup script được chỉnh để dùng virtual environment, cài `libgomp1`, dừng khi lỗi và kiểm tra import trước khi báo hoàn tất. Mã triển khai thực tế cùng lockfile nằm trong `infra/` và `../terraform-gcp/`.

Ảnh Billing ngày 02/10/2026 được lấy từ Google Cloud Console, nhóm theo Service và kiểm tra project `project-b254970d-4ec2-4509-920`. Dashboard chưa có dữ liệu cho ngày này; ₫0 hiển thị không chứng minh lab miễn phí. Không dùng forecast hay trial credit còn lại làm chi phí lab. Project dùng chung với VM `voya-dev-01`, vì vậy chi phí tổng project cập nhật sau này cũng không thể quy hết cho lab. Tài nguyên có phí gồm VM, disk, NAT/IP, Load Balancer và traffic. Xem `logs/billing_observation.json` để biết thời điểm quan sát.

## Tái hiện trên VM đã chuẩn bị môi trường và dataset

```bash
cd ~/ml-benchmark
bash run_benchmark.sh
python3 -m json.tool benchmark_result.json
```

Để chỉ chạy model, kích hoạt `source /opt/lab16-venv/bin/activate` rồi chạy `python3 benchmark.py`; wrapper còn đặt metadata GCP và thu log tài nguyên.

`logs/SHA256SUMS` được tính trên VM cho code và JSON rồi kiểm tra với bản tải về. Terraform state/plan, `.terraform/`, Kaggle key và bộ dữ liệu được giữ ngoài gói nộp.
