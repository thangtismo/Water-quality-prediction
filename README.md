# Water-quality-prediction
Dự đoán chất lượng nước bằng các mô hình học sâu

 
Giới thiệu

Dự án này nhằm mục tiêu xây dựng một hệ thống phân loại chất lượng nước dựa trên các chỉ số hóa học và vật lý trong mẫu. Sử dụng các mô hình học máysâu khác nhau, nhóm thực hiện đánh giá độ chính xác và hiệu quả của từng thuật toán để lựa chọn mô hình phù hợp nhất.

📊 Xử lý dữ liệu

- **Tiền xử lý**:
  - Kiểm tra và xử lý giá trị thiếu
  - Chuẩn hóa/chuẩn bị dữ liệu (nếu có)
  - Phân chia dữ liệu huấn luyện và kiểm tra bằng `train_processed` và `test_processed`
  

🧠 Mô hình sử dụng

- LSTM
- GRU
- Hybird GRU + LSTM

🎯 Đánh giá mô hình

Sử dụng kỹ thuật xác thực chéo
- Sử dụng ma trận nhầm lẫn
- Đo lường hiệu quả qua độ chính xác (accuracy_score)
- Trực quan hóa kết quả bằng heatmap
