# 🎓 TIỂU LUẬN TỐT NGHIỆP: NGHIÊN CỨU & XÂY DỰNG HỆ THỐNG TRÍ TUỆ NHÂN TẠO TOÀN DIỆN
## Từ Cơ Bản Machine Learning, Deep Learning (CNN, RNN/LSTM), Knowledge Graph & RAG Đến Hệ Đa Agent (Multi-Agent System) Cho Bài Toán Y Tế

---

## 📌 Giới Thiệu Chung
Dự án tổng hợp toàn bộ các kết quả nghiên cứu lý thuyết, thực nghiệm mã nguồn và hệ thống thực thi thực tế của **Tiểu Luận Final**. 

Dự án bao gồm:
1. **7 Notebooks Jupyter hoàn chỉnh**: Có sẵn dữ liệu thực tế và đã chạy lưu đầy đủ kết quả trực quan hóa, biểu đồ chuyên sâu.
2. **Cổng Webapp Thực Nghiệm Đa Năng (Full-Stack Research Portal)**: Tương tự như kiến trúc Webapp Tiểu Luận 1, tích hợp toàn bộ 6 chương nghiên cứu với giao diện tương tác trực quan thời gian thực, sẵn sàng deploy lên **Render.com**.

---

## 📁 Cấu Trúc Thư Mục Dự Án
```text
TieuLuanFinal/
├── 00_Mo_Dau.ipynb                             # Mở đầu: Tổng quan kiến trúc & lộ trình nghiên cứu
├── 01_Chuong1_Lich_Su_Phat_Trien_AI.ipynb       # Chương 1: Lịch sử AI, Perceptron vs MLP (Ranh giới XOR)
├── 02_Chuong2_Cac_Ky_Thuat_ML_Co_Ban.ipynb      # Chương 2: Classical ML (Phân loại Tiểu đường & Hồi quy Nhà đất)
├── 03_Chuong3_CNN.ipynb                        # Chương 3: Mạng tích chập CNN (1D Feature Interaction & 2D Sobel)
├── 04_Chuong4_RNN_LSTM.ipynb                   # Chương 4: Dự báo chuỗi thời gian (VN-Index, Vàng SJC, Amazon)
├── 05_Chuong5_Knowledge_Graph_va_RAG.ipynb     # Chương 5: Đồ thị tri thức (KG) & RAG Đa miền
├── 06_Chuong6_He_Da_Agent.ipynb                # Chương 6: Hệ Đa Agent chẩn đoán và tư vấn tiểu đường
│
├── webapp/                                     # Webapp tương tác toàn diện 6 chương
│   ├── app.py                                  # Backend Flask API xử lý logic 6 chương
│   ├── static/                                 # Static assets (style.css, script.js, Chart.js)
│   ├── templates/index.html                    # Giao diện người dùng hiện đại, responsive
│   ├── requirements.txt                        # Thư viện phụ thuộc cho webapp
│   ├── Procfile                                # Cấu hình chạy trên Render/Heroku
│   └── render.yaml                             # Blueprint Infrastructure-as-Code cho Render
│
├── datasets/ (Tập dữ liệu thực tế)
│   ├── diabetes_prediction_dataset.csv         # 100,000 hồ sơ bệnh nhân tiểu đường
│   ├── vietnam_housing_dataset.csv             # Dữ liệu thị trường bất động sản Việt Nam
│   ├── HOSE_DLY_VNINDEX1D.csv                  # Chuỗi thời gian chỉ số VN-Index (HOSE)
│   ├── vietnam_gold_price.csv                  # Chuỗi thời gian giá vàng SJC
│   ├── gold_price.csv                          # Chuỗi thời gian giá vàng thế giới
│   ├── AMZN.csv                                # Chuỗi thời gian cổ phiếu Amazon
│   └── Womens Clothing E-Commerce Reviews.csv  # Dữ liệu xử lý ngôn ngữ tự nhiên
│
├── run_webapp.py                               # Script khởi chạy webapp nhanh ở thư mục gốc
├── Procfile                                    # Procfile cấp root phục vụ deploy Render
├── render.yaml                                 # Blueprint cấp root cho Render Web Service
├── requirements.txt                            # Requirements cấp root
└── README.md                                   # Tài liệu hướng dẫn sử dụng
```

---

## 🌟 Tổng Quan 6 Chương Nghiên Cứu Tích Hợp Trên Webapp

| Chương | Chủ đề nghiên cứu | Mô hình / Kỹ thuật chính | Minh họa / Tương tác trên Webapp |
| :--- | :--- | :--- | :--- |
| **Chương 1** | Lịch Sử AI & Ranh Giới XOR | Perceptron vs Multi-Layer Perceptron (MLP) | Kiểm thử các hàm logic AND, OR, XOR; chứng minh trực quan ranh giới phi tuyến |
| **Chương 2** | Classical Machine Learning | Logistic Regression, Random Forest, XGBoost, SVM | Dự đoán nguy cơ tiểu đường (Classification) & Định giá nhà đất Việt Nam (Regression) |
| **Chương 3** | Convolutional Neural Network | Deep 1D-CNN & Toán tử Sobel 2D | Trích xuất tương tác đặc trưng y sinh & ma trận tích chập phát hiện biên cạnh |
| **Chương 4** | Recurrent Neural Network | SimpleRNN, Vanilla LSTM, Stacked LSTM, BiLSTM | Dự báo xu hướng giá VN-Index, Vàng SJC, Amazon; vẽ đồ thị chuỗi thời gian (Chart.js) |
| **Chương 5** | Knowledge Graph & RAG | Symbolic Knowledge Graph + TF-IDF Cosine Vector RAG | Hỏi đáp thông minh đa miền (Chứng khoán, Vàng, Y tế) kèm trích dẫn nguồn bằng chứng |
| **Chương 6** | Hệ Đa Agent (Multi-Agent System) | Hội đồng 5 Agents + Rule Engine + Safety/Lifestyle | Bỏ phiếu đồng thuận theo trọng số AUC, đo Disagreement Index, phân tích ca khó |

---

## 🚀 Hướng Dẫn Khởi Chạy Webapp Cục Bộ (Local)

1. Cài đặt môi trường và các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

2. Khởi chạy Webapp:
```bash
python run_webapp.py
```
*Hoặc:*
```bash
cd webapp
python app.py
```

3. Mở trình duyệt và truy cập: **`http://localhost:5000`**

---

## ☁️ Hướng Dẫn Triển Khai Lên Render (Deploy on Render.com)

Dự án đã được cấu hình chuẩn mực với `render.yaml` và `Procfile`:
1. Đẩy mã nguồn lên kho chứa GitHub: `https://github.com/vutdi000001-tech/TieuLuan_Final.git`.
2. Truy cập [Render Dashboard](https://dashboard.render.com).
3. Nhấp chọn **New +** -> **Web Service**.
4. Kết nối với repository GitHub `TieuLuan_Final`.
5. Render sẽ tự động phát hiện cấu hình từ `render.yaml`:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r webapp/requirements.txt`
   - **Start Command**: `gunicorn webapp.app:app`
6. Nhấp **Deploy Web Service** và nhận URL ứng dụng sau khoảng 1-2 phút!

---

## 📊 Về Các File Jupyter Notebook
Tất cả 7 notebooks từ `00_Mo_Dau.ipynb` đến `06_Chuong6_He_Da_Agent.ipynb` đều đã được thực thi hoàn tất trên môi trường Jupyter Notebook (`http://localhost:8888`), chứa đầy đủ:
- Bảng biểu chỉ số đánh giá (Accuracy, Precision, Recall, F1, ROC-AUC, RMSE, MAE, R²).
- Các biểu đồ Heatmap Confusion Matrix, Đường cong ROC, Phân bố sai số, Feature Importance, Biểu đồ động thái hội đồng Agent.
