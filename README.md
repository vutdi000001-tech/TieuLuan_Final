# 🎓 HỆ THỐNG TRÍ TUỆ NHÂN TẠO & ĐA AGENT (AI & MULTI-AGENT SYSTEM)
## Mạng Tích Chập (CNN), Mạng Hồi Quy (RNN/LSTM), Knowledge Graph + RAG & Hệ Đa Agent (MAS)

---

## 📌 Giới Thiệu Chung
Dự án tổng hợp toàn bộ các kết quả nghiên cứu và hệ thống thực thi thực tế của **Tiểu Luận Final**:
1. **Kho Dữ Liệu & 7 Notebooks Jupyter**: Lưu trữ đầy đủ kết quả thực thi và đồ thị trực quan hóa chuyên sâu.
2. **Cổng Webapp Thực Nghiệm AI (Full-Stack AI Portal)**: Tập trung trực quan hóa 4 họ mô hình trọng tâm (CNN, RNN/LSTM, Knowledge Graph + RAG, Multi-Agent System), sẵn sàng triển khai trên **Render.com**.

---

## 📁 Cấu Trúc Thư Mục Dự Án
```text
TieuLuanFinal/
├── 00_Mo_Dau.ipynb                             # Mở đầu: Tổng quan kiến trúc & lộ trình nghiên cứu
├── 01_Chuong1_Lich_Su_Phat_Trien_AI.ipynb       # Lịch sử AI: Perceptron vs MLP (Ranh giới XOR)
├── 02_Chuong2_Cac_Ky_Thuat_ML_Co_Ban.ipynb      # Classical ML: Phân loại Tiểu đường & Hồi quy Nhà đất
├── 03_Chuong3_CNN.ipynb                        # Mạng tích chập CNN (1D Feature Interaction & 2D Sobel)
├── 04_Chuong4_RNN_LSTM.ipynb                   # Dự báo chuỗi thời gian (VN-Index, Vàng SJC, Amazon)
├── 05_Chuong5_Knowledge_Graph_va_RAG.ipynb     # Đồ thị tri thức (KG) & RAG Đa miền
├── 06_Chuong6_He_Da_Agent.ipynb                # Hệ Đa Agent chẩn đoán và tư vấn tiểu đường
│
├── webapp/                                     # Cổng webapp tương tác đa mô hình
│   ├── app.py                                  # Backend Flask API xử lý logic các mô hình
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

## 🌟 4 Họ Mô Hình Trọng Tâm Trên Webapp

| Mô Hình | Kỹ Thuật Chính | Minh Họa / Tương Tác Trên Webapp |
| :--- | :--- | :--- |
| **Mạng Tích Chập (CNN)** | Deep 1D-CNN & Toán tử Sobel 2D | Trích xuất tương tác đặc trưng y sinh & ma trận tích chập phát hiện biên cạnh |
| **Mạng Hồi Quy (RNN / LSTM)** | Vanilla LSTM, Stacked LSTM, BiLSTM, SimpleRNN | Dự báo xu hướng giá Vàng SJC, VN-Index, Amazon; vẽ đồ thị chuỗi thời gian (Chart.js) |
| **Knowledge Graph & RAG** | Symbolic Knowledge Graph + TF-IDF Cosine Vector RAG | Hỏi đáp thông minh đa miền (Vàng SJC, Chứng khoán, Y tế) kèm trích dẫn nguồn bằng chứng |
| **Hệ Đa Agent (MAS)** | Hội đồng 5 Agents + Rule Engine + Safety/Lifestyle | Bỏ phiếu đồng thuận theo trọng số AUC, đo Disagreement Index, thẩm định lâm sàng |

---

## 🚀 Hướng Dẫn Khởi Chạy Webapp Cục Bộ (Local)

1. Cài đặt môi trường:
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

## ☁️ Triển Khai Lên Render (Deploy on Render.com)

1. Mã nguồn đã được đồng bộ trên GitHub: `https://github.com/vutdi000001-tech/TieuLuan_Final.git`.
2. Trên [Render Dashboard](https://dashboard.render.com), chọn **New +** -> **Web Service**.
3. Kết nối với repository `vutdi000001-tech/TieuLuan_Final`.
4. Render sẽ tự động áp dụng blueprint từ `render.yaml`:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r webapp/requirements.txt`
   - **Start Command**: `gunicorn webapp.app:app`
5. Nhấp **Deploy Web Service** để hoàn tất.
