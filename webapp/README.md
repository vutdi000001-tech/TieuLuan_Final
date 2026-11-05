# 🚀 NỀN TẢNG THỰC NGHIỆM & DỰ BÁO AI (WEBAPP)

Webapp được thiết kế hiện đại, tinh gọn và chuẩn mực, tập trung vào **4 họ mô hình cốt lõi**:

---

## 📚 Các Họ Mô Hình Tích Hợp Trong Webapp

1. **Mạng Nơ-ron Tích Chập (CNN - 1D & 2D)**:
   - *Deep 1D-CNN*: Trích xuất tương tác đặc trưng cục bộ y sinh (Glucose, HbA1c, BMI).
   - *Toán tử Tích chập 2D*: Mô phỏng bộ lọc biên cạnh Sobel ngang và dọc qua ma trận không gian 3x3.

2. **Mạng Nơ-ron Hồi Quy (RNN & LSTM Chuỗi Thời Gian)**:
   - Dự báo xu hướng tài chính: **Giá Vàng SJC**, **VN-Index** (HOSE), và **Cổ phiếu Amazon**.
   - Hỗ trợ các kiến trúc: `Vanilla LSTM`, `Stacked LSTM`, `Bidirectional LSTM`, `SimpleRNN`.
   - Trực quan hóa tương tác thời gian thực bằng **Chart.js** (30 phiên quá khứ + dự báo 7 phiên tương lai).

3. **Đồ Thị Tri Thức & RAG (Knowledge Graph & Vector Search)**:
   - Cơ chế truy vấn lai (Hybrid Retrieval): Kết hợp Bộ ba Đồ thị Tri thức (Symbolic Triples) và Truy xuất Vector (TF-IDF Cosine Similarity) trên 3 miền: Giá vàng SJC, Chứng khoán, Y tế & Tiêu chuẩn ADA.
   - Trích xuất căn cứ và trích dẫn tài liệu nguồn rõ ràng, minh bạch.

4. **Hệ Đa Agent (Multi-Agent System - MAS)**:
   - **Hội đồng 5 Diagnostic Agents**: Bỏ phiếu đồng thuận theo trọng số AUC (`Weighted Consensus`).
   - **Chỉ số bất đồng tập thể (`Disagreement Index`)**: Tự động phát hiện trường hợp nghi ngờ hoặc ca khó.
   - **Rule Engine Agent**: Thẩm định lâm sàng độc lập theo tiêu chuẩn ADA / WHO.
   - **Lifestyle Advisor & Safety Validator Agents**: Cá nhân hóa chế độ dinh dưỡng, vận động và đảm bảo an toàn y tế.

---

## 🛠️ Hướng Dẫn Chạy Cục Bộ (Local)

Từ thư mục `TieuLuanFinal`:
```bash
python run_webapp.py
```
Hoặc từ thư mục `webapp`:
```bash
cd webapp
python app.py
```
Truy cập: **`http://localhost:5000`**

---

## ☁️ Triển Khai Lên Render.com

- **Build Command**: `pip install -r webapp/requirements.txt`
- **Start Command**: `gunicorn webapp.app:app`
