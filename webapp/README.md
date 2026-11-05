# 🚀 WEBAPP TIỂU LUẬN FINAL - CỔNG THỰC NGHIỆM AI ĐA NĂNG

Webapp được xây dựng tương tự như kiến trúc của **Tiểu Luận 1**, nâng cấp toàn diện để tích hợp **đầy đủ 6 chương nghiên cứu** của **Tiểu Luận Final**:

---

## 📚 Tổng Quan Các Chương Tích Hợp Trong Webapp

1. **Chương 1: Lịch Sử AI & Perceptron / MLP (Ranh Giới XOR)**:
   - Mô phỏng các hàm logic `AND`, `OR`, `XOR`.
   - Đối sánh thực nghiệm giữa Perceptron đơn lớp (bất lực trước XOR theo định lý Minsky-Papert 1969) và Multi-Layer Perceptron (MLP giải quyết phi tuyến hoàn hảo).

2. **Chương 2: Kỹ Thuật Học Máy Kinh Điển (Classical Machine Learning)**:
   - *Phân loại Y tế*: Dự đoán nguy cơ đái tháo đường (`diabetes_prediction_dataset.csv`) với Logistic Regression, Random Forest, XGBoost, SVM.
   - *Hồi quy Bất động sản*: Định giá nhà đất Việt Nam (`vietnam_housing_dataset.csv`) với Linear, Ridge, Random Forest Regressor, XGBoost Regressor.

3. **Chương 3: Mạng Nơ-ron Tích Chập (CNN - 1D & 2D)**:
   - *Deep 1D-CNN*: Trích xuất tương tác đặc trưng cục bộ y sinh.
   - *Toán tử Tích chập 2D*: Mô phỏng bộ lọc biên cạnh Sobel ngang và dọc.

4. **Chương 4: Mạng Hồi Quy Chuỗi Thời Gian (RNN & LSTM)**:
   - Dự báo chỉ số chứng khoán **VN-Index** (HOSE), **Giá Vàng SJC**, và **Cổ phiếu Amazon**.
   - Hỗ trợ các kiến trúc: `SimpleRNN`, `Vanilla LSTM`, `Stacked LSTM`, `Bidirectional LSTM`.
   - Vẽ biểu đồ tương tác thời gian thực bằng **Chart.js** (chuỗi 30 phiên quá khứ + dự báo 7-10 phiên tương lai).

5. **Chương 5: Knowledge Graph & RAG (Hỏi Đáp Lai Đa Miền)**:
   - Truy vấn có căn cứ nguồn từ Đồ thị tri thức (Symbolic KG Triples) kết hợp Vector Search (TF-IDF Cosine Similarity) trên 3 miền: Chứng khoán, Giá vàng SJC, Y tế & Tiểu đường.

6. **Chương 6: Hệ Đa Agent (Multi-Agent System Cho Tiểu Đường)**:
   - Hội đồng 5 Diagnostic Agents bỏ phiếu đồng thuận theo trọng số AUC (**Weighted Consensus**).
   - Đo lường chỉ số bất đồng (**Disagreement Index / Variance**) để phát hiện "Ca khó".
   - **Rule Engine Agent**: Thẩm định lâm sàng độc lập theo tiêu chuẩn ADA/WHO.
   - **Lifestyle Advisor & Safety Validator Agents**: Đưa ra hướng dẫn dinh dưỡng, vận động cá nhân hóa và bảo đảm an toàn y khoa.

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

## ☁️ Hướng Dẫn Deploy Lên Render.com

1. Đẩy mã nguồn repo lên GitHub.
2. Trên dashboard Render (`dashboard.render.com`), chọn **New +** -> **Web Service**.
3. Chọn repository và Render sẽ tự nhận diện cấu hình từ `render.yaml`:
   - **Build Command**: `pip install -r webapp/requirements.txt`
   - **Start Command**: `gunicorn webapp.app:app`
4. Bấm **Deploy Web Service** và hoàn tất!
