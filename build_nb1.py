import nbformat as nbf

def create_nb1():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title Markdown
    cells.append(nbf.v4.new_markdown_cell("""# CHƯƠNG 1: LỊCH SỬ PHÁT TRIỂN TRÍ TUỆ NHÂN TẠO

---

## 1.1. Khởi Nguyên Của Trí Tuệ Nhân Tạo (1943 - 1956)

Lịch sử Trí tuệ Nhân tạo (Artificial Intelligence - AI) bắt đầu từ những nỗ lực mô hình hóa tư duy con người bằng các hệ thống toán học hình thức:

1. **Năm 1943 - Nơ-ron hình thức McCulloch-Pitts**:
   Warren McCulloch và Walter Pitts công bố công trình *"A Logical Calculus of the Ideas Immanent in Nervous Activity"*, đề xuất mô hình toán học đầu tiên của nơ-ron sinh học. Nơ-ron nhận tín hiệu đầu vào nhị phân $\{0, 1\}$, nhân với trọng số, và kích hoạt nếu tổng vượt qua một ngưỡng $\theta$ nhất định. Mô hình này chứng minh rằng các mạch nơ-ron có thể thực thi các phép toán logic cơ bản như AND, OR, NOT.

2. **Năm 1950 - Phép thử Turing (Turing Test)**:
   Alan Turing xuất bản bài báo kinh điển *"Computing Machinery and Intelligence"*, đặt ra câu hỏi nền tảng: *"Máy móc có thể tư duy được không?"* (Can machines think?). Turing đề xuất "Trò chơi bắt chước" (Imitation Game) – tiền thân của Phép thử Turing – nhằm đánh giá trí thông minh của máy dựa trên khả năng giao tiếp ngôn ngữ tự nhiên không thể phân biệt với con người.

3. **Năm 1956 - Hội thảo Dartmouth**:
   Hội thảo mùa hè tại Đại học Dartmouth do John McCarthy, Marvin Minsky, Nathaniel Rochester và Claude Shannon đồng tổ chức. Đây là mốc lịch sử chính thức khai sinh thuật ngữ **"Trí tuệ Nhân tạo" (Artificial Intelligence)** và xác lập AI thành một ngành khoa học độc lập.
"""))

    # Section 1.2: Perceptron
    cells.append(nbf.v4.new_markdown_cell(r"""## 1.2. Kỷ Nguyên Hoàng Kim Thứ Nhất và Mô Hình Perceptron (1957 - 1969)

Năm 1957, Frank Rosenblatt phát minh ra **Perceptron** tại Phòng thí nghiệm Hàng không Cornell. Đây là mô hình học máy đầu tiên có khả năng tự động học các trọng số từ dữ liệu thực nghiệm thông qua thuật toán lặp:

### 1.2.1. Cấu trúc toán học của Perceptron
Cho véc-tơ đầu vào $x = [x_1, x_2, \dots, x_d]^T \in \mathbb{R}^d$ và véc-tơ trọng số $w = [w_1, w_2, \dots, w_d]^T \in \mathbb{R}^d$, cùng hệ số chệch $b \in \mathbb{R}$.

Tổng tổ hợp tuyến tính:
$$z = w^T x + b = \sum_{j=1}^d w_j x_j + b$$

Hàm kích hoạt bậc thang (Heaviside step function):
$$\hat{y} = f(z) = \begin{cases} 1 & \text{nếu } z \ge 0 \\ 0 & \text{nếu } z < 0 \end{cases}$$

### 1.2.2. Thuật toán học Perceptron (Perceptron Learning Rule)
Với mỗi mẫu dữ liệu $(x_i, y_i)$ trong đó $y_i \in \{0, 1\}$:
1. Dự báo: $\hat{y}_i = f(w^T x_i + b)$.
2. Tính toán sai số: $e_i = y_i - \hat{y}_i$.
3. Cập nhật tham số theo hướng giảm thiểu sai số:
   $$w^{(t+1)} = w^{(t)} + \eta \cdot (y_i - \hat{y}_i) \cdot x_i$$
   $$b^{(t+1)} = b^{(t)} + \eta \cdot (y_i - \hat{y}_i)$$
   với $\eta \in (0, 1]$ là tốc độ học (learning rate).

### 1.2.3. Định lý hội tụ Perceptron (Novikoff, 1962)
Nếu tập dữ liệu $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ có thể phân tách tuyến tính bởi một siêu phẳng với lề phân cách $\gamma > 0$, tức tồn tại $w^*$ với $\|w^*\| = 1$ sao cho:
$$(2y_i - 1)(w^{*T} x_i + b^*) \ge \gamma, \quad \forall i$$
Thì thuật toán học Perceptron được đảm bảo hội tụ sau tối đa $k$ bước cập nhật:
$$k \le \left( \frac{R}{\gamma} \right)^2$$
trong đó $R = \max_{1 \le i \le N} \|x_i\|_2$.
"""))

    # Section 1.3: AI Winter I & XOR
    cells.append(nbf.v4.new_markdown_cell(r"""## 1.3. Mùa Đông AI Lần Thứ Nhất và Giới Hạn Của Perceptron (1970 - 1979)

Năm 1969, Marvin Minsky và Seymour Papert xuất bản cuốn sách mang tính bước ngoặt: *"Perceptrons: An Introduction to Computational Geometry"*. Cuốn sách đã chứng minh một cách chặt chẽ về mặt toán học giới hạn của Perceptron đơn lớp:

### 1.3.1. Rào cản hàm XOR (The XOR Barrier)
Perceptron đơn lớp chỉ có khả năng phân tách các tập dữ liệu **phân tách tuyến tính (Linearly Separable)**, trong đó siêu phẳng $w^T x + b = 0$ chia không gian thành hai nửa không gian riêng biệt.

Đối với hàm logic XOR (Exclusive OR):
| $x_1$ | $x_2$ | $y = x_1 \oplus x_2$ |
| :---: | :---: | :---: |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Hệ bất phương trình đối với siêu phẳng phân cách:
$$\begin{cases}
w_1(0) + w_2(0) + b < 0 \implies b < 0 \\
w_1(0) + w_2(1) + b \ge 0 \implies w_2 + b \ge 0 \\
w_1(1) + w_2(0) + b \ge 0 \implies w_1 + b \ge 0 \\
w_1(1) + w_2(1) + b < 0 \implies w_1 + w_2 + b < 0
\end{cases}$$

Cộng hai bất phương trình ở giữa:
$$(w_1 + b) + (w_2 + b) \ge 0 \implies w_1 + w_2 + 2b \ge 0$$
Mặt khác, từ bất phương trình thứ tư: $w_1 + w_2 + b < 0$, suy ra:
$$(w_1 + w_2 + 2b) - b < 0 \implies w_1 + w_2 + 2b < b$$
Do $b < 0$, ta có mâu thuẫn:
$$0 \le w_1 + w_2 + 2b < b < 0 \implies 0 < 0 \quad (\text{Vô nghiệm!})$$

Kết luận: Không tồn tại bất kỳ cặp trọng số $(w_1, w_2, b)$ nào để Perceptron đơn lớp có thể phân loại đúng bài toán XOR.

### 1.3.2. Mùa đông AI lần thứ nhất (AI Winter I)
Kết luận bi quan từ cuốn sách của Minsky & Papert, kết hợp với Báo cáo Lighthill (1973) tại Anh và việc cơ quan DARPA tại Mỹ cắt giảm mạnh ngân sách tài trợ cho nghiên cứu mạng nơ-ron, đã đẩy lĩnh vực AI vào giai đoạn thoái trào kéo dài một thập kỷ.
"""))

    # Section 1.4: Expert Systems to GenAI
    cells.append(nbf.v4.new_markdown_cell("""## 1.4. Các Giai Đoạn Tiếp Theo: Từ Hệ Chuyên Gia Đến Kỷ Nguyên Học Sâu

### 1.4.1. Sự phục hưng và Hệ chuyên gia (1980 - 1987)
- **Kiến trúc**: Chuyển hướng tiếp cận từ kết nối sinh học (Connectionism) sang trí tuệ biểu tượng (Symbolic AI). Xây dựng hệ thống dựa trên tri thức (Knowledge-Based Systems) gồm hai thành phần: **Cơ sở tri thức (Knowledge Base)** chứa hàng nghìn luật `IF-THEN` và **Động cơ suy diễn (Inference Engine)**.
- **Thành tựu tiêu biểu**: DENDRAL (phân tích cấu trúc phân tử hóa học), MYCIN (chẩn đoán bệnh nhiễm trùng huyết), XCON (cấu hình hệ thống máy tính VAX).
- **Hạn chế**: Điểm nghẽn thu thập tri thức chuyên gia (Knowledge Acquisition Bottleneck) và tính dễ vỡ (brittleness) khi gặp tình huống ngoại lệ.

### 1.4.2. Mùa đông AI lần thứ hai (1987 - 1993)
Sự sụp đổ của thị trường phần cứng chuyên dụng chạy ngôn ngữ LISP (LISP Machines) trước sự phát triển vượt bậc của máy tính cá nhân vi xử lý Intel và máy trạm Unix, dẫn đến làn sóng cắt giảm đầu tư thứ hai.

### 1.4.3. Kỷ nguyên Học máy Thống kê (1993 - 2011)
- Tiếp cận dựa trên dữ liệu và lý thuyết xác suất thống kê.
- Vladimir Vapnik hoàn thiện lý thuyết máy học thống kê và phát triển **Máy véc-tơ hỗ trợ (Support Vector Machines - SVM)** cùng kỹ thuật Kernel Trick để ánh xạ không gian phi tuyến.
- Leo Breiman phát minh thuật toán **Rừng ngẫu nhiên (Random Forest - 2001)**.
- Mạng Bayes, thuật toán Tăng cường độ dốc (Gradient Boosting).

### 1.4.4. Kỷ nguyên Học sâu và Tạo sinh (2012 - 2026)
- **Năm 2012 - AlexNet**: Alex Krizhevsky, Ilya Sutskever và Geoffrey Hinton giành chiến thắng áp đảo tại cuộc thi ImageNet, chứng minh sức mạnh của mạng tích chập sâu (Deep CNN) kết hợp năng lực tính toán song song của GPU.
- **Năm 2014 - 2016**: Sự trỗi dậy của mạng nơ-ron hồi quy LSTM trong xử lý ngôn ngữ và chuỗi thời gian; AlphaGo của Google DeepMind đánh bại nhà vô địch cờ vây thế giới.
- **Năm 2017 - Transformer**: Bài báo *"Attention Is All You Need"* (Vaswani et al.) đề xuất cơ chế Tự chú ý (Self-Attention), loại bỏ hạn chế tuần tự của RNN.
- **Năm 2020 - 2026 - Kỷ nguyên LLM & RAG**: Sự xuất hiện của các mô hình ngôn ngữ quy mô lớn (GPT-3, GPT-4, Llama) và giải pháp **RAG (Retrieval-Augmented Generation)** kết hợp Đồ thị tri thức (Knowledge Graph) giúp giải quyết triệt để vấn đề ảo giác và tăng cường tính tin cậy.
"""))

    # Code: Libraries
    cells.append(nbf.v4.new_code_cell("""# Thiết lập môi trường và nạp các thư viện toán học, trực quan hóa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.autolayout'] = True
"""))

    # Sơ đồ 1: Timeline lịch sử AI
    cells.append(nbf.v4.new_markdown_cell("""### Sơ Đồ 1.1: Trục Thời Gian Tiến Hóa Trí Tuệ Nhân Tạo (1950 - 2026)
Trực quan hóa 7 giai đoạn phát triển chính của AI cùng các bước ngoặt thuật toán tiêu biểu.
"""))

    cells.append(nbf.v4.new_code_cell("""# Vẽ trục thời gian lịch sử AI (1950 - 2026)
fig, ax = plt.subplots(figsize=(15, 7), dpi=120)

eras = [
    {"name": "Khởi Nguyên AI\\n(1950 - 1956)", "start": 1950, "end": 1956, "color": "#1E40AF", "event": "Turing Test (1950)\\nDartmouth Workshop (1956)"},
    {"name": "Kỷ Nguyên Vàng I\\n(1956 - 1970)", "start": 1956, "end": 1970, "color": "#0284C7", "event": "Perceptron (Rosenblatt 1957)\\nĐịnh lý Novikoff (1962)"},
    {"name": "Mùa Đông AI I\\n(1970 - 1980)", "start": 1970, "end": 1980, "color": "#64748B", "event": "Rào cản XOR (Minsky 1969)\\nLighthill Report (1973)"},
    {"name": "Hệ Chuyên Gia\\n(1980 - 1987)", "start": 1980, "end": 1987, "color": "#059669", "event": "MYCIN, XCON\\nKnowledge Acquisition"},
    {"name": "Mùa Đông AI II\\n(1987 - 1993)", "start": 1987, "end": 1993, "color": "#94A3B8", "event": "Sụp đổ LISP Machine\\nCắt giảm ngân sách"},
    {"name": "Học Máy Thống Kê\\n(1993 - 2011)", "start": 1993, "end": 2011, "color": "#D97706", "event": "SVM (Vapnik), Kernel Trick\\nRandom Forest (2001)"},
    {"name": "Học Sâu & GenAI\\n(2012 - 2026)", "start": 2012, "end": 2026, "color": "#DC2626", "event": "AlexNet (2012), Transformer (2017)\\nLLM & Hybrid KG+RAG (2026)"}
]

y_base = 0
for idx, era in enumerate(eras):
    width = era["end"] - era["start"]
    rect = plt.Rectangle((era["start"], 0.2), width, 0.6, 
                         facecolor=era["color"], edgecolor='white', alpha=0.9, lw=2)
    ax.add_patch(rect)
    
    # Era title
    ax.text(era["start"] + width/2, 0.5, era["name"], 
            ha='center', va='center', color='white', weight='bold', fontsize=8.5)
    
    # Detailed event annotation
    y_event = 1.2 if idx % 2 == 0 else -0.4
    ax.annotate(era["event"], 
                xy=(era["start"] + width/2, 0.8 if idx % 2 == 0 else 0.2),
                xytext=(era["start"] + width/2, y_event),
                ha='center', va='center', fontsize=8, weight='semibold', color='#1F2937',
                bbox=dict(boxstyle="round,pad=0.4", fc="#F3F4F6", ec=era["color"], lw=1.5),
                arrowprops=dict(arrowstyle="->", color=era["color"], lw=1.5))

ax.set_xlim(1948, 2028)
ax.set_ylim(-0.9, 1.7)
ax.set_yticks([])
ax.set_xticks(range(1950, 2030, 10))
ax.set_xlabel("Năm Lịch Sử", weight='bold', fontsize=11)
ax.set_title("TIẾN TRÌNH LỊCH SỬ PHÁT TRIỂN TRÍ TUỆ NHÂN TẠO VÀ CÁC BƯỚC NGOẶT CÔNG NGHỆ", weight='bold', fontsize=13, pad=15)
plt.show()
"""))

    # Sơ đồ 2: Gartner Hype Cycle
    cells.append(nbf.v4.new_markdown_cell("""### Sơ Đồ 1.2: Chu Kỳ Kỳ Vọng (Hype Cycle) và Sự Trỗi Dậy của AI
Minh họa trực quan diễn biến thăng trầm của sự kỳ vọng xã hội và mức độ trưởng thành thực tế của công nghệ AI qua các thập kỷ.
"""))

    cells.append(nbf.v4.new_code_cell("""# Biểu đồ chu kỳ kỳ vọng và các mùa đông AI
fig, ax = plt.subplots(figsize=(13, 6), dpi=120)

x = np.linspace(0, 10, 1000)
# Composite curve modeling peaks of inflated expectations, troughs of disillusionment, and slope of enlightenment
y = (3.5 * np.exp(-((x - 2)**2) / 0.4) + 
     2.8 * np.exp(-((x - 5)**2) / 0.5) + 
     4.8 / (1.0 + np.exp(-1.5 * (x - 7.5))))

ax.plot(x, y, color='#1E40AF', lw=3, label="Mức độ kỳ vọng & Đầu tư")

# Highlighting peaks and winters
ax.scatter([2.0, 5.0], [3.5, 2.8], color='#DC2626', s=120, zorder=5)
ax.text(2.0, 3.8, "Đỉnh Cao Kỳ Vọng I\\n(Perceptron 1960s)", ha='center', weight='bold', fontsize=9, color='#DC2626')
ax.text(5.0, 3.1, "Đỉnh Cao Kỳ Vọng II\\n(Hệ Chuyên Gia 1980s)", ha='center', weight='bold', fontsize=9, color='#DC2626')

ax.scatter([3.5, 6.2], [0.3, 0.4], color='#4B5563', s=120, zorder=5)
ax.text(3.5, -0.2, "MÙA ĐÔNG AI I\\n(1973 - 1980)", ha='center', weight='bold', fontsize=9, color='#1F2937',
        bbox=dict(boxstyle="square", fc="#E5E7EB", ec="#9CA3AF"))
ax.text(6.2, -0.2, "MÙA ĐÔNG AI II\\n(1987 - 1993)", ha='center', weight='bold', fontsize=9, color='#1F2937',
        bbox=dict(boxstyle="square", fc="#E5E7EB", ec="#9CA3AF"))

ax.text(9.0, 4.6, "KỶ NGUYÊN BÙNG NỔ THỰC TẾ\\n(Deep Learning & GenAI/RAG)", ha='center', weight='bold', fontsize=9.5, color='#059669',
        bbox=dict(boxstyle="round", fc="#D1FAE5", ec="#059669"))

ax.set_ylabel("Mức độ Quan tâm / Kỳ vọng Xã hội", weight='bold')
ax.set_xlabel("Tiến trình thời gian qua các thập kỷ", weight='bold')
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylim(-0.5, 5.2)
ax.set_title("MÔ HÌNH CHU KỲ KỲ VỌNG VÀ CÁC MÙA ĐÔNG TRÍ TUỆ NHÂN TẠO", weight='bold', fontsize=12, pad=15)
ax.legend(loc='upper left')
plt.show()
"""))

    # Thực nghiệm: Cài đặt Perceptron từ đầu
    cells.append(nbf.v4.new_markdown_cell(r"""## 1.5. Cài Đặt Thuật Toán Perceptron Từ Đầu Bằng NumPy

Dưới đây là lớp `PerceptronFromScratch` được xây dựng hoàn toàn từ các phép toán đại số tuyến tính của NumPy, kế thừa chính xác công thức toán học của Frank Rosenblatt.
"""))

    cells.append(nbf.v4.new_code_cell("""# Cài đặt mô hình Perceptron từ đầu bằng NumPy
class PerceptronFromScratch:
    def __init__(self, learning_rate=0.1, n_epochs=50):
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None
        self.errors_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Khởi tạo trọng số bằng 0
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.errors_history = []

        for epoch in range(self.n_epochs):
            total_error = 0
            for xi, yi in zip(X, y):
                # Tính toán giá trị dự báo
                linear_output = np.dot(xi, self.weights) + self.bias
                y_predicted = 1 if linear_output >= 0 else 0
                
                # Cập nhật theo sai số
                update = self.lr * (yi - y_predicted)
                self.weights += update * xi
                self.bias += update
                total_error += int(update != 0.0)
            
            self.errors_history.append(total_error)
            if total_error == 0:
                break
        return self

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)
"""))

    # Thực nghiệm trên AND, OR, XOR
    cells.append(nbf.v4.new_markdown_cell("""## 1.6. Thực Nghiệm Kiểm Chứng: Khả Năng Của AND, OR và Sự Bất Lực Trước XOR

Tiến hành huấn luyện mô hình Perceptron trên 3 bài toán logic:
1. Hàm AND (Linearly Separable)
2. Hàm OR (Linearly Separable)
3. Hàm XOR (Non-linearly Separable)
"""))

    cells.append(nbf.v4.new_code_cell("""# Tập dữ liệu kiểm chứng
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

y_and = np.array([0, 0, 0, 1])
y_or  = np.array([0, 1, 1, 1])
y_xor = np.array([0, 1, 1, 0])

# Huấn luyện Perceptron trên 3 bài toán
p_and = PerceptronFromScratch(learning_rate=0.2, n_epochs=20).fit(X, y_and)
p_or  = PerceptronFromScratch(learning_rate=0.2, n_epochs=20).fit(X, y_or)
p_xor = PerceptronFromScratch(learning_rate=0.2, n_epochs=20).fit(X, y_xor)

# Đánh giá độ chính xác
acc_and = np.mean(p_and.predict(X) == y_and) * 100
acc_or  = np.mean(p_or.predict(X) == y_or) * 100
acc_xor = np.mean(p_xor.predict(X) == y_xor) * 100

print(f"Hàm AND - Hội tụ: {p_and.errors_history[-1] == 0} - Số epoch: {len(p_and.errors_history)} - Độ chính xác: {acc_and:.1f}%")
print(f"Hàm OR  - Hội tụ: {p_or.errors_history[-1] == 0} - Số epoch: {len(p_or.errors_history)} - Độ chính xác: {acc_or:.1f}%")
print(f"Hàm XOR - Hội tụ: {p_xor.errors_history[-1] == 0} - Số epoch: {len(p_xor.errors_history)} - Độ chính xác: {acc_xor:.1f}%")
"""))

    # Sơ đồ 3: Trực quan hóa ranh giới AND, OR, XOR
    cells.append(nbf.v4.new_markdown_cell("""### Sơ Đồ 1.3: Trực Quan Hóa Ranh Giới Quyết Định Phẳng của Perceptron (AND, OR và XOR)
Đồ thị dưới đây chứng minh trực quan việc Perceptron có thể vẽ một đường thẳng chia đôi mặt phẳng cho AND và OR, nhưng hoàn toàn không thể vẽ một đường thẳng nào để phân tách nhãn của XOR.
"""))

    cells.append(nbf.v4.new_code_cell("""# Vẽ ranh giới quyết định của Perceptron trên 3 bài toán
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=120)

titles = ["Hàm Logic AND (Khả ly tuyến tính)", "Hàm Logic OR (Khả ly tuyến tính)", "Hàm Logic XOR (Bất khả ly tuyến tính)"]
models = [p_and, p_or, p_xor]
labels = [y_and, y_or, y_xor]

x_grid = np.linspace(-0.5, 1.5, 200)
y_grid = np.linspace(-0.5, 1.5, 200)
XX, YY = np.meshgrid(x_grid, y_grid)
mesh_points = np.c_[XX.ravel(), YY.ravel()]

for ax, model, y_true, title in zip(axes, models, labels, titles):
    # Dự báo trên lưới
    Z = model.predict(mesh_points).reshape(XX.shape)
    ax.contourf(XX, YY, Z, alpha=0.25, cmap=plt.cm.coolwarm)
    
    # Vẽ siêu phẳng quyết định w1*x1 + w2*x2 + b = 0
    if model.weights[1] != 0:
        decision_y = -(model.weights[0] * x_grid + model.bias) / model.weights[1]
        ax.plot(x_grid, decision_y, color='#1E3A8A', lw=2.5, linestyle='--', label='Siêu phẳng phân cách')
    
    # Điểm dữ liệu
    for i in range(len(y_true)):
        color = '#DC2626' if y_true[i] == 1 else '#1E40AF'
        marker = 's' if y_true[i] == 1 else 'o'
        ax.scatter(X[i, 0], X[i, 1], color=color, s=180, marker=marker, edgecolors='black', lw=1.5, zorder=5)
        ax.text(X[i, 0] + 0.08, X[i, 1] + 0.05, f"({X[i, 0]},{X[i, 1]})\\ny={y_true[i]}", fontsize=9, weight='bold')

    ax.set_xlim(-0.3, 1.4)
    ax.set_ylim(-0.3, 1.4)
    ax.set_xlabel("$x_1$", weight='bold')
    ax.set_ylabel("$x_2$", weight='bold')
    ax.set_title(title, weight='bold', fontsize=10.5)
    ax.legend(loc='lower left')

plt.tight_layout()
plt.show()
"""))

    # Giải quyết bài toán XOR bằng MLP
    cells.append(nbf.v4.new_markdown_cell(r"""## 1.7. Giải Quyết Triệt Để Bài Toán XOR Bằng Multi-Layer Perceptron (MLP)

Để vượt qua rào cản Minsky-Papert, mạng nơ-ron đa tầng (Multi-Layer Perceptron) được trang bị lớp ẩn phi tuyến.

### 1.7.1. Kiến trúc mạng MLP cho bài toán XOR
- Tầng đầu vào: $x \in \mathbb{R}^2$
- Tầng ẩn: 2 nơ-ron với hàm kích hoạt Sigmoid:
  $$h = \sigma(W^{(1)} x + b^{(1)}), \quad \sigma(z) = \frac{1}{1 + e^{-z}}$$
- Tầng đầu ra: 1 nơ-ron:
  $$\hat{y} = \sigma(W^{(2)} h + b^{(2)})$$
"""))

    cells.append(nbf.v4.new_code_cell("""# Cài đặt MLP giải quyết bài toán XOR từ đầu bằng NumPy
class MLP_XOR_NumPy:
    def __init__(self, lr=0.5, epochs=5000):
        self.lr = lr
        self.epochs = epochs
        # Khởi tạo trọng số ngẫu nhiên
        np.random.seed(42)
        self.W1 = np.random.randn(2, 2)
        self.b1 = np.zeros((1, 2))
        self.W2 = np.random.randn(2, 1)
        self.b2 = np.zeros((1, 1))

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_derivative(self, a):
        return a * (1.0 - a)

    def fit(self, X, y):
        y = y.reshape(-1, 1)
        for _ in range(self.epochs):
            # Lan truyền xuôi (Forward Pass)
            z1 = np.dot(X, self.W1) + self.b1
            a1 = self.sigmoid(z1)
            z2 = np.dot(a1, self.W2) + self.b2
            a2 = self.sigmoid(z2)

            # Lan truyền ngược (Backpropagation)
            loss_grad = a2 - y
            delta2 = loss_grad * self.sigmoid_derivative(a2)
            dW2 = np.dot(a1.T, delta2)
            db2 = np.sum(delta2, axis=0, keepdims=True)

            delta1 = np.dot(delta2, self.W2.T) * self.sigmoid_derivative(a1)
            dW1 = np.dot(X.T, delta1)
            db1 = np.sum(delta1, axis=0, keepdims=True)

            # Cập nhật trọng số
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
        return self

    def predict_proba(self, X):
        a1 = self.sigmoid(np.dot(X, self.W1) + self.b1)
        return self.sigmoid(np.dot(a1, self.W2) + self.b2)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

# Huấn luyện MLP trên dữ liệu XOR
mlp_xor = MLP_XOR_NumPy(lr=0.8, epochs=3000).fit(X, y_xor)
preds = mlp_xor.predict(X).ravel()
probas = mlp_xor.predict_proba(X).ravel()

print("Kết quả dự báo của MLP trên bài toán XOR:")
for i in range(4):
    print(f"Đầu vào: {X[i]} -> Nhãn thực tế: {y_xor[i]} -> Xác suất dự báo: {probas[i]:.4f} -> Phân loại: {preds[i]}")
print(f"Độ chính xác đạt được: {np.mean(preds == y_xor) * 100:.1f}%")
"""))

    # Sơ đồ 4: Ranh giới phi tuyến của MLP trên XOR
    cells.append(nbf.v4.new_markdown_cell("""### Sơ Đồ 1.4: Ranh Giới Quyết Định Phi Tuyến Của Multi-Layer Perceptron Trên Bài Toán XOR
Trực quan hóa bề mặt quyết định phi tuyến (Non-linear Decision Boundary) được tạo ra bởi tầng ẩn phi tuyến của MLP, phân tách chính xác tuyệt đối 4 điểm nhị phân của hàm XOR.
"""))

    cells.append(nbf.v4.new_code_cell("""# Trực quan hóa ranh giới quyết định phi tuyến của MLP
fig, ax = plt.subplots(figsize=(8, 6), dpi=120)

Z_mlp = mlp_xor.predict_proba(mesh_points).reshape(XX.shape)

contour = ax.contourf(XX, YY, Z_mlp, levels=25, alpha=0.3, cmap=plt.cm.coolwarm)
cbar = plt.colorbar(contour, ax=ax)
cbar.set_label("Xác suất dự báo nhãn 1: $P(y=1|x)$", weight='bold')

# Đường ranh giới 0.5
ax.contour(XX, YY, Z_mlp, levels=[0.5], colors=['#1E3A8A'], linewidths=[2.5], linestyles=['--'])

for i in range(4):
    color = '#DC2626' if y_xor[i] == 1 else '#1E40AF'
    marker = 's' if y_xor[i] == 1 else 'o'
    ax.scatter(X[i, 0], X[i, 1], color=color, s=200, marker=marker, edgecolors='black', lw=1.8, zorder=5)
    ax.text(X[i, 0] + 0.07, X[i, 1] + 0.05, f"({X[i, 0]},{X[i, 1]})\\ny={y_xor[i]}", fontsize=9.5, weight='bold')

ax.set_xlim(-0.3, 1.4)
ax.set_ylim(-0.3, 1.4)
ax.set_xlabel("$x_1$", weight='bold', fontsize=11)
ax.set_ylabel("$x_2$", weight='bold', fontsize=11)
ax.set_title("RANH GIỚI QUYẾT ĐỊNH PHI TUYẾN CỦA MULTI-LAYER PERCEPTRON (GIẢI QUYẾT XOR)", weight='bold', fontsize=11.5, pad=15)
plt.show()
"""))

    # Sơ đồ 5: Cây tiến hóa kiến trúc
    cells.append(nbf.v4.new_markdown_cell("""### Sơ Đồ 1.5: Cây Phả Hệ Tiến Hóa Các Kiến Trúc Nơ-ron (1957 - 2026)
Sơ đồ mô tả sự kế thừa và đột phá kiến trúc từ Perceptron đơn lớp đến các mô hình hiện đại.
"""))

    cells.append(nbf.v4.new_code_cell("""# Sơ đồ phả hệ tiến hóa kiến trúc nơ-ron
import networkx as nx

G = nx.DiGraph()

nodes = [
    ("Perceptron", "Perceptron (1957)\\nĐơn lớp, phân tách phẳng"),
    ("MLP", "Multi-Layer Perceptron (1986)\\nLớp ẩn phi tuyến, Backpropagation"),
    ("CNN", "Convolutional Neural Network\\nLeNet (1998) -> AlexNet (2012)\\nBất biến không gian"),
    ("RNN", "Recurrent Neural Network (1986)\\nTrạng thái ẩn, chuỗi thời gian"),
    ("LSTM", "Long Short-Term Memory (1997)\\nCơ chế cổng, khắc phục triệt tiêu gradient"),
    ("Transformer", "Transformer (2017)\\nSelf-Attention song song hóa"),
    ("RAG_KG", "Hệ Thống Lai KG + RAG (2020-2026)\\nBiểu diễn cấu trúc + Vector ngữ nghĩa")
]

edges = [
    ("Perceptron", "MLP"),
    ("MLP", "CNN"),
    ("MLP", "RNN"),
    ("RNN", "LSTM"),
    ("LSTM", "Transformer"),
    ("Transformer", "RAG_KG"),
    ("CNN", "RAG_KG")
]

pos = {
    "Perceptron": (0.5, 0.95),
    "MLP": (0.5, 0.78),
    "CNN": (0.25, 0.55),
    "RNN": (0.75, 0.55),
    "LSTM": (0.75, 0.35),
    "Transformer": (0.5, 0.18),
    "RAG_KG": (0.5, 0.02)
}

fig, ax = plt.subplots(figsize=(13, 8), dpi=120)
ax.axis('off')

for u, v in edges:
    ax.annotate('', xy=pos[v], xytext=pos[u],
                xycoords='data', textcoords='data',
                arrowprops=dict(arrowstyle="->", color="#4B5563", lw=2.0, shrinkA=22, shrinkB=22))

for nid, label in nodes:
    x, y = pos[nid]
    ax.text(x, y, label, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.5", fc="#EFF6FF", ec="#1D4ED8", lw=2.0),
            fontsize=9, weight='bold', color='#1E3A8A')

ax.set_title("CÂY PHẢ HỆ TIẾN HÓA CÁC KIẾN TRÚC MẠNG NƠ-RON TỪ PERCEPTRON ĐẾN RAG & KG", weight='bold', fontsize=12.5, pad=15)
plt.show()
"""))

    # Summary
    cells.append(nbf.v4.new_markdown_cell("""## 1.8. Kết Luận Chương 1

Quá trình tiến hóa của Trí tuệ Nhân tạo là một chuỗi biện chứng giữa kỳ vọng xã hội, rào cản toán học lý thuyết và đột phá kỹ nghệ:
1. **Perceptron** xác lập nền tảng cho việc máy tính có thể tự học trọng số từ dữ liệu, nhưng bị hạn chế bởi bản chất phân tách tuyến tính phẳng.
2. **Bài toán XOR** do Minsky & Papert chỉ ra đã trở thành động lực để cộng đồng phát triển mạng nơ-ron đa tầng (MLP) và giải thuật lan truyền ngược (Backpropagation).
3. Sự tích lũy về dữ liệu lớn và năng lực tính toán GPU đã mở ra kỷ nguyên **Học sâu (Deep Learning)**, làm tiền đề cho các mô hình tích chập (CNN - Chương 3) và chuỗi thời gian (RNN/LSTM - Chương 4).
4. Xu hướng hiện đại hướng tới việc kết hợp giữa học biểu diễn phi cấu trúc với tri thức biểu tượng có cấu trúc thông qua **Knowledge Graph và RAG (Chương 5)** nhằm đảm bảo tính chính xác và minh bạch.
"""))

    nb.cells = cells
    return nb

if __name__ == '__main__':
    nb = create_nb1()
    out_path = '01_Chuong1_Lich_Su_Phat_Trien_AI.ipynb'
    with open(out_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Created {out_path} successfully!")
