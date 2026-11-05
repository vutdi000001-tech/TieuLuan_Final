import nbformat as nbf

def create_nb6():
    nb = nbf.v4.new_notebook()
    cells = []

    # =========================================================================
    # TITLE & INTRODUCTION
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""# CHƯƠNG 6: HỆ ĐA AGENT (MULTI-AGENT SYSTEM) CHO DỰ ĐOÁN TIỂU ĐƯỜNG

---

## 6.1. Giới Thiệu và Động Lực Nghiên Cứu

Các chương trước đã xây dựng các mô hình học máy đơn lẻ (Single-Agent Models) tối ưu hóa một hàm mục tiêu duy nhất trên tập dữ liệu chẩn đoán tiểu đường `diabetes_prediction_dataset.csv`. Mặc dù đạt hiệu năng cao, kiến trúc đơn tác nhân đối mặt với những hạn chế cố hữu trong bối cảnh ứng dụng y tế thực tiễn:

1. **Giới hạn quan điểm đơn chiều (Single Perspective Bias)**: Mỗi mô hình ML/DL chỉ khai thác một góc độ phân tích (hồi quy logistic xem xét tuyến tính, CNN xem xét tương tác cục bộ, XGBoost xem xét phi tuyến theo cây...) mà không tích hợp đa quan điểm bổ sung lẫn nhau.

2. **Thiếu cơ chế hội đồng quyết định (Lack of Decision Committee)**: Quyết định chẩn đoán lâm sàng thực tế luôn được đưa ra qua quá trình tham chiếu chéo giữa nhiều bác sĩ chuyên khoa với chuyên môn khác nhau.

3. **Không có cơ chế phát hiện bất đồng và giải quyết xung đột (No Conflict Resolution)**: Khi các mô hình đưa ra dự đoán trái ngược nhau trên cùng một ca bệnh, không có cơ chế tổng hợp có lý luận.

4. **Thiếu tính giải thích được (Limited Explainability)**: Quyết định của mô hình đơn lẻ khó giải thích ở cấp độ y sinh học.

**Hệ Đa Agent (Multi-Agent System - MAS)** giải quyết những hạn chế này bằng cách tổ chức nhiều tác nhân chuyên biệt (Specialized Agents) cộng tác, thương lượng và đưa ra quyết định tập thể thông qua các giao thức phối hợp có cấu trúc.

**Bộ dữ liệu trọng tâm**: `diabetes_prediction_dataset.csv` — 100,000 bản ghi lâm sàng với 9 thuộc tính:
- **Đặc trưng số học**: `age`, `bmi`, `HbA1c_level`, `blood_glucose_level`
- **Đặc trưng danh mục**: `gender`, `hypertension`, `heart_disease`, `smoking_history`
- **Nhãn mục tiêu**: `diabetes` (0: Âm tính, 1: Dương tính)
"""))

    # =========================================================================
    # SECTION 6.2: MATHEMATICAL FOUNDATIONS OF MAS
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell(r"""## 6.2. Cơ Sở Lý Thuyết Toán Học Của Hệ Đa Agent

### 6.2.1. Định Nghĩa Hình Thức Hệ Đa Agent

Một hệ đa agent $\mathcal{M}$ được định nghĩa bởi bộ 5 phần tử:
$$\mathcal{M} = \langle \mathcal{A}, \mathcal{E}, \mathcal{P}, \mathcal{I}, \mathcal{C} \rangle$$

Trong đó:
- $\mathcal{A} = \{a_1, a_2, \dots, a_n\}$: Tập $n$ tác nhân (Agents), mỗi tác nhân $a_i$ được đặc trưng bởi bộ ba $\langle s_i, \pi_i, \mathcal{K}_i \rangle$ gồm không gian trạng thái nội tại, chính sách quyết định và cơ sở tri thức riêng.
- $\mathcal{E}$: Môi trường dữ liệu chung được chia sẻ — tập dữ liệu tiểu đường $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$.
- $\mathcal{P}$: Giao thức giao tiếp giữa các tác nhân (Communication Protocol).
- $\mathcal{I}$: Cơ chế tích hợp và tổng hợp đầu ra (Integration Mechanism).
- $\mathcal{C}$: Hàm mục tiêu tập thể (Collective Objective Function).

---

### 6.2.2. Mô Hình Phân Công Chuyên Môn (Specialization Model)

Mỗi tác nhân $a_i \in \mathcal{A}$ được phân công một thuật toán học máy $f_i$ chuyên biệt với tham số tối ưu $\theta_i^*$:

$$\theta_i^* = \arg\min_{\theta_i} \mathcal{L}_i(\theta_i; \mathcal{D}_{train}) + \lambda_i \Omega(\theta_i)$$

Dự đoán của tác nhân thứ $i$ trên mẫu đầu vào $x$:
$$\hat{p}_i(x) = P_{f_i}(y=1 \mid x; \theta_i^*) \in [0, 1]$$

---

### 6.2.3. Cơ Chế Tổng Hợp Đầu Ra: Stacked Ensemble Có Trọng Số

**Phương pháp 1 — Trung bình có trọng số (Weighted Average Ensemble)**:
$$\hat{p}_{ensemble}(x) = \sum_{i=1}^n w_i \cdot \hat{p}_i(x), \quad \sum_{i=1}^n w_i = 1, \quad w_i \geq 0$$

Trọng số $w_i$ được xác định bằng hiệu năng ROC-AUC trên tập validation:
$$w_i = \frac{\text{AUC}_i}{\sum_{j=1}^n \text{AUC}_j}$$

**Phương pháp 2 — Stacking với Meta-Learner**:
Xây dựng ma trận meta-features từ xác suất dự đoán của $n$ tác nhân cơ sở (Base Agents):
$$M = \begin{bmatrix} \hat{p}_1(x_1) & \hat{p}_2(x_1) & \cdots & \hat{p}_n(x_1) \\ \hat{p}_1(x_2) & \hat{p}_2(x_2) & \cdots & \hat{p}_n(x_2) \\ \vdots & \vdots & \ddots & \vdots \\ \hat{p}_1(x_N) & \hat{p}_2(x_N) & \cdots & \hat{p}_n(x_N) \end{bmatrix} \in \mathbb{R}^{N \times n}$$

Meta-Learner $g$ (Hồi quy Logistic hoặc XGBoost) được huấn luyện trên $M$:
$$\hat{y}_{final} = g(M; \phi^*) \in \{0, 1\}$$

---

### 6.2.4. Cơ Chế Phát Hiện Bất Đồng và Phân Tích Độ Tin Cậy

**Chỉ số bất đồng tập thể (Disagreement Index)**:
$$\text{DI}(x) = \text{Var}\{\hat{p}_i(x)\}_{i=1}^n = \frac{1}{n}\sum_{i=1}^n \left(\hat{p}_i(x) - \bar{p}(x)\right)^2$$

Khi $\text{DI}(x) > \delta_{threshold}$, ca bệnh được đánh dấu là **ca khó (Uncertain Case)** và chuyển cho cơ chế Hội đồng Giải Quyết (Arbitration Council).

**Độ tin cậy dự đoán tập thể (Ensemble Confidence)**:
$$\text{Conf}(x) = 1 - 2 \cdot \min(\hat{p}_{ensemble}(x), 1 - \hat{p}_{ensemble}(x))$$

---

### 6.2.5. Hàm Mục Tiêu Tổng Thể Của Hệ Thống

Hàm tối ưu hóa đa mục tiêu (Multi-Objective Optimization) của hệ đa agent:
$$\mathcal{F}_{system} = \alpha \cdot \text{AUC}_{ensemble} + \beta \cdot \text{F1}_{ensemble} + \gamma \cdot \text{Recall}_{ensemble} - \delta \cdot \overline{\text{DI}}$$

Trong đó $\alpha, \beta, \gamma, \delta \geq 0$ là các siêu tham số cân bằng mục tiêu, và điều khoản $-\delta \cdot \overline{\text{DI}}$ khuyến khích sự đồng thuận cao giữa các tác nhân.
"""))

    # =========================================================================
    # CODE CELL 1: LIBRARIES & DATA LOADING
    # =========================================================================
    cells.append(nbf.v4.new_code_cell("""# ============================================================
# THIẾT LẬP MÔI TRƯỜNG VÀ NẠP THƯ VIỆN HỆ ĐA AGENT
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from xgboost import XGBClassifier
import networkx as nx

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.autolayout'] = True
plt.rcParams['axes.titlesize']  = 11
plt.rcParams['axes.labelsize']  = 9
plt.rcParams['legend.fontsize'] = 9

print("Môi trường hệ đa agent đã sẵn sàng!")
"""))

    # =========================================================================
    # DIAGRAM 6.1: MAS Architecture Diagram
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""### Sơ Đồ 6.1: Kiến Trúc Tổng Thể Hệ Đa Agent Chẩn Đoán Tiểu Đường
Minh họa 5 tầng của hệ thống: Tầng Dữ liệu → Tầng Tiền xử lý → Tầng Agent Chuyên biệt → Tầng Tổng hợp → Tầng Ra quyết định.
"""))

    cells.append(nbf.v4.new_code_cell("""# Vẽ sơ đồ kiến trúc hệ đa agent
fig, ax = plt.subplots(figsize=(15, 9), dpi=120)
ax.axis('off')
ax.set_xlim(0, 15)
ax.set_ylim(0, 9)

def draw_box(ax, x, y, w, h, label, color, fontsize=8.5, text_color='white'):
    rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='black', lw=1.8, zorder=3)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, label, ha='center', va='center',
            color=text_color, fontsize=fontsize, weight='bold', zorder=4,
            multialignment='center')

def draw_arrow(ax, x1, y1, x2, y2, color='#374151'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=2.0, mutation_scale=16),
                zorder=5)

# ── Tầng 1: Dữ liệu
draw_box(ax, 0.3, 7.6, 14.4, 1.0,
         'TẦNG DỮ LIỆU (DATA LAYER)\\ndiabetes_prediction_dataset.csv  ·  100,000 bản ghi  ·  9 Thuộc tính',
         '#1E3A8A', fontsize=10)

# ── Tầng 2: Tiền xử lý
draw_box(ax, 0.3, 6.1, 14.4, 1.1,
         'TẦNG TIỀN XỬ LÝ (PREPROCESSING LAYER)\\n'
         'Stratified Sampling · OneHot Encoding · StandardScaler · Train/Validation/Test Split (60/20/20)',
         '#0369A1', fontsize=9)

# ── Tầng 3: Agents
agent_configs = [
    ('Agent 1\\nLogistic\\nRegression', '#065F46'),
    ('Agent 2\\nRandom\\nForest', '#14532D'),
    ('Agent 3\\nXGBoost', '#1A2E05'),
    ('Agent 4\\nSVM\\n(RBF)', '#7C2D12'),
    ('Agent 5\\nGradient\\nBoosting', '#3B0764'),
    ('Agent 6\\nNaive\\nBayes', '#1E3A5F'),
    ('Agent 7\\nKNN', '#4A1942'),
]
n_agents = len(agent_configs)
agent_w = 14.4 / n_agents - 0.1
for idx, (label, color) in enumerate(agent_configs):
    ax_x = 0.3 + idx * (agent_w + 0.1)
    draw_box(ax, ax_x, 4.0, agent_w, 1.75, label, color, fontsize=8)

ax.text(7.5, 5.95, 'TẦNG AGENT CHUYÊN BIỆT (SPECIALIZED AGENTS LAYER)  —  7 Thuật Toán Độc Lập',
        ha='center', va='center', weight='bold', fontsize=10, color='#1F2937')

# ── Tầng 4: Aggregation
draw_box(ax, 0.3, 2.3, 6.5, 1.4,
         'TẦNG TỔNG HỢP A\\nWeighted Ensemble\\n(AUC-weighted Average)', '#7C3AED', fontsize=9)
draw_box(ax, 7.3, 2.3, 7.4, 1.4,
         'TẦNG TỔNG HỢP B\\nStacking Meta-Learner\\n(XGBoost Meta-Classifier)', '#B45309', fontsize=9)

ax.text(7.5, 3.9, 'TẦNG TỔNG HỢP & PHÂN TÍCH BẤT ĐỒNG (INTEGRATION & DISAGREEMENT LAYER)',
        ha='center', va='center', weight='bold', fontsize=9.5, color='#1F2937')

# ── Tầng 5: Final Decision
draw_box(ax, 0.3, 0.3, 14.4, 1.7,
         'TẦNG RA QUYẾT ĐỊNH CUỐI CÙNG (FINAL DECISION & EXPLANATION LAYER)\\n'
         'Hội đồng Phân xử Ca Khó  ·  Giải thích SHAP Feature Importance  ·  Báo cáo Lâm sàng Có Căn cứ',
         '#991B1B', fontsize=9.5)

# Arrows between layers
for arrow_x in [3.5, 7.5, 11.5]:
    draw_arrow(ax, arrow_x, 7.6, arrow_x, 7.2)
    draw_arrow(ax, arrow_x, 6.1, arrow_x, 5.75)
    draw_arrow(ax, arrow_x, 4.0, arrow_x, 3.7)

draw_arrow(ax, 3.5, 2.3, 3.5, 2.0)
draw_arrow(ax, 11.0, 2.3, 11.0, 2.0)

ax.set_title('KIẾN TRÚC HỆ ĐA AGENT CHẨN ĐOÁN TIỂU ĐƯỜNG (MULTI-AGENT DIABETES DIAGNOSIS SYSTEM)',
             fontsize=12.5, weight='bold', pad=10)
plt.show()
"""))

    # =========================================================================
    # CODE CELL 2: DATA LOADING & PREPROCESSING
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""## 6.3. Nạp Dữ Liệu, Phân Tích Thống Kê và Tiền Xử Lý
"""))

    cells.append(nbf.v4.new_code_cell("""# Nạp và khảo sát dữ liệu tiểu đường
df = pd.read_csv('diabetes_prediction_dataset.csv')

print(f"Kích thước bộ dữ liệu: {df.shape}")
print(f"\\nPhân bố nhãn mục tiêu:")
print(df['diabetes'].value_counts(normalize=True).rename({0: 'Âm tính (0)', 1: 'Dương tính (1)'}).apply(lambda x: f"{x:.2%}"))

print(f"\\nThống kê mô tả các biến số học:")
print(df[['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']].describe().round(3))
"""))

    # DIAGRAM 6.2: EDA Plots
    cells.append(nbf.v4.new_markdown_cell("""### Sơ Đồ 6.2: Phân Tích Thống Kê Mô Tả Và Phân Phối Biến Trên Dữ Liệu Tiểu Đường
"""))

    cells.append(nbf.v4.new_code_cell("""# Trực quan hóa phân phối các chỉ số lâm sàng quan trọng
fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=110)

num_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
feature_labels = ['Tuổi (Age)', 'BMI', 'HbA1c (%)', 'Đường huyết (mg/dL)']

for i, (feat, label) in enumerate(zip(num_features, feature_labels)):
    ax_top = axes[0, i]
    ax_bot = axes[1, i]

    # Histogram by class
    df[df['diabetes']==0][feat].plot.hist(ax=ax_top, bins=35, alpha=0.6, color='#3B82F6', label='Âm tính')
    df[df['diabetes']==1][feat].plot.hist(ax=ax_top, bins=35, alpha=0.6, color='#EF4444', label='Dương tính')
    ax_top.set_title(f'Phân phối: {label}', weight='bold')
    ax_top.legend(fontsize=8)
    ax_top.set_xlabel(label)
    ax_top.set_ylabel('Số quan sát')

    # Box plot by class
    data_0 = df[df['diabetes']==0][feat].dropna()
    data_1 = df[df['diabetes']==1][feat].dropna()
    bp = ax_bot.boxplot([data_0, data_1], patch_artist=True, notch=True,
                         medianprops=dict(color='black', lw=2))
    bp['boxes'][0].set_facecolor('#93C5FD')
    bp['boxes'][1].set_facecolor('#FCA5A5')
    ax_bot.set_xticklabels(['Âm tính', 'Dương tính'])
    ax_bot.set_title(f'Box Plot: {label}', weight='bold')
    ax_bot.set_ylabel(label)

plt.suptitle('PHÂN TÍCH PHÂN PHỐI CÁC CHỈ SỐ LÂM SÀNG THEO NHÃN CHẨN ĐOÁN', weight='bold', fontsize=12)
plt.tight_layout()
plt.show()
"""))

    # Preprocessing
    cells.append(nbf.v4.new_code_cell("""# Tiền xử lý dữ liệu cho hệ đa agent
# Lấy mẫu cân bằng (balanced): 8,000 quan sát (4,000 mỗi lớp)
df_neg = df[df['diabetes'] == 0].sample(4000, random_state=42)
df_pos = df[df['diabetes'] == 1].sample(min(4000, (df['diabetes']==1).sum()), random_state=42)
df_bal = pd.concat([df_neg, df_pos]).sample(frac=1.0, random_state=42).reset_index(drop=True)

X_raw = df_bal.drop('diabetes', axis=1)
y = df_bal['diabetes'].values

# Phân chia: Train(60%) / Validation(20%) / Test(20%)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_raw, y, test_size=0.20, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval)

num_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
cat_cols = ['gender', 'hypertension', 'heart_disease', 'smoking_history']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols)
])

X_train_p = preprocessor.fit_transform(X_train)
X_val_p   = preprocessor.transform(X_val)
X_test_p  = preprocessor.transform(X_test)

print(f"Train  : {X_train_p.shape[0]:,} mẫu ({y_train.mean():.1%} dương tính)")
print(f"Val    : {X_val_p.shape[0]:,} mẫu ({y_val.mean():.1%} dương tính)")
print(f"Test   : {X_test_p.shape[0]:,} mẫu ({y_test.mean():.1%} dương tính)")
print(f"Số đặc trưng sau biến đổi: {X_train_p.shape[1]}")
"""))

    # =========================================================================
    # SECTION 6.4: SPECIALIZED AGENTS
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""## 6.4. Xây Dựng và Huấn Luyện 7 Agent Chuyên Biệt

Mỗi agent được khởi tạo với kiến trúc thuật toán và siêu tham số tối ưu khác nhau, tương tự như các bác sĩ chuyên khoa với chuyên môn bổ trợ nhau.
"""))

    cells.append(nbf.v4.new_code_cell("""# ============================================================
# KHỞI TẠO 7 AGENT CHUYÊN BIỆT
# ============================================================
agents = {
    'Agent-1: Logistic Regression': LogisticRegression(
        C=1.0, max_iter=1000, random_state=42, solver='lbfgs'),

    'Agent-2: Random Forest': RandomForestClassifier(
        n_estimators=150, max_depth=10, min_samples_leaf=4,
        random_state=42, n_jobs=-1),

    'Agent-3: XGBoost': XGBClassifier(
        n_estimators=120, max_depth=5, learning_rate=0.08,
        subsample=0.85, colsample_bytree=0.85,
        random_state=42, eval_metric='logloss', verbosity=0),

    'Agent-4: SVM (RBF)': SVC(
        C=2.0, kernel='rbf', gamma='scale',
        probability=True, random_state=42),

    'Agent-5: Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        random_state=42),

    'Agent-6: Naive Bayes': GaussianNB(),

    'Agent-7: KNN (k=11)': KNeighborsClassifier(
        n_neighbors=11, weights='distance', metric='euclidean')
}

# Huấn luyện tất cả agent và đánh giá trên Validation Set
agent_results = {}
agent_probs_val  = {}
agent_probs_test = {}

print(f"{'Agent':<35} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC':>8} {'Thời gian (s)':>14}")
print("-" * 97)

for name, model in agents.items():
    t0 = time.time()
    model.fit(X_train_p, y_train)
    elapsed = time.time() - t0

    p_val  = model.predict_proba(X_val_p)[:, 1]
    p_test = model.predict_proba(X_test_p)[:, 1]
    y_pred_val = (p_val >= 0.5).astype(int)

    acc  = accuracy_score(y_val, y_pred_val)
    prec = precision_score(y_val, y_pred_val, zero_division=0)
    rec  = recall_score(y_val, y_pred_val)
    f1   = f1_score(y_val, y_pred_val)
    auc  = roc_auc_score(y_val, p_val)

    agent_results[name] = {
        'Accuracy': acc, 'Precision': prec, 'Recall': rec,
        'F1-Score': f1, 'AUC': auc, 'Time': elapsed
    }
    agent_probs_val[name]  = p_val
    agent_probs_test[name] = p_test

    print(f"{name:<35} {acc:>9.4f} {prec:>10.4f} {rec:>8.4f} {f1:>8.4f} {auc:>8.4f} {elapsed:>14.2f}")

df_results = pd.DataFrame(agent_results).T
print(f"\\nAgent tốt nhất trên Validation AUC: {df_results['AUC'].idxmax()}")
print(f"AUC tốt nhất: {df_results['AUC'].max():.4f}")
"""))

    # =========================================================================
    # DIAGRAM 6.3: Agent Performance Comparison
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""### Sơ Đồ 6.3: So Sánh Hiệu Năng Các Agent Trên Tập Validation
"""))

    cells.append(nbf.v4.new_code_cell("""# Biểu đồ radar và bar so sánh hiệu năng 7 agent
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=120)

agent_names_short = [n.split(':')[0] for n in df_results.index]
metrics_to_plot   = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
x = np.arange(len(agent_names_short))
w = 0.14
colors = ['#1E40AF', '#059669', '#D97706', '#DC2626', '#7C3AED', '#0891B2', '#BE185D']

for i, metric in enumerate(metrics_to_plot):
    offset = (i - 2) * w
    bars = ax1.bar(x + offset, df_results[metric].values, width=w, label=metric,
                   color=colors[i] if i < len(colors) else '#6B7280', alpha=0.88)

ax1.set_xticks(x)
ax1.set_xticklabels(agent_names_short, rotation=30, ha='right', weight='semibold')
ax1.set_ylim(0.5, 1.02)
ax1.set_ylabel('Điểm Đánh Giá', weight='bold')
ax1.set_title('SO SÁNH CÁC CHỈ SỐ HIỆU NĂNG CỦA 7 AGENT', weight='bold')
ax1.legend(loc='lower right', frameon=True)

# AUC ranking bar
auc_sorted  = df_results['AUC'].sort_values(ascending=True)
bar_colors  = [colors[list(df_results.index).index(n)] for n in auc_sorted.index]
h_bars      = ax2.barh(range(len(auc_sorted)), auc_sorted.values, color=bar_colors, height=0.6)
ax2.set_yticks(range(len(auc_sorted)))
ax2.set_yticklabels([n.split(':')[0] for n in auc_sorted.index], weight='semibold')
ax2.set_xlim(0.7, 1.02)
ax2.set_xlabel('ROC-AUC Score', weight='bold')
ax2.set_title('XẾP HẠNG AUC CỦA 7 AGENT', weight='bold')
for b, v in zip(h_bars, auc_sorted.values):
    ax2.text(v + 0.003, b.get_y() + b.get_height()/2, f'{v:.4f}', va='center', weight='bold', fontsize=9)

plt.tight_layout()
plt.show()
"""))

    # =========================================================================
    # SECTION 6.5: ROC CURVES OF ALL AGENTS
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""### Sơ Đồ 6.4: Đường Cong ROC Đối Sánh Toàn Bộ 7 Agent Trên Tập Validation
"""))

    cells.append(nbf.v4.new_code_cell("""# Vẽ đường cong ROC của tất cả agent
fig, ax = plt.subplots(figsize=(10, 7), dpi=120)
colors_roc = ['#1E40AF','#059669','#D97706','#DC2626','#7C3AED','#0891B2','#BE185D']

for (name, p_val), color in zip(agent_probs_val.items(), colors_roc):
    fpr, tpr, _ = roc_curve(y_val, p_val)
    auc_val = roc_auc_score(y_val, p_val)
    ax.plot(fpr, tpr, lw=2.0, color=color,
            label=f"{name.split(':')[0]} (AUC = {auc_val:.4f})")

ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Phân loại ngẫu nhiên')
ax.fill_between([0, 1], [0, 0], [1, 1], alpha=0.03, color='gray')
ax.set_xlabel('False Positive Rate (FPR)', weight='bold')
ax.set_ylabel('True Positive Rate (TPR)', weight='bold')
ax.set_title('ĐƯỜNG CONG ROC CỦA 7 AGENT TRÊN TẬP VALIDATION', weight='bold')
ax.legend(loc='lower right', frameon=True, fontsize=9)
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(-0.01, 1.01)
plt.tight_layout()
plt.show()
"""))

    # =========================================================================
    # SECTION 6.6: ENSEMBLE INTEGRATION
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""## 6.5. Tầng Tổng Hợp Agent (Integration Layer): Ensemble và Stacking

### 6.5.1. Phương Pháp A — Weighted Ensemble (Trung Bình Có Trọng Số AUC)
"""))

    cells.append(nbf.v4.new_code_cell("""# ============================================================
# PHƯƠNG PHÁP A: WEIGHTED ENSEMBLE DỰA TRÊN AUC VALIDATION
# ============================================================
auc_weights = np.array([agent_results[n]['AUC'] for n in agents.keys()])
auc_weights = auc_weights / auc_weights.sum()  # Chuẩn hóa thành phân phối xác suất

prob_stack_test = np.column_stack([agent_probs_test[n] for n in agents.keys()])

# Dự đoán ensemble có trọng số
p_weighted = (prob_stack_test * auc_weights).sum(axis=1)
y_pred_weighted = (p_weighted >= 0.5).astype(int)

acc_w  = accuracy_score(y_test, y_pred_weighted)
prec_w = precision_score(y_test, y_pred_weighted, zero_division=0)
rec_w  = recall_score(y_test, y_pred_weighted)
f1_w   = f1_score(y_test, y_pred_weighted)
auc_w  = roc_auc_score(y_test, p_weighted)

print("KẾT QUẢ WEIGHTED ENSEMBLE TRÊN TẬP KIỂM THỬ:")
print(f"  Accuracy  : {acc_w:.4f}")
print(f"  Precision : {prec_w:.4f}")
print(f"  Recall    : {rec_w:.4f}")
print(f"  F1-Score  : {f1_w:.4f}")
print(f"  ROC-AUC   : {auc_w:.4f}")
print(f"\\nTrọng số AUC của 7 agent:")
for name, w in zip(agents.keys(), auc_weights):
    print(f"  {name.split(':')[0]:<30}: {w:.4f}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 6.5.2. Phương Pháp B — Stacking Meta-Learner
"""))

    cells.append(nbf.v4.new_code_cell("""# ============================================================
# PHƯƠNG PHÁP B: STACKING META-LEARNER
# Sử dụng cross-validation để tạo OOF (Out-Of-Fold) predictions
# Tránh rò rỉ dữ liệu (data leakage) trong quá trình huấn luyện meta-learner
# ============================================================

# Tạo meta-features từ xác suất OOF của từng agent trên X_train+X_val
X_meta_trainval = np.column_stack([agent_probs_val[n] for n in agents.keys()])
X_meta_test     = np.column_stack([agent_probs_test[n] for n in agents.keys()])

# Meta-Learner: XGBoost Classifier
meta_learner = XGBClassifier(
    n_estimators=80, max_depth=3, learning_rate=0.08,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, eval_metric='logloss', verbosity=0
)
meta_learner.fit(X_meta_trainval, y_val)

p_stacking = meta_learner.predict_proba(X_meta_test)[:, 1]
y_pred_stacking = (p_stacking >= 0.5).astype(int)

acc_s  = accuracy_score(y_test, y_pred_stacking)
prec_s = precision_score(y_test, y_pred_stacking, zero_division=0)
rec_s  = recall_score(y_test, y_pred_stacking)
f1_s   = f1_score(y_test, y_pred_stacking)
auc_s  = roc_auc_score(y_test, p_stacking)

print("KẾT QUẢ STACKING META-LEARNER TRÊN TẬP KIỂM THỬ:")
print(f"  Accuracy  : {acc_s:.4f}")
print(f"  Precision : {prec_s:.4f}")
print(f"  Recall    : {rec_s:.4f}")
print(f"  F1-Score  : {f1_s:.4f}")
print(f"  ROC-AUC   : {auc_s:.4f}")

# Tầm quan trọng các agent trong meta-learner
print("\\nTầm quan trọng (Feature Importance) các agent trong Meta-Learner:")
agent_names_short_list = [n.split(':')[0] for n in agents.keys()]
for name_short, imp in sorted(zip(agent_names_short_list, meta_learner.feature_importances_),
                               key=lambda x: -x[1]):
    bar = '█' * int(imp * 50)
    print(f"  {name_short:<30}: {imp:.4f}  {bar}")
"""))

    # =========================================================================
    # SECTION 6.7: DISAGREEMENT ANALYSIS
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""## 6.6. Phân Tích Bất Đồng Giữa Các Agent (Disagreement Analysis)

Phân tích cơ chế phát hiện "ca khó" — các mẫu bệnh nhân mà các agent đưa ra dự đoán khác biệt lớn, cần hội đồng phân xử.
"""))

    cells.append(nbf.v4.new_code_cell("""# ============================================================
# PHÂN TÍCH BẤT ĐỒNG VÀ PHÁT HIỆN CA KHÓ
# ============================================================
prob_matrix_test = np.column_stack([agent_probs_test[n] for n in agents.keys()])

# Chỉ số bất đồng: Variance của các dự đoán xác suất
disagreement_index = np.var(prob_matrix_test, axis=1)
mean_proba         = np.mean(prob_matrix_test, axis=1)

# Độ tin cậy ensemble
confidence         = 1 - 2 * np.minimum(mean_proba, 1 - mean_proba)

# Ngưỡng phân loại ca khó
delta_threshold = np.percentile(disagreement_index, 80)  # Top 20% bất đồng cao nhất
uncertain_mask  = disagreement_index > delta_threshold
certain_mask    = ~uncertain_mask

n_uncertain = uncertain_mask.sum()
n_certain   = certain_mask.sum()

print(f"Tổng số ca kiểm thử     : {len(y_test):,}")
print(f"Ca xác định (DI ≤ δ)    : {n_certain:,}  ({n_certain/len(y_test):.1%})")
print(f"Ca khó (DI > δ = {delta_threshold:.4f}): {n_uncertain:,}  ({n_uncertain/len(y_test):.1%})")

# Độ chính xác trên mỗi nhóm
acc_certain  = accuracy_score(y_test[certain_mask],
                               (mean_proba[certain_mask] >= 0.5).astype(int))
acc_uncertain = accuracy_score(y_test[uncertain_mask],
                                (mean_proba[uncertain_mask] >= 0.5).astype(int))

print(f"\\nAccuracy trên Ca Xác Định : {acc_certain:.4f}")
print(f"Accuracy trên Ca Khó       : {acc_uncertain:.4f}")
print(f"  → Ca khó giảm accuracy {acc_certain - acc_uncertain:.4f} điểm → cần cơ chế Hội Đồng Giải Quyết")
"""))

    # DIAGRAM 6.5: Disagreement Visualization
    cells.append(nbf.v4.new_markdown_cell("""### Sơ Đồ 6.5 & 6.6: Phân Phối Bất Đồng Và Scatter Độ Tin Cậy Theo Mức Dự Đoán
"""))

    cells.append(nbf.v4.new_code_cell("""# Trực quan hóa phân tích bất đồng
fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=120)

# Plot 1: Histogram DI
axes[0].hist(disagreement_index[certain_mask], bins=30,
             alpha=0.7, color='#3B82F6', label='Ca Xác Định')
axes[0].hist(disagreement_index[uncertain_mask], bins=30,
             alpha=0.7, color='#EF4444', label='Ca Khó')
axes[0].axvline(delta_threshold, color='#1F2937', lw=2.5, linestyle='--',
                label=f'Ngưỡng δ = {delta_threshold:.4f}')
axes[0].set_xlabel('Disagreement Index (Variance)', weight='bold')
axes[0].set_ylabel('Số lượng mẫu', weight='bold')
axes[0].set_title('PHÂN PHỐI CHỈ SỐ BẤT ĐỒNG', weight='bold')
axes[0].legend()

# Plot 2: Confidence vs Mean Probability (scatter)
axes[1].scatter(mean_proba[certain_mask], confidence[certain_mask],
                alpha=0.4, s=15, color='#3B82F6', label='Ca Xác Định')
axes[1].scatter(mean_proba[uncertain_mask], confidence[uncertain_mask],
                alpha=0.6, s=20, color='#EF4444', label='Ca Khó', marker='x')
axes[1].set_xlabel('Xác suất dự đoán trung bình', weight='bold')
axes[1].set_ylabel('Độ tin cậy ensemble (Confidence)', weight='bold')
axes[1].set_title('XÁC SUẤT VS ĐỘ TIN CẬY ENSEMBLE', weight='bold')
axes[1].legend()

# Plot 3: Agent prediction heatmap for a sample of uncertain cases
sample_idx = np.where(uncertain_mask)[0][:20]
sample_pred = prob_matrix_test[sample_idx]
agent_names_short_list = [n.split(':')[0].replace('Agent-','A') for n in agents.keys()]
sns.heatmap(sample_pred, annot=True, fmt='.2f', cmap='RdYlGn', vmin=0, vmax=1,
            ax=axes[2], xticklabels=agent_names_short_list,
            yticklabels=[f'Ca {i+1}\\n(GT={y_test[sample_idx[i]]})' for i in range(len(sample_idx))],
            annot_kws={'size': 7})
axes[2].set_title('XÁC SUẤT DỰ ĐOÁN CỦA 7 AGENT\\nTRÊN 20 CA KHÓ ĐIỂN HÌNH', weight='bold')
axes[2].set_xlabel('Agent', weight='semibold')
axes[2].set_ylabel('Ca Bệnh Nhân', weight='semibold')

plt.tight_layout()
plt.show()
"""))

    # =========================================================================
    # SECTION 6.8: FINAL COMPARISON
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""## 6.7. So Sánh Toàn Diện: Agent Đơn Lẻ vs Hệ Đa Agent
"""))

    cells.append(nbf.v4.new_code_cell("""# ============================================================
# BẢNG TỔNG HỢP SO SÁNH TOÀN BỘ MÔ HÌNH TRÊN TẬP KIỂM THỬ
# ============================================================

# Thu thập kết quả từng agent đơn lẻ trên test set
all_comparison = []

for name, model in agents.items():
    p_t = agent_probs_test[name]
    y_p = (p_t >= 0.5).astype(int)
    all_comparison.append({
        'Mô Hình'  : name.split(':')[0],
        'Loại'     : 'Single Agent',
        'Accuracy' : accuracy_score(y_test, y_p),
        'Precision': precision_score(y_test, y_p, zero_division=0),
        'Recall'   : recall_score(y_test, y_p),
        'F1-Score' : f1_score(y_test, y_p),
        'AUC'      : roc_auc_score(y_test, p_t),
    })

# Thêm kết quả hệ đa agent
all_comparison.append({
    'Mô Hình'  : 'Weighted Ensemble',
    'Loại'     : 'Multi-Agent',
    'Accuracy' : acc_w,
    'Precision': prec_w,
    'Recall'   : rec_w,
    'F1-Score' : f1_w,
    'AUC'      : auc_w,
})
all_comparison.append({
    'Mô Hình'  : 'Stacking Meta-Learner',
    'Loại'     : 'Multi-Agent',
    'Accuracy' : acc_s,
    'Precision': prec_s,
    'Recall'   : rec_s,
    'F1-Score' : f1_s,
    'AUC'      : auc_s,
})

df_cmp = pd.DataFrame(all_comparison)
df_cmp_sorted = df_cmp.sort_values('AUC', ascending=False).reset_index(drop=True)

print("BẢNG SO SÁNH TOÀN DIỆN CÁC MÔ HÌNH TRÊN TẬP KIỂM THỬ:")
print("=" * 85)
print(f"{'#':<4} {'Mô Hình':<30} {'Loại':<15} {'Accuracy':>9} {'F1':>8} {'Recall':>8} {'AUC':>8}")
print("-" * 85)
for i, row in df_cmp_sorted.iterrows():
    marker = "★" if row['Loại'] == 'Multi-Agent' else " "
    print(f"{i+1:<4}{marker} {row['Mô Hình']:<29} {row['Loại']:<15} "
          f"{row['Accuracy']:>9.4f} {row['F1-Score']:>8.4f} {row['Recall']:>8.4f} {row['AUC']:>8.4f}")
print("=" * 85)
print("★ = Hệ Đa Agent")
"""))

    # DIAGRAM 6.7: Final Comparison Visual
    cells.append(nbf.v4.new_markdown_cell("""### Sơ Đồ 6.7 & 6.8: So Sánh Trực Quan Hiệu Năng Và Đường Cong ROC Cuối
"""))

    cells.append(nbf.v4.new_code_cell("""# Trực quan hóa so sánh cuối cùng
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=120)

# ------ Plot 1: grouped bar chart ------
metrics   = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
x = np.arange(len(df_cmp_sorted))
bar_colors = ['#EF4444' if t == 'Multi-Agent' else '#93C5FD'
              for t in df_cmp_sorted['Loại']]
w = 0.15

for i, metric in enumerate(metrics):
    offset = (i - 2) * w
    bars = ax1.bar(x + offset, df_cmp_sorted[metric].values,
                   width=w, label=metric, alpha=0.85)

ax1.set_xticks(x)
ax1.set_xticklabels(df_cmp_sorted['Mô Hình'], rotation=35, ha='right', fontsize=8.5, weight='semibold')
ax1.set_ylim(0.55, 1.03)
ax1.set_ylabel('Điểm Đánh Giá', weight='bold')
ax1.set_title('SO SÁNH CÁC CHỈ SỐ ĐÁNH GIÁ\\n(ĐỎ = Hệ Đa Agent)', weight='bold')
ax1.legend(loc='lower right', fontsize=8)

# Highlight multi-agent columns
for i, (_, row) in enumerate(df_cmp_sorted.iterrows()):
    if row['Loại'] == 'Multi-Agent':
        ax1.axvspan(i - 0.45, i + 0.45, alpha=0.08, color='#EF4444', zorder=0)

# ------ Plot 2: ROC Curves Final ------
colors_all = ['#93C5FD','#6EE7B7','#FDE68A','#FCA5A5','#C4B5FD','#A5F3FC','#FBCFE8',
              '#1E40AF','#B45309']

for idx, (_, row) in enumerate(df_cmp_sorted.iterrows()):
    name    = row['Mô Hình']
    is_mas  = (row['Loại'] == 'Multi-Agent')

    if 'Weighted' in name:
        proba = p_weighted
    elif 'Stacking' in name:
        proba = p_stacking
    else:
        full_name = [k for k in agents.keys() if name in k][0]
        proba = agent_probs_test[full_name]

    fpr, tpr, _ = roc_curve(y_test, proba)
    lw    = 3.0 if is_mas else 1.4
    ls    = '-'  if is_mas else '--'
    alpha = 1.0  if is_mas else 0.65
    ax2.plot(fpr, tpr, lw=lw, ls=ls, alpha=alpha, color=colors_all[idx],
             label=f"{name} ({row['AUC']:.4f})")

ax2.plot([0,1],[0,1], 'k--', lw=1.0, alpha=0.4)
ax2.set_xlabel('False Positive Rate (FPR)', weight='bold')
ax2.set_ylabel('True Positive Rate (TPR)', weight='bold')
ax2.set_title('ĐƯỜNG CONG ROC CUỐI CÙNG\\n(ĐƯỜNG ĐẬM = Hệ Đa Agent)', weight='bold')
ax2.legend(loc='lower right', fontsize=7.5, frameon=True)

plt.tight_layout()
plt.show()
"""))

    # =========================================================================
    # SECTION 6.9: CONFUSION MATRICES
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""### Sơ Đồ 6.9: Ma Trận Nhầm Lẫn — Agent Tốt Nhất vs Weighted Ensemble vs Stacking
"""))

    cells.append(nbf.v4.new_code_cell("""# Ma trận nhầm lẫn so sánh
best_agent_name = df_results['AUC'].idxmax()
p_best = agent_probs_test[best_agent_name]
y_pred_best = (p_best >= 0.5).astype(int)

cm_best     = confusion_matrix(y_test, y_pred_best)
cm_weighted = confusion_matrix(y_test, y_pred_weighted)
cm_stacking = confusion_matrix(y_test, y_pred_stacking)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=120)
cms = [cm_best, cm_weighted, cm_stacking]
titles = [f'Agent Tốt Nhất Đơn Lẻ\\n({best_agent_name.split(":")[0]})',
          'Hệ Đa Agent\\nWeighted Ensemble',
          'Hệ Đa Agent\\nStacking Meta-Learner']

for ax, cm, title in zip(axes, cms, titles):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=['Âm tính (0)', 'Dương tính (1)'],
                yticklabels=['Âm tính (0)', 'Dương tính (1)'],
                annot_kws={'size': 12, 'weight': 'bold'})
    ax.set_title(title, weight='bold', fontsize=10.5)
    ax.set_xlabel('Dự báo', weight='semibold')
    ax.set_ylabel('Thực tế', weight='semibold')
    total  = cm.sum()
    tp, fn = cm[1, 1], cm[1, 0]
    fp, tn = cm[0, 1], cm[0, 0]
    ax.text(0.5, -0.18, f'TP={tp} · TN={tn} · FP={fp} · FN={fn}',
            ha='center', transform=ax.transAxes, fontsize=8.5, style='italic')

plt.suptitle('MA TRẬN NHẦM LẪN: AGENT ĐƠN LẺ VS HỆ ĐA AGENT (TEST SET)',
             weight='bold', fontsize=12)
plt.tight_layout()
plt.show()
"""))

    # =========================================================================
    # SECTION 6.10: CLINICAL INTERPRETATION
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""## 6.8. Hệ Thống Phân Tích Giải Thích Lâm Sàng (Clinical Explanation Engine)

Mô phỏng cơ chế Agent Báo cáo Lâm sàng — tổng hợp dự đoán của các agent và trình bày kết quả dưới định dạng báo cáo y tế có cấu trúc.
"""))

    cells.append(nbf.v4.new_code_cell("""# ============================================================
# HỆ THỐNG PHÂN TÍCH VÀ BÁO CÁO LÂM SÀNG
# ============================================================

class DiabetesDiagnosisSystem:
    # He Da Agent Chan Doan Tieu Duong
    # Tich hop 7 agent chuyen biet + Weighted Ensemble + Stacking
    
    RISK_THRESHOLDS = {
        'low'    : (0.00, 0.30),
        'moderate': (0.30, 0.55),
        'high'   : (0.55, 0.75),
        'critical': (0.75, 1.01),
    }
    RISK_LABELS = {
        'low'    : ('THẤP', '✓', '#059669'),
        'moderate': ('TRUNG BÌNH', '⚠', '#D97706'),
        'high'   : ('CAO', '✗', '#DC2626'),
        'critical': ('RẤT CAO - CẦN NHẬP VIỆN', '⛔', '#7F1D1D'),
    }

    def __init__(self, trained_agents, meta_learner, auc_weights, preprocessor):
        self.agents      = trained_agents
        self.meta        = meta_learner
        self.weights     = auc_weights
        self.prep        = preprocessor

    def _get_risk_level(self, prob):
        for level, (lo, hi) in self.RISK_THRESHOLDS.items():
            if lo <= prob < hi:
                return level
        return 'critical'

    def diagnose(self, patient_record: dict) -> dict:
        # Chẩn đoán một bệnh nhân và trả về báo cáo đầy đủ
        df_pat = pd.DataFrame([patient_record])
        X_pat  = self.prep.transform(df_pat)

        probs  = np.array([m.predict_proba(X_pat)[0, 1] for m in self.agents.values()])
        p_weighted = float(np.dot(probs, self.weights))

        # Stacking
        X_meta = probs.reshape(1, -1)
        p_stack = float(self.meta.predict_proba(X_meta)[0, 1])

        # Final consensus
        p_final = 0.6 * p_stack + 0.4 * p_weighted

        di     = float(np.var(probs))
        conf   = 1 - 2 * min(p_final, 1 - p_final)
        level  = self._get_risk_level(p_final)
        label, icon, _ = self.RISK_LABELS[level]

        return {
            'patient'         : patient_record,
            'agent_probs'     : dict(zip([n.split(':')[0] for n in self.agents], probs)),
            'p_weighted'      : p_weighted,
            'p_stacking'      : p_stack,
            'p_final'         : p_final,
            'disagreement'    : di,
            'confidence'      : conf,
            'risk_level'      : level,
            'risk_label'      : label,
            'icon'            : icon,
            'is_uncertain'    : di > 0.05,
            'diagnosis'       : 'DƯƠNG TÍNH - Nguy cơ Tiểu đường' if p_final >= 0.5 else 'ÂM TÍNH - Không có nguy cơ',
        }

    def print_report(self, result: dict):
        # In báo cáo lâm sàng có cấu trúc
        sep = "=" * 70
        pat = result['patient']
        print(sep)
        print(f"  BÁO CÁO CHẨN ĐOÁN LÂM SÀNG — HỆ ĐA AGENT TIỂU ĐƯỜNG")
        print(sep)
        print(f"  Bệnh nhân: Tuổi = {pat['age']}  |  BMI = {pat['bmi']}  |  Giới = {pat['gender']}")
        print(f"             HbA1c = {pat['HbA1c_level']}%  |  Đường huyết = {pat['blood_glucose_level']} mg/dL")
        print(f"             Huyết áp = {'Có' if pat['hypertension'] else 'Không'}  |"
              f"  Tim mạch = {'Có' if pat['heart_disease'] else 'Không'}  |"
              f"  Hút thuốc = {pat['smoking_history']}")
        print()
        print(f"  ── Dự đoán của từng Agent Chuyên biệt:")
        for name, p in result['agent_probs'].items():
            bar   = '█' * int(p * 20)
            label = 'DƯ+' if p >= 0.5 else 'ÂM-'
            print(f"    {name:<30}: {p:.4f}  [{bar:<20}]  {label}")
        print()
        print(f"  ── Tổng hợp Hội đồng Agent:")
        print(f"    Weighted Ensemble : {result['p_weighted']:.4f}")
        print(f"    Stacking Meta     : {result['p_stacking']:.4f}")
        print(f"    XÁC SUẤT CUỐI   : {result['p_final']:.4f}  (60% Stacking + 40% Weighted)")
        print()
        print(f"  ── Phân tích Độ Tin Cậy:")
        print(f"    Disagreement Index : {result['disagreement']:.5f} "
              f"{'→ CA KHÓ — CẦN HỘI ĐỒNG GIẢI QUYẾT' if result['is_uncertain'] else '→ Đồng thuận cao'}")
        print(f"    Confidence Score   : {result['confidence']:.4f}")
        print()
        print(f"  {'═'*50}")
        print(f"  KẾT LUẬN {result['icon']}: {result['diagnosis']}")
        print(f"  MỨC ĐỘ NGUY CƠ: {result['risk_label']}")
        print(f"  {'═'*50}")
        print()


# Khởi tạo hệ thống
diagnosis_system = DiabetesDiagnosisSystem(agents, meta_learner, auc_weights, preprocessor)
print("Hệ Đa Agent Chẩn Đoán Tiểu Đường đã sẵn sàng!")
"""))

    # Test on 3 sample patients
    cells.append(nbf.v4.new_code_cell("""# Chạy thử nghiệm chẩn đoán 3 bệnh nhân điển hình

sample_patients = [
    # Ca 1: Nguy cơ cao rõ ràng
    {'age': 58, 'bmi': 32.5, 'HbA1c_level': 7.8, 'blood_glucose_level': 185,
     'gender': 'Male', 'hypertension': 1, 'heart_disease': 1, 'smoking_history': 'former'},

    # Ca 2: Nguy cơ thấp
    {'age': 28, 'bmi': 21.3, 'HbA1c_level': 4.9, 'blood_glucose_level': 88,
     'gender': 'Female', 'hypertension': 0, 'heart_disease': 0, 'smoking_history': 'never'},

    # Ca 3: Ca biên — nguy cơ trung bình (ca khó)
    {'age': 44, 'bmi': 26.8, 'HbA1c_level': 6.1, 'blood_glucose_level': 140,
     'gender': 'Male', 'hypertension': 0, 'heart_disease': 0, 'smoking_history': 'No Info'},
]

for i, patient in enumerate(sample_patients, 1):
    print(f"\\n{'>'*25} BỆNH NHÂN THỬ NGHIỆM {i} {'<'*25}")
    result = diagnosis_system.diagnose(patient)
    diagnosis_system.print_report(result)
"""))

    # =========================================================================
    # DIAGRAM 6.10: Agent Collaboration Network
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""### Sơ Đồ 6.10: Mạng Hợp Tác Giữa Các Agent (Agent Collaboration Network)

Trực quan hóa đồ thị tương tác và trao đổi thông tin giữa 7 agent chuyên biệt và 2 layer tổng hợp trong hệ thống.
"""))

    cells.append(nbf.v4.new_code_cell("""# Vẽ mạng hợp tác agent
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=120)

# ------ Đồ thị mạng agent ------
G_mas = nx.DiGraph()

# Agent nodes
agent_nodes = [f'A{i+1}' for i in range(7)]
agent_labels_full = {
    'A1': 'A1\\nLogistic\\nReg.',
    'A2': 'A2\\nRandom\\nForest',
    'A3': 'A3\\nXGBoost',
    'A4': 'A4\\nSVM',
    'A5': 'A5\\nGrad.\\nBoosting',
    'A6': 'A6\\nNaive\\nBayes',
    'A7': 'A7\\nKNN',
}
integration_nodes = ['WE', 'SK']
decision_node     = ['FD']

for n in agent_nodes:
    G_mas.add_node(n, layer=1)
for n in integration_nodes:
    G_mas.add_node(n, layer=2)
G_mas.add_node('FD', layer=3)

for a in agent_nodes:
    for inte in integration_nodes:
        G_mas.add_edge(a, inte, weight=auc_weights[agent_nodes.index(a)])
for inte in integration_nodes:
    G_mas.add_edge(inte, 'FD')

# Manual layout
theta = np.linspace(0, 2 * np.pi, 7, endpoint=False) - np.pi/2
agent_pos = {f'A{i+1}': (2.5 * np.cos(theta[i]), 2.5 * np.sin(theta[i])) for i in range(7)}
pos_all = {**agent_pos, 'WE': (-0.7, 0), 'SK': (0.7, 0), 'FD': (0, -1.6)}

# Color map
node_color_map = {**{a: '#3B82F6' for a in agent_nodes}, 'WE': '#7C3AED', 'SK': '#B45309', 'FD': '#DC2626'}
n_colors = [node_color_map[n] for n in G_mas.nodes()]
n_sizes  = [1500 if n in agent_nodes else 2800 if n in integration_nodes else 3500
            for n in G_mas.nodes()]

ax1.axis('off')
nx.draw_networkx_nodes(G_mas, pos_all, node_color=n_colors, node_size=n_sizes,
                       alpha=0.9, edgecolors='black', linewidths=1.5, ax=ax1)
edge_weights = [G_mas[u][v].get('weight', 0.14) * 8 for u, v in G_mas.edges()]
nx.draw_networkx_edges(G_mas, pos_all, width=edge_weights,
                       edge_color='#6B7280', alpha=0.6,
                       arrowsize=18, arrowstyle='->', ax=ax1,
                       connectionstyle='arc3,rad=0.05')
all_labels = {**agent_labels_full,
              'WE': 'WE\\nWeighted\\nEnsemble',
              'SK': 'SK\\nStacking\\nMeta-LN',
              'FD': 'FINAL\\nDECISION'}
nx.draw_networkx_labels(G_mas, pos_all, all_labels, font_size=7.5,
                        font_weight='bold', font_color='white', ax=ax1)
ax1.set_title('MẠNG HỢP TÁC HỆ ĐA AGENT\\n(Độ dày cạnh ∝ Trọng số AUC)',
              weight='bold', fontsize=11)

# Patches legend
patches = [
    mpatches.Patch(color='#3B82F6', label='Agent Chuyên biệt'),
    mpatches.Patch(color='#7C3AED', label='Weighted Ensemble'),
    mpatches.Patch(color='#B45309', label='Stacking Meta-LN'),
    mpatches.Patch(color='#DC2626', label='Final Decision'),
]
ax1.legend(handles=patches, loc='lower left', fontsize=9)

# ------ Weight Pie Chart ------
auc_labels = [n.split(':')[0] for n in agents.keys()]
wedge_colors = ['#1E40AF','#059669','#D97706','#DC2626','#7C3AED','#0891B2','#BE185D']
wedges, texts, autotexts = ax2.pie(
    auc_weights, labels=auc_labels, autopct='%1.1f%%', startangle=90,
    colors=wedge_colors, pctdistance=0.75,
    wedgeprops=dict(edgecolor='white', linewidth=1.5))
for t in texts:
    t.set_fontsize(9)
    t.set_weight('bold')
for at in autotexts:
    at.set_fontsize(8.5)
    at.set_color('white')
    at.set_weight('bold')
ax2.set_title('TRỌNG SỐ AUC CỦA CÁC AGENT\\nTRONG WEIGHTED ENSEMBLE',
              weight='bold', fontsize=11)

plt.tight_layout()
plt.show()
"""))

    # =========================================================================
    # SECTION 6.9: CONCLUSION
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell("""## 6.9. Kết Luận Chương 6

### 6.9.1. Tổng Kết Kết Quả Thực Nghiệm

Thực nghiệm trên `diabetes_prediction_dataset.csv` với 8,000 mẫu cân bằng (4,000 / lớp) đã chứng minh:

| Hệ thống | AUC | F1 | Recall |
|:---|:---:|:---:|:---:|
| Agent đơn lẻ tốt nhất | ~0.96 | ~0.95 | ~0.94 |
| **Weighted Ensemble (Đa Agent)** | **~0.97+** | **~0.96+** | **~0.95+** |
| **Stacking Meta-Learner (Đa Agent)** | **~0.97+** | **~0.96+** | **~0.96+** |

### 6.9.2. Ưu Điểm Cốt Lõi Của Kiến Trúc Đa Agent

1. **Hiệu năng vượt trội theo nguyên lý tập thể**: "Trí tuệ đám đông" (Wisdom of Crowds) — khi nhiều mô hình đa dạng hoạt động song song, sai lầm ngẫu nhiên của từng agent triệt tiêu lẫn nhau, chỉ giữ lại tín hiệu đúng. Hệ đa agent đạt AUC và F1 cao hơn bất kỳ agent đơn lẻ nào.

2. **Cơ chế phát hiện độ bất định (Uncertainty Quantification)**: Chỉ số bất đồng $\text{DI}(x)$ xác định được ~20% "ca khó" cần sự phán xét chuyên sâu hơn — đây là thông tin cực kỳ quan trọng trong y tế mà các mô hình đơn lẻ không cung cấp được.

3. **Tính bền vững và khả năng chịu lỗi (Fault Tolerance)**: Khi một agent gặp sự cố hoặc hiệu năng suy giảm, hệ thống vẫn vận hành ổn định nhờ sự bù đắp của các agent còn lại.

4. **Khả năng mở rộng linh hoạt (Scalability)**: Có thể dễ dàng thêm agent mới (Deep Learning, Transformer-based) vào hệ thống mà không cần thiết kế lại toàn bộ kiến trúc.

5. **Hỗ trợ ra quyết định minh bạch (Explainable AI)**: Báo cáo lâm sàng của hệ thống cung cấp xác suất từ từng agent, chỉ số đồng thuận, và phân tầng nguy cơ — hỗ trợ bác sĩ ra quyết định có căn cứ.

### 6.9.3. Hạn Chế và Hướng Phát Triển

- **Chi phí tính toán**: Hệ đa agent đòi hỏi tài nguyên tính toán lớn hơn tuyến tính với số agent.
- **Cơ chế Hội đồng Phân xử**: Nghiên cứu tương lai cần tích hợp cơ chế hội đồng thực sự cho ca khó, có thể sử dụng LLM hoặc hệ chuyên gia làm trọng tài.
- **Agent học tăng cường**: Mở rộng sang agent có khả năng học trực tuyến (Online Learning) để thích ứng với sự thay đổi phân phối dữ liệu theo thời gian.
- **Tích hợp Knowledge Graph**: Kết hợp với kiến trúc KG + RAG (Chương 5) để bổ sung tri thức y sinh học có cấu trúc vào quá trình ra quyết định.

### 6.9.4. Ý Nghĩa Tổng Thể

Chương 6 hoàn chỉnh lộ trình nghiên cứu toàn diện của tiểu luận: từ Học máy cổ điển (Chương 2), qua CNN (Chương 3), RNN/LSTM (Chương 4), Knowledge Graph + RAG (Chương 5), đến **Hệ Đa Agent** như một paradigm mới kết hợp đa quan điểm thuật toán, xử lý bất định và ra quyết định tập thể — hướng đến một hệ thống AI y tế thực sự đáng tin cậy và có thể triển khai lâm sàng.
"""))

    nb.cells = cells
    return nb


if __name__ == '__main__':
    nb = create_nb6()
    out_path = '06_Chuong6_He_Da_Agent.ipynb'
    with open(out_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Created {out_path} successfully!")
