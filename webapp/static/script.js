let currentChapter = 'chuong6';
let predictionChart = null;

// =============================================================================
// CẤU HÌNH THAM SỐ VÀ GIAO DIỆN CỦA TỪNG CHƯƠNG
// =============================================================================
const chapterConfigs = {
    chuong1: {
        subtaskLabel: 'Hàm Logic Kiểm Chứng',
        subtasks: [
            { id: 'XOR', name: 'Hàm Logic XOR (Bất Khả Ly Tuyến Tính - Rào Cản Minsky)' },
            { id: 'AND', name: 'Hàm Logic AND (Khả Ly Tuyến Tính)' },
            { id: 'OR', name: 'Hàm Logic OR (Khả Ly Tuyến Tính)' }
        ],
        modelLabel: 'Kiến Trúc Mạng Nơ-ron',
        models: [
            { id: 'perceptron', name: 'Perceptron Đơn Lớp (1957)', checked: false },
            { id: 'mlp', name: 'Multi-Layer Perceptron (MLP)', checked: true }
        ],
        inputs: [
            { id: 'x1', label: 'Tín hiệu đầu vào x1', type: 'select', options: [{ val: 1, text: '1 (Kích hoạt)' }, { val: 0, text: '0 (Không kích hoạt)' }] },
            { id: 'x2', label: 'Tín hiệu đầu vào x2', type: 'select', options: [{ val: 1, text: '1 (Kích hoạt)' }, { val: 0, text: '0 (Không kích hoạt)' }] }
        ],
        info: 'Chương 1 kiểm chứng định lý hội tụ Novikoff và sự bất lực của Perceptron đơn lớp trước bài toán XOR, giải quyết triệt để nhờ tầng ẩn phi tuyến của MLP.'
    },

    chuong2: {
        subtaskLabel: 'Bài Toán Học Máy Cổ Điển',
        subtasks: [
            { id: 'diabetes', name: 'Phân loại Bệnh Tiểu Đường (diabetes_prediction_dataset.csv)' },
            { id: 'housing', name: 'Hồi quy Định Giá Nhà Đất Việt Nam (vietnam_housing_dataset.csv)' }
        ],
        modelLabel: 'Thuật Toán Học Máy',
        models: [
            { id: 'Logistic Regression', name: 'Logistic / Linear', checked: false },
            { id: 'Random Forest', name: 'Random Forest', checked: true },
            { id: 'XGBoost', name: 'XGBoost', checked: false },
            { id: 'SVM', name: 'SVM (RBF Kernel)', checked: false }
        ],
        inputsDiabetes: [
            { id: 'blood_glucose_level', label: 'Đường huyết (mg/dL)', type: 'number', val: 145 },
            { id: 'HbA1c_level', label: 'Chỉ số HbA1c (%)', type: 'number', step: '0.1', val: 6.8 },
            { id: 'bmi', label: 'Chỉ số BMI (kg/m²)', type: 'number', step: '0.1', val: 28.5 },
            { id: 'age', label: 'Độ tuổi', type: 'number', val: 52 },
            { id: 'hypertension', label: 'Tăng huyết áp', type: 'select', options: [{ val: 0, text: 'Không' }, { val: 1, text: 'Có' }] }
        ],
        inputsHousing: [
            { id: 'area', label: 'Diện tích nhà (m²)', type: 'number', val: 85 },
            { id: 'floors', label: 'Số tầng', type: 'number', val: 3 },
            { id: 'bedrooms', label: 'Số phòng ngủ', type: 'number', val: 3 },
            { id: 'frontage', label: 'Mặt tiền (m)', type: 'number', step: '0.5', val: 4.5 },
            { id: 'access_road', label: 'Đường vào (m)', type: 'number', step: '0.5', val: 5.0 }
        ]
    },

    chuong3: {
        subtaskLabel: 'Bài Toán Mạng Nơ-ron Tích Chập (CNN)',
        subtasks: [
            { id: 'conv1d', name: 'Deep 1D-CNN Phân Loại Đặc Trưng Y Tế' },
            { id: 'conv2d', name: 'Toán Tử Tích Chập 2D (Sobel Filter Trích Xuất Biên Cạnh)' }
        ],
        modelLabel: 'Bộ Lọc / Kiến Trúc CNN',
        models: [
            { id: 'Deep_1D_CNN', name: 'Deep 1D-CNN (Keras/TF)', checked: true },
            { id: 'Sobel_Vertical', name: 'Sobel Dọc (NumPy)', checked: false },
            { id: 'Sobel_Horizontal', name: 'Sobel Ngang (NumPy)', checked: false }
        ],
        inputsConv1D: [
            { id: 'blood_glucose_level', label: 'Đường huyết (mg/dL)', type: 'number', val: 150 },
            { id: 'HbA1c_level', label: 'Chỉ số HbA1c (%)', type: 'number', step: '0.1', val: 7.0 },
            { id: 'bmi', label: 'Chỉ số BMI (kg/m²)', type: 'number', step: '0.1', val: 29.0 }
        ],
        inputsConv2D: []
    },

    chuong4: {
        subtaskLabel: 'Thị Trường Tài Chính Chuỗi Thời Gian',
        subtasks: [
            { id: 'vnindex', name: 'Chỉ Số Chứng Khoán VN-Index (HOSE_DLY_VNINDEX1D.csv)' },
            { id: 'gold', name: 'Giá Vàng SJC Việt Nam (vietnam_gold_price.csv)' },
            { id: 'amazon', name: 'Cổ Phiếu Amazon (AMZN.csv)' }
        ],
        modelLabel: 'Kiến Trúc Mạng Hồi Quy (TensorFlow Custom Layer)',
        models: [
            { id: 'Vanilla LSTM', name: 'Vanilla LSTM', checked: true },
            { id: 'Stacked LSTM', name: 'Stacked LSTM', checked: false },
            { id: 'Bidirectional LSTM', name: 'BiLSTM', checked: false },
            { id: 'SimpleRNN', name: 'SimpleRNN', checked: false }
        ],
        inputs: [],
        info: 'Mô hình sẽ tự động trích xuất chuỗi lịch sử 30 phiên giao dịch gần nhất từ file dữ liệu thực tế tại Việt Nam để dự báo đường xu hướng 7 phiên tiếp theo.'
    },

    chuong5: {
        subtaskLabel: 'Miền Tri Thức Hỏi Đáp',
        subtasks: [
            { id: 'ck', name: 'Miền Chứng Khoán Việt Nam (HOSE, VN30, P/E)' },
            { id: 'gold', name: 'Miền Giá Vàng Việt Nam (SJC, 9999, Nghị định 24)' },
            { id: 'med', name: 'Miền Y Tế & Tiêu Chuẩn ADA Tiểu Đường' }
        ],
        modelLabel: 'Cơ Chế Truy Xuất Tri Thức',
        models: [
            { id: 'Hybrid_KG_RAG', name: 'Đồ Thị Tri Thức + Vector RAG (TF-IDF Cosine)', checked: true }
        ],
        inputs: [
            { id: 'query', label: 'Câu hỏi truy vấn chuyên sâu', type: 'text', val: 'Quy định biên độ sàn HOSE là bao nhiêu và VN30 gồm những cổ phiếu nào?' }
        ]
    },

    chuong6: {
        subtaskLabel: 'Mục Tiêu Hội Đồng Đa Agent',
        subtasks: [
            { id: 'council', name: 'Hội Đồng 5 Diagnostic Agents + Rule Engine + Safety Advisor' }
        ],
        modelLabel: 'Cơ Chế Bỏ Phiếu Hội Đồng (MAS)',
        models: [
            { id: 'Weighted_Consensus', name: 'Đồng Thuận Có Trọng Số (AUC-Weighted)', checked: true }
        ],
        inputs: [
            { id: 'age', label: 'Tuổi bệnh nhân', type: 'number', val: 52 },
            { id: 'gender', label: 'Giới tính', type: 'select', options: [{ val: 'Male', text: 'Nam' }, { val: 'Female', text: 'Nữ' }] },
            { id: 'blood_glucose_level', label: 'Đường huyết đói (mg/dL)', type: 'number', val: 165 },
            { id: 'HbA1c_level', label: 'Chỉ số HbA1c (%)', type: 'number', step: '0.1', val: 7.2 },
            { id: 'bmi', label: 'Chỉ số BMI (kg/m²)', type: 'number', step: '0.1', val: 30.5 },
            { id: 'hypertension', label: 'Tăng huyết áp', type: 'select', options: [{ val: 1, text: 'Có' }, { val: 0, text: 'Không' }] },
            { id: 'heart_disease', label: 'Bệnh tim mạch', type: 'select', options: [{ val: 0, text: 'Không' }, { val: 1, text: 'Có' }] },
            { id: 'smoking_history', label: 'Tiền sử hút thuốc', type: 'select', options: [{ val: 'former', text: 'Đã từng hút' }, { val: 'never', text: 'Chưa từng' }, { val: 'current', text: 'Hiện đang hút' }] }
        ]
    }
};

// =============================================================================
// CHUYỂN ĐỔI CHƯƠNG VÀ CẬP NHẬT GIAO DIỆN FORM
// =============================================================================
function setupChapter(chKey) {
    currentChapter = chKey;
    const cfg = chapterConfigs[chKey];

    // Cập nhật Subtask Dropdown
    const subtaskGroup = document.getElementById('subtask-group');
    const subtaskSelect = document.getElementById('subtask-select');
    const subtaskLabel = document.getElementById('subtask-label');

    subtaskLabel.textContent = cfg.subtaskLabel;
    subtaskSelect.innerHTML = '';
    cfg.subtasks.forEach(s => {
        const opt = document.createElement('option');
        opt.value = s.id;
        opt.textContent = s.name;
        subtaskSelect.appendChild(opt);
    });

    // Cập nhật Model Radios
    const modelRadios = document.getElementById('model-radios');
    const modelLabel = document.getElementById('model-label');
    modelLabel.textContent = cfg.modelLabel;
    modelRadios.innerHTML = '';

    cfg.models.forEach((m, idx) => {
        modelRadios.innerHTML += `
            <label class="radio-card">
                <input type="radio" name="model_choice" value="${m.id}" ${m.checked ? 'checked' : ''} required>
                <span class="card-content">${m.name}</span>
            </label>
        `;
    });

    // Render dynamic inputs
    renderInputsForCurrentState();

    // Ẩn kết quả cũ
    document.getElementById('result-container').classList.add('hidden');
    document.getElementById('chart-container').classList.add('hidden');
    document.getElementById('rag-details').classList.add('hidden');
    document.getElementById('agent-details').classList.add('hidden');
}

function renderInputsForCurrentState() {
    const dynamicInputs = document.getElementById('dynamic-inputs');
    dynamicInputs.innerHTML = '';

    const subtaskVal = document.getElementById('subtask-select').value;
    const cfg = chapterConfigs[currentChapter];

    let inputsToRender = [];

    if (currentChapter === 'chuong1') {
        inputsToRender = cfg.inputs;
        if (cfg.info) {
            dynamicInputs.innerHTML += `<div class="info-text">${cfg.info}</div>`;
        }
    } else if (currentChapter === 'chuong2') {
        inputsToRender = (subtaskVal === 'housing') ? cfg.inputsHousing : cfg.inputsDiabetes;
    } else if (currentChapter === 'chuong3') {
        inputsToRender = (subtaskVal === 'conv1d') ? cfg.inputsConv1D : cfg.inputsConv2D;
        if (subtaskVal === 'conv2d') {
            dynamicInputs.innerHTML += `<div class="info-text">Mô phỏng phép nhân chập ma trận 3x3 Sobel trượt qua ảnh 5x5 để làm nổi bật đường biên theo phương giải tích.</div>`;
        }
    } else if (currentChapter === 'chuong4') {
        dynamicInputs.innerHTML += `<div class="info-text">${cfg.info}</div>`;
    } else if (currentChapter === 'chuong5') {
        inputsToRender = cfg.inputs;
        // Gợi ý câu hỏi mẫu khi đổi subtask
        if (subtaskVal === 'ck') {
            inputsToRender[0].val = 'Quy định biên độ sàn HOSE là bao nhiêu và VN30 gồm những cổ phiếu nào?';
        } else if (subtaskVal === 'gold') {
            inputsToRender[0].val = 'Vàng miếng SJC chịu sự quản lý theo nghị định nào và yếu tố nào chi phối giá vàng?';
        } else {
            inputsToRender[0].val = 'Mức đường huyết lúc đói bao nhiêu thì được chẩn đoán mắc bệnh đái tháo đường?';
        }
    } else if (currentChapter === 'chuong6') {
        inputsToRender = cfg.inputs;
    }

    // Render các thẻ input
    inputsToRender.forEach(inp => {
        let fieldHtml = '';
        if (inp.type === 'select') {
            let opts = inp.options.map(o => `<option value="${o.val}">${o.text}</option>`).join('');
            fieldHtml = `<select id="${inp.id}">${opts}</select>`;
        } else {
            fieldHtml = `<input type="${inp.type || 'text'}" id="${inp.id}" value="${inp.val !== undefined ? inp.val : ''}" ${inp.step ? `step="${inp.step}"` : ''} required>`;
        }
        dynamicInputs.innerHTML += `
            <div class="input-item">
                <label for="${inp.id}">${inp.label}</label>
                ${fieldHtml}
            </div>
        `;
    });
}

function onSubtaskChange() {
    renderInputsForCurrentState();
}

// Bắt sự kiện chuyển tab chương
document.querySelectorAll('.chapter-btn').forEach(btn => {
    btn.addEventListener('click', function() {
        document.querySelectorAll('.chapter-btn').forEach(b => b.classList.remove('active'));
        this.classList.add('active');
        setupChapter(this.getAttribute('data-chapter'));
    });
});

// =============================================================================
// SUBMIT FORM VÀ HIỂN THỊ KẾT QUẢ ĐA NĂNG
// =============================================================================
document.getElementById('prediction-form').addEventListener('submit', async function(e) {
    e.preventDefault();

    const submitBtn = document.getElementById('submit-btn');
    const resultContainer = document.getElementById('result-container');
    const resultValue = document.getElementById('result-value');
    const resultConfidence = document.getElementById('result-confidence');
    const modelBadge = document.getElementById('model-badge');
    const chartContainer = document.getElementById('chart-container');
    const ragDetails = document.getElementById('rag-details');
    const agentDetails = document.getElementById('agent-details');

    // UI Loading state
    submitBtn.classList.add('loading');
    submitBtn.disabled = true;
    resultContainer.classList.add('hidden');
    chartContainer.classList.add('hidden');
    ragDetails.classList.add('hidden');
    agentDetails.classList.add('hidden');

    const subtask = document.getElementById('subtask-select').value;
    const modelChecked = document.querySelector('input[name="model_choice"]:checked');
    const modelChoice = modelChecked ? modelChecked.value : '';

    // Thu thập payload theo từng chương
    const payload = {
        chapter: currentChapter,
        subtask: subtask,
        model_name: modelChoice,
        model_type: modelChoice,
        architecture: modelChoice
    };

    // Lấy giá trị của tất cả inputs động hiện diện
    document.querySelectorAll('#dynamic-inputs input, #dynamic-inputs select').forEach(el => {
        payload[el.id] = el.value;
    });

    if (currentChapter === 'chuong1') {
        payload.gate = subtask;
    } else if (currentChapter === 'chuong3') {
        payload.cnn_mode = subtask;
        payload.kernel_type = modelChoice;
    } else if (currentChapter === 'chuong4') {
        payload.market = subtask;
    }

    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await res.json();

        setTimeout(() => {
            if (data.success) {
                resultValue.textContent = data.prediction;
                resultConfidence.textContent = data.confidence !== undefined ? data.confidence : '--';
                modelBadge.textContent = data.model_used || 'AI Model';

                // 1. Dạng biểu đồ chuỗi thời gian (Chương 4)
                if (data.type === 'chart' && data.chartData) {
                    chartContainer.classList.remove('hidden');
                    renderTimeSeriesChart(data.chartData, data.unit || 'Giá trị');
                }

                // 2. Dạng RAG & Knowledge Graph (Chương 5)
                if (data.type === 'rag_qa') {
                    ragDetails.classList.remove('hidden');
                    let factsHtml = data.kg_facts.map(f => `<li><code>${f}</code></li>`).join('');
                    let docsHtml = data.sources.map(s => `
                        <div class="citation-box">
                            <strong>[${s.id}] ${s.title}</strong> (Độ tương đồng: ${(s.score * 100).toFixed(1)}%)<br>
                            <em>"${s.content}"</em>
                        </div>
                    `).join('');

                    ragDetails.innerHTML = `
                        <div style="margin-top: 10px;">
                            <strong>1. Căn Cứ Từ Đồ Thị Tri Thức (Knowledge Graph):</strong>
                            <ul style="padding-left: 20px; margin-top: 4px;">${factsHtml}</ul>
                        </div>
                        <div style="margin-top: 10px;">
                            <strong>2. Đoạn Trích Dẫn Nguồn RAG (Vector Search):</strong>
                            ${docsHtml}
                        </div>
                        <div style="margin-top: 10px; font-weight: 600; color: #1e3a8a;">
                            💬 Trả lời tổng hợp: ${data.answer}
                        </div>
                    `;
                }

                // 3. Dạng Hệ Đa Agent (Chương 6)
                if (data.type === 'multi_agent') {
                    agentDetails.classList.remove('hidden');

                    let rows = '';
                    for (let [aName, vote] of Object.entries(data.agent_votes)) {
                        let badgeClass = (vote.decision === 'Dương tính') ? 'badge-pos' : 'badge-neg';
                        rows += `
                            <tr>
                                <td><strong>${aName}</strong></td>
                                <td><span class="${badgeClass}">${vote.decision}</span></td>
                                <td>${(vote.prob * 100).toFixed(1)}%</td>
                                <td>${(vote.weight * 100).toFixed(1)}%</td>
                            </tr>
                        `;
                    }

                    let ruleAlerts = data.clinical_rules.length > 0 
                        ? data.clinical_rules.map(r => `<div class="rule-alert">⚠️ ${r}</div>`).join('')
                        : `<div style="color: #059669; font-size: 0.82rem; margin-top: 4px;">✓ Các chỉ số nằm trong giới hạn kiểm soát chuẩn ADA.</div>`;

                    let recsList = data.recommendations.map(r => `<li>${r}</li>`).join('');

                    agentDetails.innerHTML = `
                        <div style="margin-top: 8px;">
                            <strong>1. Bỏ Phiếu Của 5 Diagnostic Agents (Weighted Consensus):</strong>
                            <table class="agent-table">
                                <thead>
                                    <tr>
                                        <th>Chuyên Viên Agent</th>
                                        <th>Quyết Định</th>
                                        <th>Xác Suất</th>
                                        <th>Trọng Số AUC</th>
                                    </tr>
                                </thead>
                                <tbody>${rows}</tbody>
                            </table>
                            <div style="margin-top: 6px; font-size: 0.8rem; color: #475569;">
                                Chỉ số bất đồng tập thể: <strong>${data.disagreement_index}</strong> 
                                ${data.is_uncertain ? '<span style="color: #dc2626; font-weight: bold;">(Cảnh báo Ca Khó - Bất Đồng Cao)</span>' : '<span style="color: #059669;">(Đồng thuận cao)</span>'}
                            </div>
                        </div>

                        <div style="margin-top: 12px;">
                            <strong>2. Thẩm Định Lâm Sàng Từ Rule Engine Agent:</strong>
                            ${ruleAlerts}
                        </div>

                        <div style="margin-top: 12px;">
                            <strong>3. Khuyến Nghị Từ Lifestyle & Safety Agents:</strong>
                            <ul style="padding-left: 18px; margin-top: 4px; color: #334155;">
                                ${recsList}
                            </ul>
                        </div>
                    `;
                }

                // 4. Giải thích thêm nếu có
                if (data.explanation) {
                    ragDetails.classList.remove('hidden');
                    ragDetails.innerHTML = `<div class="citation-box" style="border-left-color: #2563eb;">${data.explanation}</div>`;
                }

                resultContainer.classList.remove('hidden');
            } else {
                alert('Lỗi: ' + data.error);
            }

            submitBtn.classList.remove('loading');
            submitBtn.disabled = false;
        }, 300);

    } catch (err) {
        console.error('Lỗi kết nối:', err);
        alert('Đã xảy ra lỗi khi gửi yêu cầu tới server.');
        submitBtn.classList.remove('loading');
        submitBtn.disabled = false;
    }
});

// =============================================================================
// HÀM VẼ BIỂU ĐỒ CHUỖI THỜI GIAN (CHART.JS)
// =============================================================================
function renderTimeSeriesChart(chartData, unitLabel) {
    const ctx = document.getElementById('predictionChart').getContext('2d');

    if (predictionChart) {
        predictionChart.destroy();
    }

    predictionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartData.labels,
            datasets: [
                {
                    label: 'Quá khứ (Thực tế)',
                    data: chartData.past,
                    borderColor: '#2563eb',
                    backgroundColor: 'rgba(37, 99, 235, 0.08)',
                    borderWidth: 2.2,
                    tension: 0.3,
                    fill: true,
                    pointRadius: 2
                },
                {
                    label: 'Tương lai (Dự báo LSTM)',
                    data: chartData.future,
                    borderColor: '#dc2626',
                    borderWidth: 2.2,
                    borderDash: [5, 5],
                    tension: 0.3,
                    fill: false,
                    pointRadius: 3
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: { color: '#0f172a', font: { weight: '600', size: 11 } }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Phiên Giao Dịch', color: '#64748b', font: { size: 10 } },
                    ticks: { color: '#64748b', font: { size: 9 }, maxTicksLimit: 12 },
                    grid: { color: 'rgba(0,0,0,0.04)' }
                },
                y: {
                    title: { display: true, text: unitLabel, color: '#64748b', font: { size: 10 } },
                    ticks: { color: '#64748b', font: { size: 10 } },
                    grid: { color: 'rgba(0,0,0,0.06)' }
                }
            }
        }
    });
}

// Khởi tạo tab mặc định khi tải trang
setupChapter('chuong6');
