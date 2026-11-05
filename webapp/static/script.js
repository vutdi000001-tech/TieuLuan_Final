let currentModel = 'multi_agent';
let predictionChart = null;

// =============================================================================
// CẤU HÌNH THAM SỐ VÀ GIAO DIỆN CỦA TỪNG MÔ HÌNH (ĐÃ LOẠI BỎ CHƯƠNG 1, 2)
// =============================================================================
const modelConfigs = {
    cnn: {
        id: 'cnn',
        subtaskLabel: 'Nhiệm Vụ Mạng Tích Chập (CNN)',
        subtasks: [
            { id: 'conv1d', name: 'Deep 1D-CNN Trích Xuất Đặc Trưng Y Sinh' },
            { id: 'conv2d', name: 'Toán Tử Tích Chập 2D (Sobel Filter Trích Xuất Biên Cạnh)' }
        ],
        modelLabel: 'Kiến Trúc / Bộ Lọc Tích Chập',
        models: [
            { id: 'Deep_1D_CNN', name: 'Deep 1D-CNN (Conv1D + MaxPool)', checked: true },
            { id: 'Sobel_Vertical', name: 'Sobel Dọc (NumPy Kernel 3x3)', checked: false },
            { id: 'Sobel_Horizontal', name: 'Sobel Ngang (NumPy Kernel 3x3)', checked: false }
        ],
        inputsConv1D: [
            { id: 'blood_glucose_level', label: 'Đường huyết đói (mg/dL)', type: 'number', val: 155 },
            { id: 'HbA1c_level', label: 'Chỉ số HbA1c (%)', type: 'number', step: '0.1', val: 7.0 },
            { id: 'bmi', label: 'Chỉ số BMI (kg/m²)', type: 'number', step: '0.1', val: 29.5 }
        ],
        inputsConv2D: []
    },

    rnn_lstm: {
        id: 'rnn_lstm',
        subtaskLabel: 'Chuỗi Thời Gian Thị Trường Tài Chính',
        subtasks: [
            { id: 'gold', name: 'Giá Vàng SJC Việt Nam (vietnam_gold_price.csv)' },
            { id: 'vnindex', name: 'Chỉ Số Chứng Khoán VN-Index (HOSE_DLY_VNINDEX1D.csv)' },
            { id: 'amazon', name: 'Cổ Phiếu Amazon (AMZN.csv)' }
        ],
        modelLabel: 'Kiến Trúc Mạng Nơ-ron Hồi Quy',
        models: [
            { id: 'Vanilla LSTM', name: 'Vanilla LSTM', checked: true },
            { id: 'Stacked LSTM', name: 'Stacked LSTM', checked: false },
            { id: 'Bidirectional LSTM', name: 'BiLSTM', checked: false },
            { id: 'SimpleRNN', name: 'SimpleRNN', checked: false }
        ],
        inputs: [],
        info: 'Mô hình tự động trích xuất chuỗi lịch sử 30 phiên giao dịch gần nhất từ file dữ liệu thực tế để dự báo đường xu hướng 7 phiên tiếp theo.'
    },

    kg_rag: {
        id: 'kg_rag',
        subtaskLabel: 'Miền Tri Thức Hỏi Đáp',
        subtasks: [
            { id: 'gold', name: 'Miền Giá Vàng Việt Nam (SJC, 9999, Nghị định 24)' },
            { id: 'ck', name: 'Miền Chứng Khoán Việt Nam (HOSE, VN30, P/E)' },
            { id: 'med', name: 'Miền Y Tế & Tiêu Chuẩn ADA Tiểu Đường' }
        ],
        modelLabel: 'Cơ Chế Truy Xuất Tri Thức',
        models: [
            { id: 'Hybrid_KG_RAG', name: 'Đồ Thị Tri Thức + Vector RAG (TF-IDF Cosine)', checked: true }
        ],
        inputs: [
            { id: 'query', label: 'Câu hỏi truy vấn chuyên sâu', type: 'text', val: 'Vàng miếng SJC chịu sự quản lý theo nghị định nào và yếu tố nào chi phối giá vàng?' }
        ]
    },

    multi_agent: {
        id: 'multi_agent',
        subtaskLabel: 'Hệ Thống Hội Đồng Đa Agent (MAS)',
        subtasks: [
            { id: 'council', name: 'Hội Đồng 5 Diagnostic Agents + Rule Engine + Safety Advisor' }
        ],
        modelLabel: 'Cơ Chế Bỏ Phiếu Hội Đồng',
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
// RESET KẾT QUẢ CŨ (NGĂN CHẶN LỖI HIỂN THỊ CHÉO GIỮA CÁC CHỨC NĂNG)
// =============================================================================
function resetResults() {
    const resultContainer = document.getElementById('result-container');
    const chartContainer = document.getElementById('chart-container');
    const ragDetails = document.getElementById('rag-details');
    const agentDetails = document.getElementById('agent-details');
    const explanationBox = document.getElementById('explanation-box');

    // Ẩn tất cả các khối
    resultContainer.classList.add('hidden');
    chartContainer.classList.add('hidden');
    ragDetails.classList.add('hidden');
    agentDetails.classList.add('hidden');
    if (explanationBox) {
        explanationBox.classList.add('hidden');
        explanationBox.innerHTML = '';
    }

    // Xóa trắng toàn bộ nội dung HTML cũ bên trong
    ragDetails.innerHTML = '';
    agentDetails.innerHTML = '';

    // Hủy chart cũ nếu đang chạy
    if (predictionChart) {
        predictionChart.destroy();
        predictionChart = null;
    }
}

// =============================================================================
// CHUYỂN ĐỔI MÔ HÌNH VÀ CẬP NHẬT FORM
// =============================================================================
function setupModel(modelKey) {
    currentModel = modelKey;
    const cfg = modelConfigs[modelKey];
    if (!cfg) return;

    // Reset sạch kết quả cũ
    resetResults();

    // Cập nhật Subtask Dropdown
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

    cfg.models.forEach(m => {
        modelRadios.innerHTML += `
            <label class="radio-card">
                <input type="radio" name="model_choice" value="${m.id}" ${m.checked ? 'checked' : ''} required>
                <span class="card-content">${m.name}</span>
            </label>
        `;
    });

    // Render dynamic inputs
    renderInputsForCurrentState();
}

function renderInputsForCurrentState() {
    const dynamicInputs = document.getElementById('dynamic-inputs');
    dynamicInputs.innerHTML = '';

    const subtaskVal = document.getElementById('subtask-select').value;
    const cfg = modelConfigs[currentModel];
    if (!cfg) return;

    let inputsToRender = [];

    if (currentModel === 'cnn') {
        inputsToRender = (subtaskVal === 'conv1d') ? cfg.inputsConv1D : cfg.inputsConv2D;
        if (subtaskVal === 'conv2d') {
            dynamicInputs.innerHTML += `<div class="info-text">Mô phỏng phép nhân chập ma trận 3x3 Sobel trượt qua ảnh 5x5 để làm nổi bật đường biên theo phương giải tích.</div>`;
        }
    } else if (currentModel === 'rnn_lstm') {
        dynamicInputs.innerHTML += `<div class="info-text">${cfg.info}</div>`;
    } else if (currentModel === 'kg_rag') {
        inputsToRender = cfg.inputs;
        if (subtaskVal === 'gold') {
            inputsToRender[0].val = 'Vàng miếng SJC chịu sự quản lý theo nghị định nào và yếu tố nào chi phối giá vàng?';
        } else if (subtaskVal === 'ck') {
            inputsToRender[0].val = 'Quy định biên độ sàn HOSE là bao nhiêu và VN30 gồm những cổ phiếu nào?';
        } else {
            inputsToRender[0].val = 'Mức đường huyết lúc đói bao nhiêu thì được chẩn đoán mắc bệnh đái tháo đường?';
        }
    } else if (currentModel === 'multi_agent') {
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
    resetResults();
    renderInputsForCurrentState();
}

// Bắt sự kiện chuyển tab mô hình
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', function() {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        this.classList.add('active');
        setupModel(this.getAttribute('data-model'));
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
    const explanationBox = document.getElementById('explanation-box');

    // Reset sạch sẽ tất cả kết quả cũ trước khi thực hiện request mới
    resetResults();

    // Bật Loading
    submitBtn.classList.add('loading');
    submitBtn.disabled = true;

    const subtask = document.getElementById('subtask-select').value;
    const modelChecked = document.querySelector('input[name="model_choice"]:checked');
    const modelChoice = modelChecked ? modelChecked.value : '';

    // Thu thập payload
    const payload = {
        model_category: currentModel,
        chapter: currentModel, // tương thích backend
        subtask: subtask,
        model_name: modelChoice,
        model_type: modelChoice,
        architecture: modelChoice
    };

    // Lấy giá trị của tất cả inputs động hiện diện
    document.querySelectorAll('#dynamic-inputs input, #dynamic-inputs select').forEach(el => {
        payload[el.id] = el.value;
    });

    if (currentModel === 'cnn') {
        payload.cnn_mode = subtask;
        payload.kernel_type = modelChoice;
    } else if (currentModel === 'rnn_lstm') {
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
                // Đảm bảo reset lại trước khi bật khối tương ứng
                resetResults();

                resultValue.textContent = data.prediction;
                resultConfidence.textContent = data.confidence !== undefined ? data.confidence : '--';
                modelBadge.textContent = data.model_used || 'AI Model';

                // TRƯỜNG HỢP 1: Mạng Hồi Quy RNN / LSTM (Hiển thị biểu đồ chuỗi thời gian)
                if (data.type === 'chart' && data.chartData) {
                    chartContainer.classList.remove('hidden');
                    renderTimeSeriesChart(data.chartData, data.unit || 'Giá trị');
                }

                // TRƯỜNG HỢP 2: Knowledge Graph & RAG (Hiển thị căn cứ tri thức)
                else if (data.type === 'rag_qa') {
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

                // TRƯỜNG HỢP 3: Hệ Đa Agent - MAS (Hiển thị bảng bỏ phiếu 5 Agents & Thẩm định)
                else if (data.type === 'multi_agent') {
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

                // TRƯỜNG HỢP 4: Các giải thích phụ (ví dụ Conv1D hoặc Conv2D)
                if (data.explanation && data.type !== 'rag_qa' && data.type !== 'multi_agent') {
                    if (explanationBox) {
                        explanationBox.classList.remove('hidden');
                        explanationBox.innerHTML = `💡 ${data.explanation}`;
                    }
                }

                resultContainer.classList.remove('hidden');
            } else {
                alert('Lỗi: ' + data.error);
            }

            submitBtn.classList.remove('loading');
            submitBtn.disabled = false;
        }, 200);

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
                    borderColor: '#059669',
                    backgroundColor: 'rgba(5, 150, 105, 0.12)',
                    borderWidth: 2.4,
                    borderDash: [5, 4],
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
                    labels: { boxWidth: 12, font: { size: 11, weight: '600' } }
                },
                tooltip: {
                    callbacks: {
                        label: function(c) {
                            return `${c.dataset.label}: ${c.raw !== null ? c.raw.toLocaleString() : '--'} ${unitLabel}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { maxTicksLimit: 12, font: { size: 10 } }
                },
                y: {
                    grid: { color: '#f1f5f9' },
                    ticks: { font: { size: 10 } }
                }
            }
        }
    });
}

// Khởi tạo mặc định với tab Hệ Đa Agent
setupModel(currentModel);
