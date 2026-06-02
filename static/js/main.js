/* ----------------------------------------------------
   GreenCycle JavaScript Controller - Frontend Logic
   ---------------------------------------------------- */

document.addEventListener('DOMContentLoaded', () => {
    // ------------------------------------------------
    // 1. TAB NAVIGATION SYSTEM
    // ------------------------------------------------
    const navButtons = document.querySelectorAll('.nav-btn');
    const tabPanels = document.querySelectorAll('.tab-panel');

    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');
            
            // Toggle active button
            navButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Toggle active panel
            tabPanels.forEach(panel => {
                panel.classList.remove('active');
                if (panel.id === targetTab) {
                    // Slight delay for smooth transition effect
                    setTimeout(() => {
                        panel.classList.add('active');
                    }, 50);
                }
            });
        });
    });

    // ------------------------------------------------
    // 2. IMAGE UPLOAD & CLASSIFICATION INTERACTION
    // ------------------------------------------------
    const dropZone = document.getElementById('drop-zone');
    const imageInput = document.getElementById('image-input');
    const dropZoneContent = document.getElementById('drop-zone-content');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const btnClear = document.getElementById('btn-clear');
    const btnClassify = document.getElementById('btn-classify');
    const classifySpinner = document.getElementById('classify-spinner');
    const btnClassifyText = document.getElementById('btn-classify-text');

    const resultPlaceholder = document.getElementById('result-placeholder');
    const resultContent = document.getElementById('result-content');
    const resultBadge = document.getElementById('result-badge');
    const resultScore = document.getElementById('result-score');
    const resultMethod = document.getElementById('result-method');
    
    // Explainable AI (XAI) Elements
    const xaiSection = document.getElementById('xai-section');
    const xaiImageView = document.getElementById('xai-image-view');
    const xaiImageTag = document.getElementById('xai-image-tag');
    let btnXaiOrig = document.getElementById('btn-xai-orig');
    let btnXaiCam = document.getElementById('btn-xai-cam');
    
    // Eco Recomendation Elements
    const ecoTitle = document.getElementById('eco-title');
    const ecoDecomposition = document.getElementById('eco-decomposition');
    const ecoTip = document.getElementById('eco-tip');
    const ecoFact = document.getElementById('eco-fact');
    const predictionsList = document.getElementById('predictions-list');

    let selectedFile = null;
    let originalImageDataURL = null; // Store original image base64 data to clear state between inferences

    // Model Selector Elements & State
    const modelCards = document.querySelectorAll('.model-card');
    let activeModel = 'mobilenet'; // Default active model

    // Hook up Model Selector click handlers
    modelCards.forEach(card => {
        card.addEventListener('click', () => {
            modelCards.forEach(c => c.classList.remove('active'));
            card.classList.add('active');
            activeModel = card.getAttribute('data-model');
            console.log(`>>> Active model selected: ${activeModel}`);
            
            // Immediate prediction UI state clearing when changing models
            resetClassifierResults();
            if (originalImageDataURL) {
                imagePreview.src = originalImageDataURL;
            }
        });
    });

    // Trigger input click on drop zone click
    dropZone.addEventListener('click', (e) => {
        // Prevent clicking subelements from multiple triggers
        if (e.target !== btnClear && !btnClear.contains(e.target)) {
            imageInput.click();
        }
    });

    // Handle drag events
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
        }, false);
    });

    // Handle drop
    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    // Handle file dialog selection
    imageInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    function handleFileSelect(file) {
        if (!file.type.startsWith('image/')) {
            alert('Vui lòng chọn một file hình ảnh hợp lệ.');
            return;
        }

        // Flush previous UI prediction state immediately when selecting a new file
        resetClassifierResults();

        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            originalImageDataURL = e.target.result;
            imagePreview.src = originalImageDataURL;
            dropZoneContent.style.display = 'none';
            previewContainer.style.display = 'flex';
            btnClassify.removeAttribute('disabled');
        };
        reader.readAsDataURL(file);
    }

    // Clear Selected Image
    btnClear.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        selectedFile = null;
        originalImageDataURL = null;
        imageInput.value = '';
        imagePreview.src = '#';
        previewContainer.style.display = 'none';
        dropZoneContent.style.display = 'block';
        btnClassify.setAttribute('disabled', 'true');
        resetClassifierResults();
    });

    function resetClassifierResults() {
        resultContent.style.display = 'none';
        resultPlaceholder.style.display = 'flex';
        if (xaiSection) {
            xaiSection.style.display = 'none';
        }
        // Completely flush old results text/tags to avoid UI overlap or stale text
        resultBadge.textContent = '';
        resultScore.textContent = '';
        resultMethod.innerHTML = '';
        ecoTitle.textContent = '';
        ecoDecomposition.textContent = '';
        ecoTip.textContent = '';
        ecoFact.textContent = '';
        predictionsList.innerHTML = '';
    }

    // Call API to Classify Image
    btnClassify.addEventListener('click', async () => {
        if (!selectedFile) return;

        // Flush previous prediction results from the UI before running the next inference state
        resetClassifierResults();

        // Restore clean original image to the preview panel while prediction is running
        if (originalImageDataURL) {
            imagePreview.src = originalImageDataURL;
        }

        // Set Loading State
        btnClassify.setAttribute('disabled', 'true');
        btnClassifyText.style.display = 'none';
        classifySpinner.style.display = 'block';
        
        const formData = new FormData();
        formData.append('image', selectedFile);
        formData.append('model_type', activeModel);

        try {
            const response = await fetch('/classify', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            
            if (response.ok) {
                renderClassificationResults(data);
            } else {
                alert(`Lỗi phân loại: ${data.error || 'Có lỗi xảy ra'}`);
                resetClassifierResults();
            }
        } catch (error) {
            console.error('Classification error:', error);
            alert('Kết nối máy chủ thất bại. Vui lòng kiểm tra Flask server.');
            resetClassifierResults();
        } finally {
            // Restore Button State
            btnClassify.removeAttribute('disabled');
            btnClassifyText.style.display = 'block';
            classifySpinner.style.display = 'none';
        }
    });

    function renderClassificationResults(res) {
        resultPlaceholder.style.display = 'none';
        resultContent.style.display = 'block';

        // Update preview image to the server-returned path
        // This will display the YOLO bounding boxes in the main preview container if object detection is used
        if (res.filepath) {
            imagePreview.src = res.filepath + '?t=' + new Date().getTime();
        }

        // Set top result
        resultBadge.textContent = res.class;
        resultScore.textContent = `${res.confidence.toFixed(1)}%`;
        resultMethod.innerHTML = `<i class="fa-solid fa-microchip"></i> Thuật toán: ${res.method}`;
        
        // Dynamic styling for category
        const ecoInfo = res.info;
        resultBadge.style.color = ecoInfo.color;
        resultBadge.style.textShadow = `0 0 20px ${ecoInfo.color}33`;
        
        // Explainable AI (XAI) setup
        if (res.gradcam_filepath) {
            xaiSection.style.display = 'block';
            xaiImageView.src = res.filepath;
            xaiImageTag.textContent = 'Ảnh Gốc';
            xaiImageTag.className = 'xai-image-tag';
            
            // Reset button classes
            btnXaiOrig.className = 'btn-xai active';
            btnXaiCam.className = 'btn-xai';
            
            // Re-clone buttons to clear previous event listeners cleanly
            const newBtnOrig = btnXaiOrig.cloneNode(true);
            const newBtnCam = btnXaiCam.cloneNode(true);
            btnXaiOrig.parentNode.replaceChild(newBtnOrig, btnXaiOrig);
            btnXaiCam.parentNode.replaceChild(newBtnCam, btnXaiCam);
            
            // Re-assign references
            btnXaiOrig = newBtnOrig;
            btnXaiCam = newBtnCam;
            
            btnXaiOrig.addEventListener('click', () => {
                btnXaiOrig.classList.add('active');
                btnXaiCam.classList.remove('active');
                xaiImageView.src = res.filepath;
                xaiImageTag.textContent = 'Ảnh Gốc';
                xaiImageTag.className = 'xai-image-tag';
            });
            
            btnXaiCam.addEventListener('click', () => {
                btnXaiCam.classList.add('active');
                btnXaiOrig.classList.remove('active');
                xaiImageView.src = res.gradcam_filepath;
                xaiImageTag.textContent = 'Vùng Chú Ý AI';
                xaiImageTag.className = 'xai-image-tag xai-tag-cam';
            });
        } else {
            xaiSection.style.display = 'none';
        }
        
        // Eco Recommendations
        ecoTitle.textContent = ecoInfo.title;
        ecoTitle.style.color = ecoInfo.color;
        ecoDecomposition.textContent = ecoInfo.decomposition;
        ecoTip.textContent = ecoInfo.tip;
        ecoFact.textContent = ecoInfo.fact;

        // Probability bars
        predictionsList.innerHTML = '';
        res.all_predictions.forEach(pred => {
            const barItem = document.createElement('div');
            barItem.className = 'prediction-bar-item';
            barItem.innerHTML = `
                <div class="pred-bar-label">
                    <span>${pred.class.toUpperCase()}</span>
                    <span style="font-weight: 700; color: ${pred.info.color}">${pred.confidence.toFixed(1)}%</span>
                </div>
                <div class="pred-bar-bg">
                    <div class="pred-bar-fill" style="width: 0%; background: ${pred.info.color}"></div>
                </div>
            `;
            predictionsList.appendChild(barItem);
            
            // Animation for bar expansion
            setTimeout(() => {
                barItem.querySelector('.pred-bar-fill').style.width = `${pred.confidence}%`;
            }, 100);
        });
    }

    // ------------------------------------------------
    // 3. LIVE MODEL TRAINING BOARD LOGIC
    // ------------------------------------------------
    const btnStartTraining = document.getElementById('btn-start-training');
    const trainSpinner = document.getElementById('train-spinner');
    const btnTrainText = document.getElementById('btn-train-text');
    const epochsSelect = document.getElementById('epochs-input');
    
    const trainingMetricsPanel = document.getElementById('training-metrics-panel');
    const progressBarFill = document.getElementById('progress-bar-fill');
    const progressPercent = document.getElementById('progress-percent');
    const metricEpoch = document.getElementById('metric-epoch');
    const metricAcc = document.getElementById('metric-acc');
    const metricLoss = document.getElementById('metric-loss');
    const metricValAcc = document.getElementById('metric-val-acc');
    
    const terminalBody = document.getElementById('terminal-body');
    const consolePulse = document.getElementById('console-pulse');

    let trainingStatusInterval = null;
    let trainingChart = null;

    // Initialize Chart.js
    function initTrainingChart() {
        const ctx = document.getElementById('training-chart').getContext('2d');
        if (trainingChart) {
            trainingChart.destroy();
        }
        
        trainingChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [], // Epoch numbers
                datasets: [
                    {
                        label: 'Training Accuracy',
                        data: [],
                        borderColor: '#10b981', // Emerald
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        tension: 0.3,
                        fill: true
                    },
                    {
                        label: 'Training Loss',
                        data: [],
                        borderColor: '#ef4444', // Red
                        backgroundColor: 'rgba(239, 68, 68, 0.05)',
                        tension: 0.3,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: '#94a3b8', font: { family: 'Outfit' } }
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.03)' },
                        ticks: { color: '#94a3b8', font: { family: 'Inter' } }
                    },
                    y: {
                        grid: { color: 'rgba(255, 255, 255, 0.03)' },
                        ticks: { color: '#94a3b8', font: { family: 'Inter' } },
                        min: 0,
                        max: 1.2
                    }
                }
            }
        });
    }

    if (btnStartTraining) {
        btnStartTraining.addEventListener('click', async () => {
            // Confirm triggers
            btnStartTraining.setAttribute('disabled', 'true');
            btnTrainText.style.display = 'none';
            trainSpinner.style.display = 'block';
            consolePulse.style.display = 'block';
            
            initTrainingChart();
            trainingMetricsPanel.style.display = 'block';
            terminalBody.value = "Gửi lệnh khởi chạy huấn luyện học sâu lên Backend...\n";
            
            try {
                const response = await fetch('/train', {
                    method: 'POST'
                });
                const data = await response.json();
                
                if (response.ok) {
                    terminalBody.value += "Khởi tạo thành công. Bắt đầu luồng kiểm soát ngầm...\n";
                    startTrackingTraining();
                } else {
                    alert(`Lỗi khởi tạo huấn luyện: ${data.error || 'Có lỗi xảy ra'}`);
                    resetTrainingButton();
                }
            } catch (error) {
                console.error('Training trigger error:', error);
                alert('Không thể kết nối máy chủ để chạy huấn luyện.');
                resetTrainingButton();
            }
        });
    }

    function resetTrainingButton() {
        if (btnStartTraining) {
            btnStartTraining.removeAttribute('disabled');
            btnTrainText.style.display = 'block';
            trainSpinner.style.display = 'none';
        }
        consolePulse.style.display = 'none';
    }

    function startTrackingTraining() {
        if (trainingStatusInterval) {
            clearInterval(trainingStatusInterval);
        }
        
        trainingStatusInterval = setInterval(async () => {
            try {
                const response = await fetch('/train_status');
                const data = await response.json();
                
                updateTrainingUI(data);
                
                if (data.status === 'completed' || data.status === 'failed') {
                    clearInterval(trainingStatusInterval);
                    resetTrainingButton();
                    
                    if (data.status === 'completed') {
                        alert('Chúc mừng! Mô hình tùy chỉnh của bạn đã huấn luyện và tích hợp thành công vào bộ nhận diện rác!');
                        // Reload index parameters to show trained model is active
                        setTimeout(() => location.reload(), 1500);
                    } else {
                        alert(`Huấn luyện thất bại: ${data.error}`);
                    }
                }
            } catch (error) {
                console.error('Error tracking training status:', error);
            }
        }, 1000);
    }

    function updateTrainingUI(status) {
        // Update progress bar
        progressBarFill.style.width = `${status.progress}%`;
        progressPercent.textContent = `${status.progress}%`;
        
        // Update metrics
        if (status.epoch > 0) {
            metricEpoch.textContent = `${status.epoch}/${status.total_epochs}`;
            metricAcc.textContent = `${(status.accuracy * 100).toFixed(1)}%`;
            metricLoss.textContent = status.loss.toFixed(4);
            metricValAcc.textContent = status.val_accuracy > 0 ? `${(status.val_accuracy * 100).toFixed(1)}%` : 'Đang đo...';
        }
        
        // Update Terminal console and scroll to bottom
        if (status.log) {
            terminalBody.value = status.log;
            terminalBody.scrollTop = terminalBody.scrollHeight;
        }

        // Dynamically add data point to Chart if a new epoch completed
        if (status.epoch > 0 && trainingChart) {
            const chartLabels = trainingChart.data.labels;
            const currentPointsCount = chartLabels.length;
            
            // Check if we need to append the new epoch stats
            if (status.epoch > currentPointsCount) {
                // Add new labels & values
                trainingChart.data.labels.push(`Epoch ${status.epoch}`);
                trainingChart.data.datasets[0].data.push(status.accuracy);
                trainingChart.data.datasets[1].data.push(status.loss);
                
                // Adjust dynamic Y scale max if loss is high
                if (status.loss > 1.2) {
                    trainingChart.options.scales.y.max = Math.ceil(status.loss * 10) / 10;
                }
                
                trainingChart.update();
            }
        }
    }
});
