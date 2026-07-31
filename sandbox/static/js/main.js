// State Management for Chat History
let conversationHistory = [];

document.addEventListener('DOMContentLoaded', () => {
    // 1. Initial System Diagnostics
    checkSystemStatus();

    // 2. Tab Navigation Binding
    setupTabNavigation();

    // 3. Hyperparameter Sliders Binding
    setupSliders();

    // 4. Gemma Sandbox Setup
    setupGemmaSandbox();

    // 5. CNN Classifier Upload & Processing
    setupCNNClassifier();

    // 6. Redistribution Pipeline Setup
    setupRedistributionPipeline();

    // 7. Chatbot Setup
    setupChatbot();

    // 8. NGO Recommendation Setup
    setupNGORecommendation();

    // 9. Rebuild Index Ingest Setup
    setupDatabaseIngester();
});

/* ==============================================================================
   SYSTEM DIAGNOSTICS & STATUS
   ============================================================================== */
function checkSystemStatus() {
    const statusFlask = document.getElementById('status-flask-val');
    const statusApi = document.getElementById('status-api-val');
    const statusCnn = document.getElementById('status-cnn-val');
    const statusFaiss = document.getElementById('status-faiss-val');
    const systemDot = document.getElementById('system-status-dot');

    const apiBadge = document.getElementById('settings-api-badge');
    const cnnBadge = document.getElementById('settings-cnn-badge');
    const faissBadge = document.getElementById('settings-faiss-badge');

    const apiDesc = document.getElementById('settings-api-desc');
    const cnnDesc = document.getElementById('settings-cnn-desc');
    const faissDesc = document.getElementById('settings-faiss-desc');

    fetch('/api/status')
        .then(res => res.json())
        .then(data => {
            // Flask status
            statusFlask.textContent = "Online";
            statusFlask.classList.add('success');
            systemDot.classList.add('online');

            // API key status
            if (data.api_key.configured) {
                statusApi.textContent = data.api_key.model;
                apiBadge.textContent = "Configured";
                apiBadge.style.background = "rgba(16, 185, 129, 0.15)";
                apiBadge.style.color = "var(--success-color)";
                apiBadge.style.borderColor = "var(--success-color)";
                apiDesc.textContent = `Using default model: ${data.api_key.model}`;
            } else {
                statusApi.textContent = "Error / Missing";
                apiBadge.textContent = "Not Configured";
                apiBadge.style.background = "rgba(239, 68, 68, 0.15)";
                apiBadge.style.color = "var(--error-color)";
                apiBadge.style.borderColor = "var(--error-color)";
                apiDesc.textContent = `Error: ${data.api_key.error || "Invalid GEMINI_API_KEY value. Check `.env` file."}`;
                showWarningBanner("GEMINI_API_KEY is not configured or uses placeholder key! LLM operations will fail. Update your `.env` file.");
            }

            // CNN Model Checkpoint status
            if (data.cnn_model.loaded) {
                statusCnn.textContent = "Loaded";
                cnnBadge.textContent = "Loaded Successfully";
                cnnBadge.style.background = "rgba(16, 185, 129, 0.15)";
                cnnBadge.style.color = "var(--success-color)";
                cnnBadge.style.borderColor = "var(--success-color)";
                cnnDesc.textContent = `Weights active at: ${data.cnn_model.path}`;
            } else {
                statusCnn.textContent = "Missing";
                cnnBadge.textContent = "Not Found";
                cnnBadge.style.background = "rgba(239, 68, 68, 0.15)";
                cnnBadge.style.color = "var(--error-color)";
                cnnBadge.style.borderColor = "var(--error-color)";
                cnnDesc.textContent = `Error: ${data.cnn_model.error || "Weights not loaded. Check model checkpoint."}`;
                showWarningBanner("CNN model weights `best_model.pth` not found in root directory! Classifier will fail.");
            }

            // FAISS Index Database status
            if (data.faiss_index.loaded) {
                statusFaiss.textContent = "Ready";
                faissBadge.textContent = "Ready (FAISS Loaded)";
                faissBadge.style.background = "rgba(16, 185, 129, 0.15)";
                faissBadge.style.color = "var(--success-color)";
                faissBadge.style.borderColor = "var(--success-color)";
                faissDesc.textContent = `Vector DB index active at: ${data.faiss_index.path}`;
            } else {
                statusFaiss.textContent = "Missing Index";
                faissBadge.textContent = "No Index Found";
                faissBadge.style.background = "rgba(245, 158, 11, 0.15)";
                faissBadge.style.color = "var(--warning-color)";
                faissBadge.style.borderColor = "var(--warning-color)";
                faissDesc.textContent = `Error: ${data.faiss_index.error || "No vector index found."}`;
                showWarningBanner("FAISS Index files missing! RAG Chatbot will automatically trigger index building when you query it, or you can rebuild it in System Settings.");
            }

            // Ollama Engine Status
            const ollamaBadge = document.getElementById('settings-ollama-badge');
            const ollamaDesc = document.getElementById('settings-ollama-desc');
            const ollamaUrlInput = document.getElementById('settings-ollama-url');

            if (data.ollama) {
                if (ollamaUrlInput) ollamaUrlInput.value = data.ollama.host;
                
                if (data.ollama.online) {
                    ollamaBadge.textContent = "Online";
                    ollamaBadge.style.background = "rgba(16, 185, 129, 0.15)";
                    ollamaBadge.style.color = "var(--success-color)";
                    ollamaBadge.style.borderColor = "var(--success-color)";
                    ollamaDesc.textContent = `Ollama host active at ${data.ollama.host}. Models: ${data.ollama.models.join(', ') || 'None'}`;
                    window.ollamaModels = data.ollama.models;
                } else {
                    ollamaBadge.textContent = "Offline";
                    ollamaBadge.style.background = "rgba(245, 158, 11, 0.15)";
                    ollamaBadge.style.color = "var(--warning-color)";
                    ollamaBadge.style.borderColor = "var(--warning-color)";
                    ollamaDesc.textContent = `Offline or unreachable at ${data.ollama.host}. Run Ollama locally to bypass Cloud API keys.`;
                }
            }
        })
        .catch(err => {
            statusFlask.textContent = "Offline";
            systemDot.classList.add('offline');
            console.error("Health check error:", err);
            showErrorBanner("Could not establish connection to the local Flask server.");
        });
}

/* ==============================================================================
   NAVIGATION & UI EVENT HANDLERS
   ============================================================================== */
function setupTabNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    const tabPanels = document.querySelectorAll('.tab-panel');

    navItems.forEach(item => {
        item.querySelector('button').addEventListener('click', () => {
            // Remove active classes
            navItems.forEach(n => n.classList.remove('active'));
            tabPanels.forEach(p => p.classList.remove('active'));

            // Set active states
            item.classList.add('active');
            const tabId = item.getAttribute('data-tab');
            document.getElementById(tabId).classList.add('active');
        });
    });
}

function setupSliders() {
    const tempSlider = document.getElementById('gemma-temp-slider');
    const tempVal = document.getElementById('gemma-temp-val');
    tempSlider.addEventListener('input', () => {
        tempVal.textContent = tempSlider.value;
    });
}

// Banners for Alerts
function showWarningBanner(message) {
    const alertsContainer = document.getElementById('app-alerts-container');
    const alertId = `alert-${Date.now()}`;
    const html = `
        <div class="alert-banner warning" id="${alertId}">
            <i class="fa-solid fa-triangle-exclamation"></i>
            <span style="flex-grow: 1;">${message}</span>
            <i class="fa-solid fa-times" style="cursor: pointer;" onclick="document.getElementById('${alertId}').remove()"></i>
        </div>
    `;
    alertsContainer.insertAdjacentHTML('beforeend', html);
}

function showErrorBanner(message) {
    const alertsContainer = document.getElementById('app-alerts-container');
    const alertId = `alert-${Date.now()}`;
    const html = `
        <div class="alert-banner" id="${alertId}">
            <i class="fa-solid fa-circle-exclamation"></i>
            <span style="flex-grow: 1;">${message}</span>
            <i class="fa-solid fa-times" style="cursor: pointer;" onclick="document.getElementById('${alertId}').remove()"></i>
        </div>
    `;
    alertsContainer.insertAdjacentHTML('beforeend', html);
}

function showSuccessBanner(message) {
    const alertsContainer = document.getElementById('app-alerts-container');
    const alertId = `alert-${Date.now()}`;
    const html = `
        <div class="alert-banner info" id="${alertId}" style="background: rgba(16, 185, 129, 0.15); border-color: var(--success-color); color: #a7f3d0;">
            <i class="fa-solid fa-circle-check" style="color: var(--success-color);"></i>
            <span style="flex-grow: 1;">${message}</span>
            <i class="fa-solid fa-times" style="cursor: pointer;" onclick="document.getElementById('${alertId}').remove()"></i>
        </div>
    `;
    alertsContainer.insertAdjacentHTML('beforeend', html);
}

/* ==============================================================================
   TAB 1: GEMMA LLM SANDBOX
   ============================================================================== */
function setupGemmaSandbox() {
    const submitBtn = document.getElementById('gemma-submit-btn');
    const promptInput = document.getElementById('gemma-prompt-input');
    const modelSelect = document.getElementById('gemma-model-select');
    const tempSlider = document.getElementById('gemma-temp-slider');
    const tokensInput = document.getElementById('gemma-tokens-input');
    const sysInstructions = document.getElementById('gemma-sys-instructions');
    const outputBox = document.getElementById('gemma-output-box');
    const outputMeta = document.getElementById('gemma-output-meta');

    submitBtn.addEventListener('click', () => {
        const prompt = promptInput.value.trim();
        if (!prompt) {
            showWarningBanner("Gemma Sandbox: Input prompt cannot be empty.");
            return;
        }

        // 1. Prepare loader shimmer
        outputBox.classList.remove('empty');
        outputBox.innerHTML = getSkeletonLoader();
        outputMeta.textContent = "Processing generation...";
        submitBtn.disabled = true;

        const config = getSelectedModelAndBackend('gemma');
        const payload = {
            prompt: prompt,
            model_name: config.model,
            backend: config.backend,
            temperature: parseFloat(tempSlider.value),
            max_output_tokens: tokensInput.value ? parseInt(tokensInput.value) : null,
            system_instruction: sysInstructions.value.trim()
        };

        fetch('/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
            .then(res => res.json())
            .then(data => {
                if (data.error) {
                    outputBox.innerHTML = `<span style="color: var(--error-color);"><i class="fa-solid fa-bug"></i> Error: ${data.error}</span>`;
                    outputMeta.textContent = "Generation failed.";
                } else {
                    // Render parsed Markdown
                    outputBox.innerHTML = marked.parse(data.response);
                    outputMeta.textContent = `Model: ${data.model_used} | Backend: ${data.backend_used} | Speed: ${data.execution_time_sec}s`;
                }
            })
            .catch(err => {
                outputBox.innerHTML = `<span style="color: var(--error-color);"><i class="fa-solid fa-bug"></i> Failed to send request: ${err.message}</span>`;
                outputMeta.textContent = "Network failure.";
            })
            .finally(() => {
                submitBtn.disabled = false;
            });
    });
}

function loadGemmaPreset(type) {
    const promptInput = document.getElementById('gemma-prompt-input');
    const sysInstructions = document.getElementById('gemma-sys-instructions');
    
    if (type === 'summarize') {
        sysInstructions.value = "You are a professional logistics parser for Jivahar redistribution platform. Formulate clean summaries.";
        promptInput.value = "Create a structured database food log description for a donation containing:\n- Food Category: Kheer (Rice Pudding)\n- Quantity: 25 portions\n- Preparation Date: 4 hours ago\n- Current Storage Condition: Chilled (4°C)\nMake it concise and format as a bulleted redistribution log entry.";
    } else if (type === 'safety') {
        sysInstructions.value = "You are Food Safety Specialist chatbot. Deliver rules based on safety manual guidelines.";
        promptInput.value = "Provide concrete storage shelf-life limits, packaging instructions, and allergen warnings for a donation of Fresh Paneer Curry kept at room temperature for 3 hours. Warn if it violates safety rules.";
    } else if (type === 'creative') {
        sysInstructions.value = "You are an inspiring outreach writer for Jivahar food bank volunteering team.";
        promptInput.value = "Write a compelling, urgent social media notification message (under 280 characters) urging local volunteers to help pick up and redistribute 50 packs of freshly prepared Biryani from a nearby restaurant before the 4-hour shelf-life limit expires.";
    }
}

/* ==============================================================================
   TAB 2: CNN FOOD CLASSIFIER
   ============================================================================== */
function setupCNNClassifier() {
    const zone = document.getElementById('cnn-upload-zone');
    const fileInput = document.getElementById('cnn-file-input');
    const previewContainer = document.getElementById('cnn-preview-container');
    const imgPreview = document.getElementById('cnn-img-preview');
    const removeBtn = document.getElementById('cnn-preview-remove');
    const submitBtn = document.getElementById('cnn-submit-btn');

    const emptyBox = document.getElementById('cnn-status-empty');
    const resultBox = document.getElementById('cnn-result-box');
    const predLabel = document.getElementById('cnn-pred-label');
    const confidenceBar = document.getElementById('cnn-confidence-bar');
    const confidenceText = document.getElementById('cnn-confidence-text');
    const metaText = document.getElementById('cnn-output-meta');

    // Drag events
    zone.addEventListener('click', () => fileInput.click());
    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('dragover');
    });
    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleCNNImageFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) {
            handleCNNImageFile(fileInput.files[0]);
        }
    });

    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.value = "";
        previewContainer.style.display = "none";
        imgPreview.src = "";
        zone.style.display = "flex";
        submitBtn.disabled = true;
    });

    function handleCNNImageFile(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            imgPreview.src = e.target.result;
            zone.style.display = "none";
            previewContainer.style.display = "block";
            submitBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    submitBtn.addEventListener('click', () => {
        if (!fileInput.files.length) return;

        emptyBox.style.display = "none";
        resultBox.style.display = "none";
        metaText.textContent = "Analyzing image...";
        submitBtn.disabled = true;

        const formData = new FormData();
        formData.append('image', fileInput.files[0]);

        fetch('/api/classify', {
            method: 'POST',
            body: formData
        })
            .then(res => res.json())
            .then(data => {
                if (data.error) {
                    emptyBox.style.display = "block";
                    emptyBox.innerHTML = `<span style="color: var(--error-color);"><i class="fa-solid fa-bug"></i> Error: ${data.error}</span>`;
                } else {
                    resultBox.style.display = "block";
                    predLabel.textContent = data.food_name;
                    
                    const pct = (data.confidence * 100).toFixed(2);
                    confidenceBar.style.width = `${pct}%`;
                    confidenceText.textContent = `${pct}% Model Confidence Score`;
                    metaText.textContent = `CNN Framework: EfficientNet-B3 | Inference Speed: ${data.execution_time_sec}s`;
                }
            })
            .catch(err => {
                emptyBox.style.display = "block";
                emptyBox.innerHTML = `<span style="color: var(--error-color);"><i class="fa-solid fa-bug"></i> Request failed: ${err.message}</span>`;
            })
            .finally(() => {
                submitBtn.disabled = false;
            });
    });
}

/* ==============================================================================
   TAB 3: INTEGRATED REDISTRIBUTION PIPELINE
   ============================================================================== */
function setupRedistributionPipeline() {
    const zone = document.getElementById('pipe-upload-zone');
    const fileInput = document.getElementById('pipe-file-input');
    const previewContainer = document.getElementById('pipe-preview-container');
    const imgPreview = document.getElementById('pipe-img-preview');
    const removeBtn = document.getElementById('pipe-preview-remove');
    const submitBtn = document.getElementById('pipe-submit-btn');

    const qtyInput = document.getElementById('pipe-quantity');
    const prepInput = document.getElementById('pipe-prep-time');
    const storageSelect = document.getElementById('pipe-storage');
    const modelSelect = document.getElementById('pipe-model-select');

    const emptyBox = document.getElementById('pipe-status-empty');
    const resultsBox = document.getElementById('pipe-results-box');
    const predLabel = document.getElementById('pipe-pred-label');
    const confidenceBar = document.getElementById('pipe-confidence-bar');
    const confidenceText = document.getElementById('pipe-confidence-text');

    const summaryBox = document.getElementById('pipe-summary-box');
    const safetyBox = document.getElementById('pipe-safety-box');
    const citationsContainer = document.getElementById('pipe-citations');

    zone.addEventListener('click', () => fileInput.click());
    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('dragover');
    });
    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handlePipeImageFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) {
            handlePipeImageFile(fileInput.files[0]);
        }
    });

    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.value = "";
        previewContainer.style.display = "none";
        imgPreview.src = "";
        zone.style.display = "flex";
        submitBtn.disabled = true;
    });

    function handlePipeImageFile(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            imgPreview.src = e.target.result;
            zone.style.display = "none";
            previewContainer.style.display = "block";
            submitBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    submitBtn.addEventListener('click', () => {
        if (!fileInput.files.length) return;

        // Reset and show loader status
        emptyBox.style.display = "none";
        resultsBox.style.display = "none";
        
        const loaderHtml = getSkeletonLoader();
        summaryBox.innerHTML = loaderHtml;
        safetyBox.innerHTML = loaderHtml;
        citationsContainer.innerHTML = "";
        resultsBox.style.display = "block";
        submitBtn.disabled = true;

        const config = getSelectedModelAndBackend('pipe');
        const formData = new FormData();
        formData.append('image', fileInput.files[0]);
        formData.append('quantity', qtyInput.value.trim());
        formData.append('prepared_time', prepInput.value.trim());
        formData.append('storage_condition', storageSelect.value);
        formData.append('model_name', config.model);
        formData.append('backend', config.backend);

        fetch('/api/process-donation', {
            method: 'POST',
            body: formData
        })
            .then(res => res.json())
            .then(data => {
                if (data.error) {
                    resultsBox.style.display = "none";
                    emptyBox.style.display = "block";
                    emptyBox.innerHTML = `<span style="color: var(--error-color);"><i class="fa-solid fa-bug"></i> Pipeline Error: ${data.error}</span>`;
                } else {
                    // Update CNN class predicted
                    predLabel.textContent = data.food_name;
                    const pct = (data.cnn_confidence * 100).toFixed(2);
                    confidenceBar.style.width = `${pct}%`;
                    confidenceText.textContent = `${pct}% CNN Model Confidence (Class: ${data.food_name})`;

                    // Render summary
                    summaryBox.innerHTML = marked.parse(data.summary);

                    // Render safety RAG assessment
                    safetyBox.innerHTML = marked.parse(data.safety_advice);

                    // Render citations
                    if (data.safety_sources && data.safety_sources.length > 0) {
                        citationsContainer.innerHTML = getCitationsHTML(data.safety_sources);
                    } else {
                        citationsContainer.innerHTML = `<div style="padding: 1rem; color: var(--text-muted); font-style: italic;">No specific PDF rules citations were returned.</div>`;
                    }
                }
            })
            .catch(err => {
                resultsBox.style.display = "none";
                emptyBox.style.display = "block";
                emptyBox.innerHTML = `<span style="color: var(--error-color);"><i class="fa-solid fa-bug"></i> Connection failure: ${err.message}</span>`;
            })
            .finally(() => {
                submitBtn.disabled = false;
            });
    });
}

/* ==============================================================================
   TAB 4: JIVAHAR RAG CHATBOT
   ============================================================================== */
function setupChatbot() {
    const chatBox = document.getElementById('chat-box');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('chat-send-btn');
    const modelSelect = document.getElementById('chat-model-select');

    sendBtn.addEventListener('click', sendChatMessage);
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendChatMessage();
        }
    });

    function sendChatMessage() {
        const message = chatInput.value.trim();
        if (!message) return;

        // Clear input field
        chatInput.value = "";

        // Render user message bubble
        appendChatBubble('user', message);
        scrollChatToBottom();

        // Show typing indicator bubble
        const typingBubbleId = appendChatBubble('bot', `<div class="skeleton-line body-1" style="margin: 0; width: 80px;"></div>`);
        scrollChatToBottom();

        const config = getSelectedModelAndBackend('chat');
        const payload = {
            message: message,
            history: conversationHistory,
            model_name: config.model,
            backend: config.backend
        };

        fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
            .then(res => res.json())
            .then(data => {
                // Remove typing bubble
                document.getElementById(typingBubbleId).remove();

                if (data.error) {
                    appendChatBubble('bot', `<span style="color: var(--error-color);"><i class="fa-solid fa-triangle-exclamation"></i> Chatbot encountered an error: ${data.error}</span>`);
                } else {
                    const parsedResponse = marked.parse(data.response);
                    
                    // Format response bubble with collapsible citations
                    let botMessageHtml = parsedResponse;
                    if (data.sources && data.sources.length > 0) {
                        botMessageHtml += `
                            <div class="citation-container">
                                <div class="citation-header" onclick="toggleChatCitation(this)">
                                    <span><i class="fa-solid fa-file-shield"></i> Verified Citations (${data.sources.length})</span>
                                    <i class="fa-solid fa-chevron-down"></i>
                                </div>
                                <div class="citation-body">
                                    ${data.sources.map(src => `
                                        <div class="citation-item">
                                            <div class="citation-title">${src.source} (Page ${src.page})</div>
                                            <div class="citation-text">"${src.text}"</div>
                                        </div>
                                    `).join('')}
                                </div>
                            </div>
                        `;
                    }
                    
                    appendChatBubble('bot', botMessageHtml);

                    // Add to conversation history state
                    conversationHistory.push({ role: 'user', content: message });
                    conversationHistory.push({ role: 'model', content: data.response });
                }
                scrollChatToBottom();
            })
            .catch(err => {
                document.getElementById(typingBubbleId).remove();
                appendChatBubble('bot', `<span style="color: var(--error-color);"><i class="fa-solid fa-triangle-exclamation"></i> Network request failed: ${err.message}</span>`);
                scrollChatToBottom();
            });
    }

    function appendChatBubble(sender, content) {
        const bubbleId = `bubble-${Date.now()}`;
        const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const bubbleHtml = `
            <div class="chat-message ${sender}" id="${bubbleId}">
                <div class="message-bubble">
                    ${content}
                </div>
                <span class="message-time">${sender === 'user' ? 'You' : 'Gemma Bot'} | ${time}</span>
            </div>
        `;
        chatBox.insertAdjacentHTML('beforeend', bubbleHtml);
        return bubbleId;
    }

    function scrollChatToBottom() {
        chatBox.scrollTop = chatBox.scrollHeight;
    }
}

// Collapsible helper for chatbot citations
window.toggleChatCitation = function(element) {
    const body = element.nextElementSibling;
    const arrow = element.querySelector('i:last-child');
    element.classList.toggle('open');
    if (element.classList.contains('open')) {
        body.style.display = 'flex';
        arrow.className = 'fa-solid fa-chevron-up';
    } else {
        body.style.display = 'none';
        arrow.className = 'fa-solid fa-chevron-down';
    }
};

function loadChatPreset(type) {
    const chatInput = document.getElementById('chat-input');
    if (type === 'shelf-life') {
        chatInput.value = "What are the rules regarding the shelf-life constraints of cooked food donations?";
    } else if (type === 'ngo') {
        chatInput.value = "How does Jivahar matching engine match local NGOs with fresh food donations?";
    } else if (type === 'safety') {
        chatInput.value = "What temperature standards must be maintained during transportation for hot-held food items?";
    }
    chatInput.focus();
}

/* ==============================================================================
   TAB 5: SYSTEM SETTINGS & FAISS INGESTER
   ============================================================================= */
function setupDatabaseIngester() {
    const logConsole = document.getElementById('settings-ingest-log');
    const ingestBtn = document.getElementById('settings-ingest-btn');

    ingestBtn.addEventListener('click', () => {
        logConsole.classList.remove('empty');
        logConsole.innerHTML = `<span style="color: var(--secondary-color);">[SYSTEM CONSOLE] Running Knowledge Base Ingestion pipeline...</span>\nScanning directory: data/knowledge_base/...\nExtracting text from PDF chapters...\nVectorizing paragraphs...\nRebuilding FAISS flat L2 index structure...\n`;
        ingestBtn.disabled = true;

        fetch('/api/ingest', {
            method: 'POST'
        })
            .then(res => res.json())
            .then(data => {
                if (data.error) {
                    logConsole.innerHTML += `\n<span style="color: var(--error-color);">[FAILURE] Ingestion failed: ${data.error}</span>`;
                    showErrorBanner("Knowledge base ingestion pipeline failed.");
                } else {
                    logConsole.innerHTML += `\n<span style="color: var(--success-color);">[SUCCESS] Index rebuilt successfully! Ingested chunks.</span>\nRebuild speed: ${data.execution_time_sec}s\nSaving index.faiss to config path...`;
                    showSuccessBanner("FAISS Index database successfully built and loaded! Jivahar Chatbot is ready.");
                    
                    // Refresh status
                    checkSystemStatus();
                }
            })
            .catch(err => {
                logConsole.innerHTML += `\n<span style="color: var(--error-color);">[NETWORK FAILURE] Pipeline request failed: ${err.message}</span>`;
            })
            .finally(() => {
                ingestBtn.disabled = false;
            });
    });

    // Binding save API configurations button
    const saveBtn = document.getElementById('settings-save-btn');
    if (saveBtn) {
        saveBtn.addEventListener('click', () => {
            const ollamaHost = document.getElementById('settings-ollama-url').value.trim();
            const hfToken = document.getElementById('settings-hf-token').value.trim();
            
            saveBtn.disabled = true;
            fetch('/api/save-settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    ollama_host: ollamaHost,
                    hf_token: hfToken
                })
            })
                .then(res => res.json())
                .then(data => {
                    if (data.success) {
                        showSuccessBanner("Multi-backend credentials saved successfully on the server!");
                        checkSystemStatus();
                    } else {
                        showErrorBanner("Failed to save credentials.");
                    }
                })
                .catch(err => {
                    showErrorBanner("Network error saving configuration: " + err.message);
                })
                .finally(() => {
                    saveBtn.disabled = false;
                });
        });
    }
}

/* ==============================================================================
   MULTI-BACKEND ROUTING UTILITIES
   ============================================================================== */
window.onBackendChange = function(module) {
    const backendSelect = document.getElementById(`${module}-backend-select`);
    const modelSelect = document.getElementById(`${module}-model-select`);
    const customGroup = document.getElementById(`${module}-custom-model-group`);
    
    const backend = backendSelect.value;
    modelSelect.innerHTML = "";
    
    if (backend === 'gemini') {
        modelSelect.innerHTML = `
            <option value="gemini-flash-latest" selected>Gemini Flash (Latest - Default)</option>
            <option value="gemini-2.0-flash">Gemini 2.0 Flash (Latest)</option>
            <option value="gemma-4-26b-a4b-it">Gemma 4 26B (Instruct - AI Studio)</option>
            <option value="gemma-4-31b-it">Gemma 4 31B (Instruct - AI Studio)</option>
        `;
        customGroup.style.display = 'none';
    } else if (backend === 'ollama') {
        const models = window.ollamaModels || [];
        if (models.length > 0) {
            models.forEach((model, i) => {
                modelSelect.innerHTML += `<option value="${model}" ${i === 0 ? 'selected' : ''}>${model}</option>`;
            });
        } else {
            modelSelect.innerHTML += `
                <option value="gemma2:9b" selected>gemma2:9b (Ollama Default)</option>
                <option value="gemma2:2b">gemma2:2b</option>
                <option value="gemma:latest">gemma:latest</option>
            `;
        }
        modelSelect.innerHTML += `<option value="custom">Custom Model Name...</option>`;
        
        if (modelSelect.value === 'custom') {
            customGroup.style.display = 'block';
        } else {
            customGroup.style.display = 'none';
        }
    } else if (backend === 'huggingface') {
        modelSelect.innerHTML = `
            <option value="google/gemma-2-9b-it" selected>google/gemma-2-9b-it (Instruct)</option>
            <option value="google/gemma-2-2b-it">google/gemma-2-2b-it</option>
            <option value="custom">Custom Model Name...</option>
        `;
        customGroup.style.display = 'none';
    }
};

window.onModelChange = function(module) {
    const modelSelect = document.getElementById(`${module}-model-select`);
    const customGroup = document.getElementById(`${module}-custom-model-group`);
    
    if (modelSelect.value === 'custom') {
        customGroup.style.display = 'block';
    } else {
        customGroup.style.display = 'none';
    }
};

function getSelectedModelAndBackend(module) {
    const backend = document.getElementById(`${module}-backend-select`).value;
    const modelSelect = document.getElementById(`${module}-model-select`);
    let model = modelSelect.value;
    
    if (model === 'custom') {
        model = document.getElementById(`${module}-custom-model-input`).value.trim();
    }
    return { backend, model };
}

/* ==============================================================================
   UI HELPER RENDERERS
   ============================================================================== */
function getSkeletonLoader() {
    return `
        <div class="skeleton">
            <div class="skeleton-line title"></div>
            <div class="skeleton-line body-1"></div>
            <div class="skeleton-line body-2"></div>
            <div class="skeleton-line body-3"></div>
            <div class="skeleton-line body-4"></div>
            <div class="skeleton-line body-5"></div>
        </div>
    `;
}

function getCitationsHTML(sources) {
    let html = `
        <div class="citation-header" onclick="toggleChatCitation(this)">
            <span><i class="fa-solid fa-book-bookmark"></i> Grounded Policy Citations (${sources.length})</span>
            <i class="fa-solid fa-chevron-down"></i>
        </div>
        <div class="citation-body">
    `;
    sources.forEach(src => {
        html += `
            <div class="citation-item">
                <div class="citation-title">${src.source} (Page ${src.page})</div>
                <div class="citation-text">"${src.text}"</div>
            </div>
        `;
    });
    html += `</div>`;
    return html;
}

/* ==============================================================================
   TAB 4: NGO MATCH ADVISOR SECTION
   ============================================================================== */
function setupNGORecommendation() {
    const submitBtn = document.getElementById('recommend-submit-btn');
    const ngoSelect = document.getElementById('recommend-ngo-select');
    const distInput = document.getElementById('recommend-distance');
    const capInput = document.getElementById('recommend-capacity');
    const ratInput = document.getElementById('recommend-rating');
    const foodInput = document.getElementById('recommend-food-details');
    
    const outputBox = document.getElementById('recommend-output-box');
    const outputMeta = document.getElementById('recommend-output-meta');
    const citationsContainer = document.getElementById('recommend-citations');
    
    if (!submitBtn) return;
    
    submitBtn.addEventListener('click', () => {
        const ngoName = ngoSelect.value;
        const distance = distInput.value.trim();
        const capacity = capInput.value.trim();
        const rating = ratInput.value.trim();
        const foodDetails = foodInput.value.trim();
        
        if (!distance || !capacity || !rating || !foodDetails) {
            showWarningBanner("NGO Recommendation: Please fill out all match parameters.");
            return;
        }
        
        // Show loading skeletons
        outputBox.classList.remove('empty');
        outputBox.innerHTML = getSkeletonLoader();
        outputMeta.textContent = "Matching and explaining...";
        citationsContainer.style.display = "none";
        citationsContainer.innerHTML = "";
        submitBtn.disabled = true;
        
        const config = getSelectedModelAndBackend('recommend');
        const payload = {
            ngo_name: ngoName,
            distance_km: parseFloat(distance),
            capacity_kg: parseFloat(capacity),
            rating: parseFloat(rating),
            food_details: foodDetails,
            model_name: config.model,
            backend: config.backend
        };
        
        fetch('/api/recommend-ngo', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
            .then(res => res.json())
            .then(data => {
                if (data.error) {
                    outputBox.innerHTML = `<span style="color: var(--error-color);"><i class="fa-solid fa-bug"></i> Error: ${data.error}</span>`;
                    outputMeta.textContent = "Match generation failed.";
                } else {
                    outputBox.innerHTML = marked.parse(data.explanation);
                    outputMeta.textContent = `Model: ${data.model_used} | Backend: ${data.backend_used} | Speed: ${data.execution_time_sec}s`;
                    
                    // Render RAG citations
                    if (data.sources && data.sources.length > 0) {
                        citationsContainer.innerHTML = getCitationsHTML(data.sources);
                        citationsContainer.style.display = "block";
                    } else {
                        citationsContainer.style.display = "none";
                    }
                }
            })
            .catch(err => {
                outputBox.innerHTML = `<span style="color: var(--error-color);"><i class="fa-solid fa-bug"></i> Connection Error: ${err.message}</span>`;
                outputMeta.textContent = "Network failure.";
            })
            .finally(() => {
                submitBtn.disabled = false;
            });
    });
}
