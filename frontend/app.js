// -------------------------------------------------------------
// LOCAL RAG SANDBOX - FRONTEND CONTROLLER (VANILLA JAVASCRIPT)
// -------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
    // STATE MANAGEMENT
    const state = {
        geminiKey: localStorage.getItem('gemini_api_key') || '',
        currentFile: null,
        currentChunks: [],
        totalChunksIndexed: 0,
        indexingReady: false,
        activeTab: 'compare-tab',
        retrievalMode: 'hybrid',
        llmProvider: 'simulation',
        geminiReady: false
    };

    // DOM ELEMENTS
    const geminiKeyInput = document.getElementById('gemini-key-input');
    const saveKeyBtn = document.getElementById('save-key-btn');
    const statChunks = document.getElementById('stat-chunks');
    const statGemini = document.getElementById('stat-gemini');
    
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const fileStatusInfo = document.getElementById('file-status-info');
    const uploadedFileName = document.getElementById('uploaded-file-name');
    const uploadedFileSize = document.getElementById('uploaded-file-size');
    const uploadProgress = document.getElementById('upload-progress');
    
    const chunkSizeSlider = document.getElementById('chunk-size-slider');
    const chunkSizeVal = document.getElementById('chunk-size-val');
    const chunkOverlapSlider = document.getElementById('chunk-overlap-slider');
    const chunkOverlapVal = document.getElementById('chunk-overlap-val');
    const chunkStrategySelect = document.getElementById('chunk-strategy');
    
    const chunkPreviewBtn = document.getElementById('chunk-preview-btn');
    const indexDbBtn = document.getElementById('index-db-btn');
    const previewPanel = document.getElementById('preview-panel');
    const previewCount = document.getElementById('preview-count');
    const chunkVisualizer = document.getElementById('chunk-visualizer');
    
    const retrievalModeSelect = document.getElementById('retrieval-mode');
    const llmProviderSelect = document.getElementById('llm-provider');
    const ollamaModelGroup = document.getElementById('ollama-model-group');
    const ollamaModelInput = document.getElementById('ollama-model-input');
    const clearDbBtn = document.getElementById('clear-db-btn');
    
    const chatThread = document.getElementById('chat-thread');
    const queryInput = document.getElementById('query-input');
    const sendBtn = document.getElementById('send-btn');
    
    const tracePanel = document.getElementById('trace-panel');
    const traceToggle = document.getElementById('trace-toggle');
    const traceDetails = document.getElementById('trace-details');
    
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');
    
    // INITIALIZATION
    initApp();

    function initApp() {
        // Hydrate key input from storage
        if (state.geminiKey) {
            geminiKeyInput.value = state.geminiKey;
            saveConfigKey(state.geminiKey, false); // Sync with backend silently
        }
        
        // Sync API status
        fetchStatus();
        
        // Setup Event Listeners
        setupEventListeners();
    }

    // EVENT LISTENERS
    function setupEventListeners() {
        // Tab switching
        tabButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                const targetTab = btn.getAttribute('data-tab');
                switchTab(targetTab);
            });
        });

        // Config Save Key
        saveKeyBtn.addEventListener('click', () => {
            const key = geminiKeyInput.value.trim();
            saveConfigKey(key, true);
        });

        // Sliders Realtime Values
        chunkSizeSlider.addEventListener('input', (e) => {
            chunkSizeVal.textContent = e.target.value;
            // Cap overlap if it exceeds size
            const maxOverlap = Math.floor(parseInt(e.target.value) / 2);
            if (parseInt(chunkOverlapSlider.value) > maxOverlap) {
                chunkOverlapSlider.value = maxOverlap;
                chunkOverlapVal.textContent = maxOverlap;
            }
            chunkOverlapSlider.max = maxOverlap;
        });

        chunkOverlapSlider.addEventListener('input', (e) => {
            chunkOverlapVal.textContent = e.target.value;
        });

        // Drag and Drop Upload
        dropZone.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', handleFileSelect);
        
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
        
        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });
        
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                fileInput.files = e.dataTransfer.files;
                handleFileSelect();
            }
        });

        // Chunk Preview
        chunkPreviewBtn.addEventListener('click', previewDocumentChunks);

        // Index to DB
        indexDbBtn.addEventListener('click', indexChunksToDatabase);

        // Clear DB
        clearDbBtn.addEventListener('click', clearDatabaseStore);

        // Query Configuration changes
        retrievalModeSelect.addEventListener('change', (e) => {
            state.retrievalMode = e.target.value;
        });

        llmProviderSelect.addEventListener('change', (e) => {
            state.llmProvider = e.target.value;
            if (state.llmProvider === 'ollama') {
                ollamaModelGroup.style.display = 'flex';
            } else {
                ollamaModelGroup.style.display = 'none';
            }
        });

        // Send Query
        sendBtn.addEventListener('click', submitRAGQuery);
        queryInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                submitRAGQuery();
            }
        });

        // Trace Toggle
        traceToggle.addEventListener('click', () => {
            tracePanel.classList.toggle('closed');
        });
    }

    // SYSTEM UTILITIES: TOASTS
    function showToast(message, type = 'info') {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        let icon = 'fa-circle-info';
        if (type === 'success') icon = 'fa-circle-check';
        if (type === 'error') icon = 'fa-circle-exclamation';
        
        toast.innerHTML = `
            <i class="fa-solid ${icon}"></i>
            <span>${message}</span>
        `;
        
        container.appendChild(toast);
        
        // Slide out and remove
        setTimeout(() => {
            toast.style.animation = 'slideIn 0.3s ease reverse forwards';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }

    // INTERFACE CONTROLS
    function switchTab(tabId) {
        tabButtons.forEach(btn => {
            if (btn.getAttribute('data-tab') === tabId) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });

        tabPanes.forEach(pane => {
            if (pane.id === tabId) {
                pane.classList.add('active');
            } else {
                pane.classList.remove('active');
            }
        });
        state.activeTab = tabId;
    }

    // API CONFIG CALLS
    async function saveConfigKey(apiKey, showNotification = true) {
        try {
            const response = await fetch('/api/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ api_key: apiKey })
            });
            
            const result = await response.json();
            
            if (response.ok) {
                state.geminiKey = apiKey;
                localStorage.setItem('gemini_api_key', apiKey);
                
                statGemini.textContent = apiKey ? 'Active' : 'Unconfigured';
                statGemini.className = apiKey ? 'value status-active' : 'value status-inactive';
                statGemini.innerHTML = apiKey 
                    ? '<i class="fa-solid fa-circle-check"></i> Connected' 
                    : '<i class="fa-solid fa-circle-dot"></i> Unconfigured';
                
                if (showNotification) {
                    showToast('Gemini API key configured successfully!', 'success');
                }
            } else {
                throw new Error(result.detail || 'Failed to save configuration');
            }
        } catch (error) {
            console.error(error);
            if (showNotification) {
                showToast(error.message, 'error');
            }
        }
    }

    async function fetchStatus() {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();
            
            state.totalChunksIndexed = data.total_chunks;
            statChunks.textContent = state.total_chunks = data.total_chunks;
            
            state.geminiReady = data.gemini_ready;
            if (data.gemini_ready) {
                statGemini.className = 'value status-active';
                statGemini.innerHTML = '<i class="fa-solid fa-circle-check"></i> Connected';
            } else {
                statGemini.className = 'value status-inactive';
                statGemini.innerHTML = '<i class="fa-solid fa-circle-dot"></i> Unconfigured';
            }
            
            document.getElementById('stat-model').textContent = data.embedding_model.split('/').pop();
            
            // Enable Query Inputs if Database contains indexed vectors
            if (state.totalChunksIndexed > 0) {
                enableQueryInterface(true);
            } else {
                enableQueryInterface(false);
            }
        } catch (error) {
            console.error('Failed to retrieve server status:', error);
        }
    }

    function enableQueryInterface(enable) {
        if (enable) {
            queryInput.disabled = false;
            queryInput.placeholder = "Ask a question about your documents...";
            sendBtn.classList.remove('disabled');
            sendBtn.disabled = false;
        } else {
            queryInput.disabled = true;
            queryInput.placeholder = "Index documents first to search...";
            sendBtn.classList.add('disabled');
            sendBtn.disabled = true;
        }
    }

    // FILE HANDLING & UPLOAD
    function handleFileSelect() {
        const file = fileInput.files[0];
        if (!file) return;

        state.currentFile = file;
        
        // Show status panel
        fileStatusInfo.classList.remove('hidden');
        uploadedFileName.innerHTML = `<i class="fa-solid ${getFileIconClass(file.name)}"></i> ${file.name}`;
        uploadedFileSize.textContent = formatBytes(file.size);
        uploadProgress.style.width = '0%';
        
        // Reset preview state
        state.currentChunks = [];
        indexDbBtn.classList.add('disabled');
        indexDbBtn.disabled = true;
        previewPanel.classList.add('hidden');
        chunkVisualizer.innerHTML = '';
        
        showToast(`Selected file: ${file.name}`, 'info');
    }

    function getFileIconClass(filename) {
        const ext = filename.split('.').pop().toLowerCase();
        if (ext === 'pdf') return 'fa-file-pdf text-red';
        if (ext === 'md') return 'fa-file-lines text-blue';
        return 'fa-file-code text-purple';
    }

    function formatBytes(bytes, decimals = 2) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    }

    // CHUNKING LOGIC & PREVIEW
    async function previewDocumentChunks() {
        if (!state.currentFile) {
            showToast('Please upload a file first.', 'error');
            return;
        }

        const formData = new FormData();
        formData.append('file', state.currentFile);
        formData.append('chunk_size', chunkSizeSlider.value);
        formData.append('chunk_overlap', chunkOverlapSlider.value);
        formData.append('chunk_strategy', chunkStrategySelect.value);

        chunkPreviewBtn.disabled = true;
        chunkPreviewBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Splitting...';
        uploadProgress.style.width = '20%';

        try {
            uploadProgress.style.width = '50%';
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            uploadProgress.style.width = '80%';
            const data = await response.json();
            
            if (response.ok) {
                state.currentChunks = data.chunks;
                
                uploadProgress.style.width = '100%';
                showToast(`Split completed! Created ${data.chunk_count} chunks.`, 'success');
                
                // Show index button
                indexDbBtn.classList.remove('disabled');
                indexDbBtn.disabled = false;
                
                // Render Visual Chunks
                renderChunksVisualization(data.chunks);
            } else {
                throw new Error(data.detail || 'Chunking failed');
            }
        } catch (error) {
            console.error(error);
            showToast(error.message, 'error');
            uploadProgress.style.width = '0%';
        } finally {
            chunkPreviewBtn.disabled = false;
            chunkPreviewBtn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Process & Preview';
        }
    }

    function renderChunksVisualization(chunks) {
        previewPanel.classList.remove('hidden');
        previewCount.textContent = `${chunks.length} Chunks`;
        chunkVisualizer.innerHTML = '';

        chunks.forEach((chunk, i) => {
            const block = document.createElement('span');
            block.className = `chunk-block color-${i % 4}`;
            block.title = `Chunk ${i + 1} (${chunk.text.length} chars) - Click to view`;
            
            // Format text inside block (sub-preview)
            // Just truncate text
            const textPreview = chunk.text.length > 30 ? chunk.text.substring(0, 30) + '...' : chunk.text;
            block.textContent = `[${i + 1}] ${textPreview.replace(/\n/g, ' ')}`;
            
            // Handle block details preview click
            block.addEventListener('click', () => {
                // Remove selected class from others
                document.querySelectorAll('.chunk-block').forEach(b => b.classList.remove('selected'));
                block.classList.add('selected');
                
                // Display chunk in Chat as simulation note
                addSystemMessage(`<strong>Previewing Chunk [${i + 1}]</strong><br/><br/><code style="font-family: var(--font-mono); font-size: 12px; background: rgba(0,0,0,0.3); padding: 8px; display:block; border-radius:4px; white-space:pre-wrap;">${escapeHTML(chunk.text)}</code>`);
            });

            chunkVisualizer.appendChild(block);
        });
    }

    function escapeHTML(str) {
        return str.replace(/[&<>'"]/g, 
            tag => ({
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                "'": '&#39;',
                '"': '&quot;'
            }[tag] || tag)
        );
    }

    // INDEX TO VECTOR DB
    async function indexChunksToDatabase() {
        if (state.currentChunks.length === 0 || !state.currentFile) {
            showToast('No chunks to index. Run Process first.', 'error');
            return;
        }

        indexDbBtn.disabled = true;
        indexDbBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Writing Vectors...';

        try {
            const response = await fetch('/api/index', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    chunks: state.currentChunks,
                    filename: state.currentFile.name
                })
            });

            const data = await response.json();
            
            if (response.ok) {
                showToast(data.message, 'success');
                fetchStatus(); // Update stat pills and check count
                
                // Switch tab to show insights
                switchTab('compare-tab');
            } else {
                throw new Error(data.detail || 'Indexing failed');
            }
        } catch (error) {
            console.error(error);
            showToast(error.message, 'error');
        } finally {
            indexDbBtn.disabled = false;
            indexDbBtn.innerHTML = '<i class="fa-solid fa-database"></i> Index to DB';
        }
    }

    // CLEAR DATABASE STORE
    async function clearDatabaseStore() {
        if (!confirm('Are you sure you want to clear the entire vector database index and documents? This cannot be undone.')) {
            return;
        }

        try {
            const response = await fetch('/api/clear', { method: 'POST' });
            const data = await response.json();
            
            if (response.ok) {
                showToast(data.message, 'success');
                
                // Reset UI & State
                state.currentFile = null;
                state.currentChunks = [];
                fileInput.value = '';
                fileStatusInfo.classList.add('hidden');
                previewPanel.classList.add('hidden');
                chunkVisualizer.innerHTML = '';
                
                fetchStatus();
                
                // Clear chat
                chatThread.innerHTML = `
                    <div class="message system-msg">
                        <div class="msg-icon"><i class="fa-solid fa-robot"></i></div>
                        <div class="msg-content">
                            <p>Vector database and document stores cleared. Ready to start fresh!</p>
                        </div>
                    </div>
                `;
                
                // Clear trace details
                traceDetails.innerHTML = '<p class="trace-empty-msg">Perform a query to inspect the search mechanics, scores, and context injection.</p>';
            } else {
                throw new Error(data.detail || 'Failed to clear database');
            }
        } catch (error) {
            console.error(error);
            showToast(error.message, 'error');
        }
    }

    // SUBMIT RAG QUERY
    async function submitRAGQuery() {
        const queryText = queryInput.value.trim();
        if (!queryText) return;

        // Add user msg to UI
        addUserMessage(queryText);
        queryInput.value = '';

        // Add loading bubble
        const loadingId = addAssistantLoadingMessage();

        try {
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: queryText,
                    mode: state.retrievalMode,
                    k: 4,
                    provider: state.llmProvider,
                    model: state.llmProvider === 'ollama' ? ollamaModelInput.value.trim() : 'gemini-1.5-flash',
                    api_key: state.geminiKey
                })
            });

            const data = await response.json();
            
            // Remove loading
            document.getElementById(loadingId).remove();

            if (response.ok) {
                // Add assistant response
                addAssistantMessage(data.answer, data.provider, data.model, data.error);
                
                // Render Trace output
                renderQueryTrace(data);
            } else {
                throw new Error(data.detail || 'Retrieval-Generation pipeline failed');
            }
        } catch (error) {
            console.error(error);
            document.getElementById(loadingId)?.remove();
            addAssistantMessage(`Pipeline Error: ${error.message}`, 'system', 'error-node', true);
        }
    }

    // CHAT THREAD BUILDERS
    function addUserMessage(text) {
        const msg = document.createElement('div');
        msg.className = 'message user-msg';
        msg.innerHTML = `
            <div class="msg-icon"><i class="fa-solid fa-user"></i></div>
            <div class="msg-content">
                <p>${escapeHTML(text)}</p>
            </div>
        `;
        chatThread.appendChild(msg);
        scrollToBottom();
    }

    function addSystemMessage(htmlContent) {
        const msg = document.createElement('div');
        msg.className = 'message system-msg';
        msg.innerHTML = `
            <div class="msg-icon"><i class="fa-solid fa-laptop-code"></i></div>
            <div class="msg-content">
                <p>${htmlContent}</p>
            </div>
        `;
        chatThread.appendChild(msg);
        scrollToBottom();
    }

    function addAssistantLoadingMessage() {
        const id = 'loading-' + Date.now();
        const msg = document.createElement('div');
        msg.className = 'message assistant-msg';
        msg.id = id;
        msg.innerHTML = `
            <div class="msg-icon"><i class="fa-solid fa-spinner fa-spin"></i></div>
            <div class="msg-content">
                <p class="typing-loading">Querying indexes, fusing rankings, synthesizing reply...</p>
            </div>
        `;
        chatThread.appendChild(msg);
        scrollToBottom();
        return id;
    }

    function addAssistantMessage(text, provider, model, isError = false) {
        const msg = document.createElement('div');
        msg.className = `message assistant-msg ${isError ? 'error-bubble' : ''}`;
        
        let providerTag = '';
        if (provider === 'gemini') {
            providerTag = `<span class="badge" style="background: rgba(59, 130, 246, 0.1); border-color: rgba(59, 130, 246, 0.3); color:#60a5fa; margin-left: 8px;">Gemini API: ${model}</span>`;
        } else if (provider === 'ollama') {
            providerTag = `<span class="badge" style="background: rgba(168, 85, 247, 0.1); border-color: rgba(168, 85, 247, 0.3); color:#c084fc; margin-left: 8px;">Ollama Local: ${model}</span>`;
        } else if (provider === 'simulation') {
            providerTag = `<span class="badge" style="background: rgba(16, 185, 129, 0.1); border-color: rgba(16, 185, 129, 0.3); color:#34d399; margin-left: 8px;">Offline Simulator</span>`;
        }

        // Simple markdown links or text bold translation
        const formattedText = text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`([^`]+)`/g, '<code style="font-family: var(--font-mono); font-size:12px; background: rgba(255,255,255,0.08); padding: 2px 4px; border-radius: 3px;">$1</code>')
            .replace(/\n/g, '<br/>');

        msg.innerHTML = `
            <div class="msg-icon"><i class="fa-solid fa-robot"></i></div>
            <div class="msg-content">
                <div class="assistant-label" style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 6px;">
                    <strong style="font-size:12px; color:var(--accent-blue)">Response</strong>
                    ${providerTag}
                </div>
                <p>${formattedText}</p>
            </div>
        `;
        chatThread.appendChild(msg);
        scrollToBottom();
    }

    function scrollToBottom() {
        chatThread.scrollTop = chatThread.scrollHeight;
    }

    // RETRIEVAL TRACE RENDER
    function renderQueryTrace(data) {
        tracePanel.classList.remove('closed');
        traceDetails.innerHTML = '';

        const metaDiv = document.createElement('div');
        metaDiv.className = 'trace-meta';
        metaDiv.innerHTML = `
            <span><strong>Mode:</strong> ${data.provider.toUpperCase()}</span>
            <span>|</span>
            <span><strong>Retrieved:</strong> ${data.retrieved_chunks.length} chunks</span>
            <span>|</span>
            <span><strong>Strategy:</strong> ${retrievalModeSelect.options[retrievalModeSelect.selectedIndex].text}</span>
        `;
        traceDetails.appendChild(metaDiv);

        if (data.retrieved_chunks.length === 0) {
            const noChunkMsg = document.createElement('p');
            noChunkMsg.className = 'trace-empty-msg';
            noChunkMsg.textContent = 'No context chunks retrieved. Vector store matches are empty.';
            traceDetails.appendChild(noChunkMsg);
            return;
        }

        const chunkList = document.createElement('div');
        chunkList.className = 'trace-chunk-list';

        data.retrieved_chunks.forEach((chunk, i) => {
            const card = document.createElement('div');
            card.className = 'trace-chunk-card';
            
            // Format scores / ranks
            let scoreHTML = `<span class="trace-chunk-score">Similarity: ${chunk.score.toFixed(4)}</span>`;
            let rrfDetailsHTML = '';
            
            if (chunk.search_details) {
                const details = chunk.search_details;
                scoreHTML = `<span class="trace-chunk-score">RRF Fusion Score: ${chunk.score.toFixed(4)}</span>`;
                rrfDetailsHTML = `
                    <div class="trace-chunk-details">
                        <span><strong>Dense Rank:</strong> ${details.dense_rank || 'N/A'} (Score: ${details.dense_score.toFixed(4)})</span>
                        <span>|</span>
                        <span><strong>Sparse Rank:</strong> ${details.sparse_rank || 'N/A'} (Score: ${details.sparse_score.toFixed(2)})</span>
                    </div>
                `;
            }

            card.innerHTML = `
                <div class="trace-chunk-title">
                    <span>Source [${i + 1}]: Chunk #${chunk.global_id + 1} (${chunk.file_name})</span>
                    ${scoreHTML}
                </div>
                <div class="trace-chunk-text">${escapeHTML(chunk.text)}</div>
                ${rrfDetailsHTML}
            `;
            
            chunkList.appendChild(card);
        });

        traceDetails.appendChild(chunkList);
    }
});
