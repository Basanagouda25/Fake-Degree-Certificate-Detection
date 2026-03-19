document.addEventListener('DOMContentLoaded', () => {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    
    const uploadSection = document.getElementById('upload-section');
    const loadingSection = document.getElementById('loading-section');
    const resultSection = document.getElementById('result-section');
    
    const previewImage = document.getElementById('preview-image');
    const progressBar = document.getElementById('progress-bar');
    
    const resultImage = document.getElementById('result-image');
    const resultBadge = document.getElementById('result-badge');
    const resultTitle = document.getElementById('result-title');
    const confidenceText = document.getElementById('confidence-text');
    const rawScore = document.getElementById('raw-score');
    
    const resetBtn = document.getElementById('reset-btn');

    // Click to upload
    dropzone.addEventListener('click', () => fileInput.click());

    // Drag and drop events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, () => dropzone.classList.add('active'));
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, () => dropzone.classList.remove('active'));
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        if (dt.files && dt.files.length) {
            handleFile(dt.files[0]);
        }
    });

    fileInput.addEventListener('change', function() {
        if (this.files && this.files.length) {
            handleFile(this.files[0]);
        }
    });

    resetBtn.addEventListener('click', () => {
        resultSection.classList.add('hidden');
        uploadSection.classList.remove('hidden');
        fileInput.value = '';
    });

    function handleFile(file) {
        if (!file.type.match('image.*')) {
            alert('Please select a valid image file (JPG, PNG)');
            return;
        }

        // Show local preview immediately
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            resultImage.src = e.target.result;
            
            // Switch UI state
            uploadSection.classList.add('hidden');
            loadingSection.classList.remove('hidden');
            
            // Upload to server
            uploadAndPredict(file);
        };
        reader.readAsDataURL(file);
    }

    function uploadAndPredict(file) {
        const formData = new FormData();
        formData.append('file', file);
        
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress > 90) progress = 90; // Wait for server to finish
            progressBar.style.width = Math.min(progress, 90) + '%';
        }, 300);

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            clearInterval(progressInterval);
            progressBar.style.width = '100%';
            
            setTimeout(() => {
                if(data.error) {
                    alert('Error: ' + data.error);
                    uploadSection.classList.remove('hidden');
                    loadingSection.classList.add('hidden');
                    return;
                }
                
                showResults(data);
            }, 500); // Give a split second for 100% progress animation
        })
        .catch(error => {
            clearInterval(progressInterval);
            console.error('Error:', error);
            alert('Failed to connect to the server. Check console for details.');
            uploadSection.classList.remove('hidden');
            loadingSection.classList.add('hidden');
        });
    }

    function showResults(data) {
        loadingSection.classList.add('hidden');
        resultSection.classList.remove('hidden');
        
        // Update UI with response data
        resultBadge.textContent = data.label;
        resultBadge.className = 'badge ' + data.color;
        
        if(data.label === 'REAL') {
            resultTitle.textContent = 'Authentication Passed';
            resultTitle.className = 'text-success';
        } else if(data.label === 'FAKE') {
            resultTitle.textContent = 'Counterfeit Detected';
            resultTitle.className = 'text-danger';
        } else {
            resultTitle.textContent = 'Authentication Uncertain';
            resultTitle.className = 'text-warning';
        }
        
        confidenceText.textContent = (data.confidence * 100).toFixed(1) + '%';
        confidenceText.className = 'text-' + data.color;
        rawScore.textContent = data.score.toFixed(4);
    }
});
