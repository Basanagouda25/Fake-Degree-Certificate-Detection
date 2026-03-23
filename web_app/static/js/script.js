document.addEventListener('DOMContentLoaded', () => {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    
    const uploadSection = document.getElementById('upload-section');
    const loadingSection = document.getElementById('loading-section');
    const resultSection = document.getElementById('result-section');
    const previewImage = document.getElementById('preview-image');
    const progressBar = document.getElementById('progress-bar');
    
    // Central Model UI elements
    const cenImage = document.getElementById('cen-image');
    const cenBadge = document.getElementById('cen-badge');
    const cenTitle = document.getElementById('cen-title');
    const cenConfidence = document.getElementById('cen-confidence');
    
    // Federated Model UI elements
    const flImage = document.getElementById('fl-image');
    const flBadge = document.getElementById('fl-badge');
    const flTitle = document.getElementById('fl-title');
    const flConfidence = document.getElementById('fl-confidence');
    
    const resetBtn = document.getElementById('reset-btn');

    dropzone.addEventListener('click', () => fileInput.click());

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) { e.preventDefault(); e.stopPropagation(); }

    ['dragenter', 'dragover'].forEach(eventName => { dropzone.addEventListener(eventName, () => dropzone.classList.add('active')); });
    ['dragleave', 'drop'].forEach(eventName => { dropzone.addEventListener(eventName, () => dropzone.classList.remove('active')); });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        if (dt.files && dt.files.length) { handleFile(dt.files[0]); }
    });

    fileInput.addEventListener('change', function() {
        if (this.files && this.files.length) { handleFile(this.files[0]); }
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

        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            cenImage.src = e.target.result;
            flImage.src = e.target.result;
            
            uploadSection.classList.add('hidden');
            loadingSection.classList.remove('hidden');
            
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
            if (progress > 90) progress = 90;
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
            }, 600);
        })
        .catch(error => {
            clearInterval(progressInterval);
            console.error('Error:', error);
            alert('Failed to connect to the server. Check console for details.');
            uploadSection.classList.remove('hidden');
            loadingSection.classList.add('hidden');
        });
    }

    function updateCardProps(data, badgeEl, titleEl, confEl) {
        if (data.status === 'error') {
            badgeEl.textContent = 'ERROR';
            badgeEl.className = 'badge warning';
            titleEl.textContent = 'Inference Failed';
            titleEl.className = 'text-warning';
            confEl.textContent = 'N/A';
            return;
        }

        badgeEl.textContent = data.label;
        badgeEl.className = 'badge ' + data.color;
        
        if(data.label === 'REAL') {
            titleEl.textContent = 'Authentication Passed';
            titleEl.className = 'text-success';
        } else if(data.label === 'FAKE') {
            titleEl.textContent = 'Counterfeit Detected';
            titleEl.className = 'text-danger';
        } else {
            titleEl.textContent = 'Authentication Uncertain';
            titleEl.className = 'text-warning';
        }
        
        confEl.textContent = (data.confidence * 100).toFixed(2) + '%';
        confEl.className = 'text-' + data.color;
    }

    function showResults(data) {
        loadingSection.classList.add('hidden');
        resultSection.classList.remove('hidden');
        
        updateCardProps(data.central, cenBadge, cenTitle, cenConfidence);
        updateCardProps(data.federated, flBadge, flTitle, flConfidence);
    }
});
