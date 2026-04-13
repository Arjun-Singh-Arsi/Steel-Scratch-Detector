document.addEventListener('DOMContentLoaded', () => {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const predictBtn = document.getElementById('predict-btn');
    const loader = document.getElementById('loader');
    const resultsSection = document.getElementById('results-section');
    const processedImg = document.getElementById('processed-img');
    const defectList = document.getElementById('defect-list');

    let selectedFile = null;

    // Trigger file input on click
    dropzone.addEventListener('click', () => fileInput.click());

    // Drag and Drop handling
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('active');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('active');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('active');
        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    function handleFileSelect(file) {
        selectedFile = file;
        dropzone.querySelector('p').innerHTML = `Selected: <span style="color: var(--primary); font-weight: 600;">${file.name}</span>`;
        predictBtn.disabled = false;
        resultsSection.style.display = 'none';
    }

    predictBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        const formData = new FormData();
        formData.append('file', selectedFile);

        // UI State
        predictBtn.disabled = true;
        loader.style.display = 'block';
        resultsSection.style.display = 'none';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.error) {
                alert(data.error);
                return;
            }

            // Display results
            processedImg.src = `/static/predictions/${data.processed_image}?t=${new Date().getTime()}`;
            
            // Render defect info
            defectList.innerHTML = '';
            data.defects.forEach((defect, index) => {
                const item = document.createElement('div');
                item.className = `metric-item ${defect.detected ? 'detected-' + (index + 1) : ''}`;
                
                const statusClass = defect.detected ? 'status-positive' : 'status-negative';
                const statusText = defect.detected ? 'DETECTED' : 'NOT DETECTED';
                
                item.innerHTML = `
                    <div>
                        <span class="class-name">${defect.class}</span>
                        <span class="status-badge ${statusClass}">${statusText}</span>
                    </div>
                    <div class="confidence-badge">
                        ${defect.confidence}
                    </div>
                `;
                defectList.appendChild(item);
            });

            resultsSection.style.display = 'grid';
            resultsSection.scrollIntoView({ behavior: 'smooth' });

        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred during prediction.');
        } finally {
            loader.style.display = 'none';
            predictBtn.disabled = false;
        }
    });
});
