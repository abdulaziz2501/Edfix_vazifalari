document.addEventListener('DOMContentLoaded', function() {
  // Elements
  const uploadContainer = document.getElementById('upload-container');
  const fileUpload = document.getElementById('file-upload');
  const imagePreviewContainer = document.getElementById('image-preview-container');
  const imagePreview = document.getElementById('image-preview');
  const removeImageBtn = document.getElementById('remove-image');
  const analyzeBtn = document.getElementById('analyze-button');
  const emptyState = document.getElementById('empty-state');
  const loadingContainer = document.getElementById('loading-container');
  const resultsContainer = document.getElementById('results-container');
  const progressBar = document.getElementById('progress-bar');
  const analysisStatus = document.getElementById('analysis-status');
  const resultsTitle = document.getElementById('results-title');

  // Demo disease prediction data - normally this would come from the AI model
  const samplePrediction = {
    "diabetic_retinopathy": {
      "confidence": 92.7,
      "severity": "O'rta NPDR"
    },
    "glaucoma": {
      "confidence": 8.3,
      "severity": "Normal"
    },
    "cataract": {
      "confidence": 5.1,
      "severity": "Normal"
    },
    "amd": {
      "confidence": 12.4,
      "severity": "Normal"
    }
  };

  // Analysis pipeline steps
  const analysisPipeline = [
    "Tasvir sifatini tekshirish...",
    "To'r parda tuzilmalarini segmentatsiya qilish...",
    "Kasallik belgilarini izlash...",
    "Diagnostika natijalarini tayyorlash..."
  ];

  // Event: Click on upload container
  uploadContainer.addEventListener('click', function() {
    fileUpload.click();
  });

  // Event: File selected
  fileUpload.addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function(event) {
      imagePreview.src = event.target.result;
      uploadContainer.classList.add('hidden');
      imagePreviewContainer.classList.remove('hidden');
      emptyState.classList.add('hidden');
      resultsContainer.classList.add('hidden');
    };
    reader.readAsDataURL(file);
  });

  // Event: Remove image
  removeImageBtn.addEventListener('click', function() {
    imagePreview.src = '';
    fileUpload.value = '';
    uploadContainer.classList.remove('hidden');
    imagePreviewContainer.classList.add('hidden');
    emptyState.classList.remove('hidden');
    resultsContainer.classList.add('hidden');
    loadingContainer.classList.add('hidden');
    resultsTitle.textContent = 'Natija';
  });

  // Event: Analyze image
  analyzeBtn.addEventListener('click', function() {
    // Show loading state
    emptyState.classList.add('hidden');
    loadingContainer.classList.remove('hidden');
    resultsContainer.classList.add('hidden');
    resultsTitle.textContent = 'Tahlil jarayoni';

    // Reset progress
    progressBar.style.width = '0%';
    let currentStage = 0;

    // Simulate analysis pipeline
    const interval = setInterval(function() {
      currentStage++;
      const progress = (currentStage / analysisPipeline.length) * 100;
      progressBar.style.width = `${progress}%`;

      if (currentStage <= analysisPipeline.length) {
        analysisStatus.textContent = analysisPipeline[currentStage - 1];
      }

      if (currentStage > analysisPipeline.length) {
        clearInterval(interval);
        showResults();
      }
    }, 1000);
  });

  function showResults() {
    // Update UI
    loadingContainer.classList.add('hidden');
    resultsContainer.classList.remove('hidden');
    resultsTitle.textContent = 'Diagnostika natijalari';

    // Update values in the UI based on the prediction
    document.getElementById('dr-confidence').textContent = `${samplePrediction.diabetic_retinopathy.confidence}%`;
    document.getElementById('dr-bar').style.width = `${samplePrediction.diabetic_retinopathy.confidence}%`;
    document.getElementById('dr-severity').textContent = `Daraja: ${samplePrediction.diabetic_retinopathy.severity}`;

    document.getElementById('gl-confidence').textContent = `${samplePrediction.glaucoma.confidence}%`;
    document.getElementById('gl-bar').style.width = `${samplePrediction.glaucoma.confidence}%`;
    document.getElementById('gl-severity').textContent = `Daraja: ${samplePrediction.glaucoma.severity}`;

    document.getElementById('cat-confidence').textContent = `${samplePrediction.cataract.confidence}%`;
    document.getElementById('cat-bar').style.width = `${samplePrediction.cataract.confidence}%`;
    document.getElementById('cat-severity').textContent = `Daraja: ${samplePrediction.cataract.severity}`;

    document.getElementById('amd-confidence').textContent = `${samplePrediction.amd.confidence}%`;
    document.getElementById('amd-bar').style.width = `${samplePrediction.amd.confidence}%`;
    document.getElementById('amd-severity').textContent = `Daraja: ${samplePrediction.amd.severity}`;
  }
});