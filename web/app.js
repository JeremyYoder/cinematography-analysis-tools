// AI Shot Classifier - Browser Native Logic
const SHOT_TYPES = ["CS", "ECS", "FS", "LS", "MS"];
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

let session;

// Elements
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const previewImg = document.getElementById('preview-img');
const resultsContainer = document.getElementById('results');
const loader = document.getElementById('loader');

// Initialize ONNX Runtime
async function init() {
    try {
        loader.classList.add('active');
        console.log("Loading ONNX Session...");
        session = await ort.InferenceSession.create('./model.onnx', {
            executionProviders: ['webgl'], // Prioritize GPU
            graphOptimizationLevel: 'all'
        });
        console.log("ONNX Model Loaded successfully.");
        loader.classList.remove('active');
    } catch (e) {
        console.error("Failed to load ONNX model:", e);
        alert("Wait! Browsers often block large .onnx files via local file protocol. Start a local server (like 'python -m http.server') to load the model.");
        loader.classList.remove('active');
    }
}

// Preprocess Image (PIL style)
async function preprocess(imgElement) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 224;
    canvas.height = 224;

    // Draw and resize
    ctx.drawImage(imgElement, 0, 0, 224, 224);
    const imageData = ctx.getImageData(0, 0, 224, 224).data;

    // Convert to Float32 Tensor (NCHW)
    const [red, green, blue] = [[], [], []];
    for (let i = 0; i < imageData.length; i += 4) {
        red.push((imageData[i] / 255 - MEAN[0]) / STD[0]);
        green.push((imageData[i + 1] / 255 - MEAN[1]) / STD[1]);
        blue.push((imageData[i + 2] / 255 - MEAN[2]) / STD[2]);
    }

    const float32Data = new Float32Array([...red, ...green, ...blue]);
    return new ort.Tensor('float32', float32Data, [1, 3, 224, 224]);
}

// Inference
async function classify(image) {
    if (!session) return;
    loader.classList.add('active');

    try {
        const inputTensor = await preprocess(image);
        const feeds = { input_tensor: inputTensor };
        const results = await session.run(feeds);
        const logits = results.classification_logits.data;

        // Apply Softmax
        const expLogits = logits.map(Math.exp);
        const sumExp = expLogits.reduce((a, b) => a + b, 0);
        const probabilities = expLogits.map(v => (v / sumExp) * 100);

        displayResults(probabilities);
    } catch (e) {
        console.error("Inference Error:", e);
    } finally {
        loader.classList.remove('active');
    }
}

function displayResults(probs) {
    resultsContainer.classList.add('visible');
    
    // Probs are mapped to SHOT_TYPES (CS, ECS, FS, LS, MS)
    SHOT_TYPES.forEach((type, i) => {
        const percent = probs[i].toFixed(1);
        document.getElementById(`p-${type}`).innerText = `${percent}%`;
        document.getElementById(`bar-${type}`).style.width = `${percent}%`;
    });
}

// Drag & Drop
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    handleFile(file);
});

dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        fileInput.click();
    }
});

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    handleFile(file);
});

function handleFile(file) {
    if (!file || !file.type.startsWith('image/')) return;

    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            previewImg.src = img.src;
            classify(img);
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

// Start
init();
