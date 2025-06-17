const experimentImageMap = {
    "Experiment 1": "experiment1",
    "Experiment 2": "experiment2",
    "Experiment 3": "experiment3"
    // Add more mappings as needed
};

// Get references to DOM elements
const loadingIndicator = document.getElementById('image-loading-indicator');
const intensityImage = document.getElementById('intensity-image');
const phaseImage = document.getElementById('phase-image');
const attenuationImage = document.getElementById('attenuation-image');
const architectureImage = document.getElementById('architecture-image');
const setupImage = document.getElementById('setup-image');

// Helper to hide all images
function hideAllImages() {
    intensityImage.classList.add('hidden');
    phaseImage.classList.add('hidden');
    attenuationImage.classList.add('hidden');
    architectureImage.classList.add('hidden');
    setupImage.classList.add('hidden');
}

// Generic function to load an image and manage loading indicator
function loadImage(imageElement, src) {
    imageElement.classList.add('hidden'); // Hide before loading new src
    loadingIndicator.classList.remove('hidden');

    imageElement.onload = () => {
        loadingIndicator.classList.add('hidden');
        imageElement.classList.remove('hidden');
    };
    imageElement.onerror = () => {
        loadingIndicator.classList.add('hidden');
        console.error(`Failed to load image: ${src}`);
        // Optionally, display a broken image icon or fallback
    };
    imageElement.src = src;
}

function showExperiment(experimentName) {
    document.getElementById('experiment-name').innerText = experimentName;
    const folder = experimentImageMap[experimentName];

    hideAllImages(); // Hide all images when a new experiment is selected

    if (!folder) {
        alert("No images found for this experiment.");
        loadingIndicator.classList.add('hidden');
        return;
    }

    loadImage(intensityImage, `images/${folder}/intensity.png`);
    loadImage(phaseImage, `images/${folder}/phase.png`);
    loadImage(attenuationImage, `images/${folder}/attenuation.png`);
}

// Function to show architecture and setup images
function showArchitectureAndSetup() {
    document.getElementById('experiment-name').innerText = "Architecture & Setup";
    hideAllImages();
    loadImage(architectureImage, 'images/architecture.png'); // Adjust path as needed
    loadImage(setupImage, 'images/setup.png'); // Adjust path as needed
}

// Add click event listeners for zoomable images
document.querySelectorAll('.zoomable').forEach(img => {
    img.addEventListener('click', () => {
        img.classList.toggle('zoomed');
    });
});