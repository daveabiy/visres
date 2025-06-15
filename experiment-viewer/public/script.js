const experimentImageMap = {
    "Experiment 1": "experiment1",
    "Experiment 2": "experiment2",
    "Experiment 3": "experiment3"
    // Add more mappings as needed
};

function showExperiment(experimentName) {
    document.getElementById('experiment-name').innerText = experimentName;
    const folder = experimentImageMap[experimentName];
    if (!folder) {
        alert("No images found for this experiment.");
        return;
    }
    document.getElementById('intensity-image').src = `images/${folder}/intensity.png`;
    document.getElementById('phase-image').src = `images/${folder}/phase.png`;
    document.getElementById('attenuation-image').src = `images/${folder}/attenuation.png`;

    document.getElementById('intensity-image').classList.remove('hidden');
    document.getElementById('phase-image').classList.remove('hidden');
    document.getElementById('attenuation-image').classList.remove('hidden');
}