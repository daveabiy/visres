<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Experiment Visualization</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="container">
        <h1>Experiment Visualization</h1>
        <div class="experiment-list">
            <div class="experiment" onclick="showResults('experiment1')">Experiment 1</div>
            <div class="experiment" onclick="showResults('experiment2')">Experiment 2</div>
            <div class="experiment" onclick="showResults('experiment3')">Experiment 3</div>
        </div>
        
        <div id="results" class="results">
            <h2 id="experiment-name"></h2>
            <div class="images">
                <img id="intensity-image" src="" alt="Intensity Image" class="hidden">
                <img id="phase-image" src="" alt="Reconstructed Phase" class="hidden">
                <img id="attenuation-image" src="" alt="Attenuation" class="hidden">
            </div>
        </div>
    </div>

    <script src="script.js"></script>
</body>
</html>