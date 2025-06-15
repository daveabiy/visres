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
        <h1>Experiment Name: <span id="experiment-name">jd_mg</span></h1>
        <button id="show-images">Show Results</button>
        
        <div id="results" class="hidden">
            <h2>Results</h2>
            <div class="image-container">
                <div>
                    <h3>Intensity Image</h3>
                    <img id="intensity-image" src="path/to/intensity_image.png" alt="Intensity Image">
                </div>
                <div>
                    <h3>Reconstructed Phase</h3>
                    <img id="phase-image" src="path/to/phase_image.png" alt="Reconstructed Phase">
                </div>
                <div>
                    <h3>Attenuation</h3>
                    <img id="attenuation-image" src="path/to/attenuation_image.png" alt="Attenuation">
                </div>
            </div>
        </div>
    </div>

    <script src="script.js"></script>
</body>
</html>