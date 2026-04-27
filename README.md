# Sign-Language-AI
This project was developed by three students from the Samsung Innovation Campus, including myself. The workflow is divided into three phases: data collection, model training, and game integration.

Game Logic and Approach
To understand the technical details, it is important to first understand how the game handles static and dynamic gestures. Since static gestures are the most common, we prioritized speed and fluidity by creating two distinct pipelines:

Static gestures: Handled by a dedicated model for quick response.

Dynamic gestures: Handled by a separate model designed for movement.

The game generates a random letter and, based on its type, selects the appropriate capture method and prediction model.

1. Data Collection
We developed a Python script using OpenCV to capture single-hand data. The program tracks 30 hand landmarks, centers them using the wrist as the origin, and normalizes the coordinates. For dynamic data, the script follows the same logic but groups frames into batches of 30 to capture temporal movement.

2. Model Training
We used two different approaches across separate notebooks:

Random Forest: Trained for static gestures due to its robustness against outliers and slightly imperfect hand positions.

LSTM (Long Short-Term Memory): A neural network trained on dynamic data to leverage its ability to process sequential patterns and "memory."

3. Integration and UI
The final Python program manages the game logic and user interface. It displays real-time progress, including the current score, the target letter, the model's prediction, and the confidence level.

In order to play the game, the data folder is necessary.

It is an engaging and dynamic game. We encourage you to play and learn more about sign language to help increase visibility for the Deaf community.
