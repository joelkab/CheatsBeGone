# CheatsBeGone: My Cheat Detection Project

This is my final project for CS422, where I developed a machine learning pipeline to identify cheaters in gaming datasets. My goal was to see if we could automatically flag "Banned" accounts based on performance anomalies like inflated K/D ratios and total matches played.

## My Approach
I decided to compare three different types of machine learning models to see which one handles the noise of player data most effectively:

*   **Random Forest:** I used a forest of 100 trees to get a robust baseline accuracy.
*   **Support Vector Machine (SVM):** Beyond just classification, I used the SVM to visualize the "Decision Boundary"—literally drawing the line between what the computer thinks is a veteran player versus a new cheating account.
*   **Custom Neural Network (Keras):** I built a multi-layer deep learning model. I included **Dropout layers** to prevent the model from simply "memorizing" the training data (overfitting) and used a **Softmax** output to categorize players as either "Legit" or "Cheater."

*   **Validation:** For the Neural Network, I used a specific validation split to monitor accuracy in real-time during training.
*   **Analysis:** I didn't just look at accuracy; I generated a **Confusion Matrix** to see if my model was accidentally banning legitimate players (False Positives) and an **ROC Curve** to measure the overall strength of the predictions.


<p align="center">
  <img src="https://github.com/user-attachments/assets/973ba52d-4314-48b4-be34-7e87613f3d4b" width="48%" />
  <img src="https://github.com/user-attachments/assets/b6ac6ee7-4a86-4de4-82f1-2370c62a12c3" width="48%" />
</p>



## How to Run It
Make sure you have the `players_stats.csv` file in the same folder, then install the dependencies and run the script:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow
python Cheatsbegone.py
