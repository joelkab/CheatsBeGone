CheatsBeGone: My Cheat Detection Project
This is my final project for CS422, where I developed a machine learning pipeline to identify cheaters in gaming datasets. The goal was to see if we could automatically flag "Banned" accounts based on performance anomalies like inflated K/D ratios and total matches played.
My Approach
I decided to compare three different types of machine learning models to see which one handles the noise of player data most effectively:
Random Forest: I used a forest of 100 trees to get a robust baseline accuracy.
Support Vector Machine (SVM): Beyond just classification, I used the SVM to visualize the "Decision Boundary"—literally drawing the line between what the computer thinks is a veteran player versus a new cheating account.
Custom Neural Network (Keras): I built a multi-layer deep learning model. I included Dropout layers to prevent the model from simply "memorizing" the training data (overfitting) and used a Softmax output to categorize players as either "Legit" or "Cheater."
