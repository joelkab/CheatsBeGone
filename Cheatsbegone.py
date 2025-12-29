import os
# I'm keeping these lines to hide those annoying red TensorFlow warnings that clutter the output. prof if you remov this not sure if you will get errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

path = "players_stats.csv"
df = pd.read_csv(path)
X = df.drop('banned', axis=1)
y = df['banned']

# I need to scale the features so they are all on the same level standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting my data into training and testing sets 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, 
    test_size=0.2, 
    random_state=42
)

# I'm creating an extra split specifically for my Neural Network validation
X_train_nn, X_val_nn, y_train_nn, y_val_nn = train_test_split(
    X_train, to_categorical(y_train, num_classes=2),
    test_size=0.2,
    random_state=42
)

print("I am training the main models now...")

# Random Forest Model
# I chose 100 trees for my forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))

# SVM 
# I'm enabling probability estimates here so I can plot the ROC curve later
svm = SVC(probability=True, random_state=42)
svm.fit(X_train, y_train)
svm_acc = accuracy_score(y_test, svm.predict(X_test))

# My Custom Neural Network Function
def model_test(ner1, ner2, epochz, batchsz, LR):
    model = keras.Sequential()
    model.add(keras.Input(shape=(X_train.shape[1],)))
    
    # Hidden Layer 1 + Dropout
    # I added dropout here to randomly turn off 20% of neurons to stop overfitting
    model.add(keras.layers.Dense(ner1, activation="relu"))
    model.add(keras.layers.Dropout(0.2)) 
    
    # Hidden Layer 2 + Dropout
    # Another dropout layer for the second hidden layer
    model.add(keras.layers.Dense(ner2, activation="relu"))
    model.add(keras.layers.Dropout(0.2)) 
    
    # Output layer with 2 nodes (Legit vs Cheater)
    model.add(keras.layers.Dense(2, activation="softmax"))
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # Training the model with validation data to track progress
    trainZ = model.fit(
        X_train_nn, y_train_nn,
        batch_size=batchsz,
        epochs=epochz,
        validation_data=(X_val_nn, y_val_nn),
        verbose=1 
    )
    return model, trainZ

# I'm training my Neural Network with the parameters I found worked best
model1, trainZ1 = model_test(32, 16, 100, 16, 0.001)
#model1, trainZ1 = model_test(64, 32, 100, 32, 0.001)

print("\nMY RESULTS:")
print(f" Random Forest Accuracy: {rf_acc:.4f}")
print(f" SVM Accuracy: {svm_acc:.4f}")
print(f" Neural Net Training Accuracy: {max(trainZ1.history['accuracy']):.4f}")
print(f" Neural Net Validation Accuracy: {max(trainZ1.history['val_accuracy']):.4f}")

# Evaluating my Neural Network on the test set
y_test_cat = to_categorical(y_test, num_classes=2)
test_loss, nn_acc = model1.evaluate(X_test, y_test_cat, verbose=0)
print(f" Neural Net Test Accuracy: {nn_acc:.4f}")

# Confusion Matrix
# This helps me see where my model is making mistakes (False Positives vs False Negatives)
y_pred_prob = model1.predict(X_test)
y_pred_class = np.argmax(y_pred_prob, axis=1)
cm = confusion_matrix(y_test, y_pred_class)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Legit', 'Cheater'], 
            yticklabels=['Legit', 'Cheater'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# plotting the ROC curve to check the trade-off between sensitivity and specificity
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='orange', label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

#seeing the SVM 
print("\nGenerating my SVM Graph...")

#selecting "Total Matches" and "K/D Ratio" because they separate the groups nicely
feature_indices = [7, 3] 
feature_names = ["Total Matches", "K/D Ratio"]
X_simple = X_scaled[:, feature_indices]
y_simple = y.values

# Training a simple 2D SVM just for this vis
svm_simple = SVC(kernel='linear', C=1.0)
svm_simple.fit(X_simple, y_simple)

# removing points that are too close to the decision boundary to make the plot cleaner
distances = svm_simple.decision_function(X_simple)
threshold = 0.5 
mask = np.abs(distances) > threshold

X_clean = X_simple[mask]
y_clean = y_simple[mask]

plt.figure(figsize=(10, 6))

# Plotting Legit (Blue) and Cheater (Red) players
plt.scatter(X_clean[y_clean==0][:, 0], X_clean[y_clean==0][:, 1], 
            c='blue', s=60, edgecolors='k', alpha=0.8, label='Legit (Vet)')
plt.scatter(X_clean[y_clean==1][:, 0], X_clean[y_clean==1][:, 1], 
            c='red', s=60, edgecolors='k', alpha=0.8, label='Cheater (New Accounts)')

# Drawing the Decision Boundary line
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = svm_simple.decision_function(xy).reshape(XX.shape)

# Drawing lines 
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.8,
           linestyles=['--', '-', '--'])

plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title('SVM Decision Boundary (Separated Clusters)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

#thank you for a great semester