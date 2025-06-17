import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from data_loader import load_data
from pipeline import PipeLine

x_train, y_train, x_val, y_val, x_test, y_test = load_data()
pl = PipeLine.load("data/model.pkl")

# print(f"Train f1-score: {pl.evaluate(x_train, y_train)}")
# print(f"Val f1-score: {pl.evaluate(x_val, y_val)}")
print(f"Test f1-score: {pl.evaluate(x_test, y_test)}")

y_predicted = pl.predict(x_val)
y_predicted = np.argmax(y_predicted, axis=1)

# Per-label F1 scores
report = classification_report(y_val, y_predicted, target_names=["sadness", "joy", "love", "anger", "fear", "surprise"])
print("\nClassification Report (per-label F1 scores):\n", report)

cm = confusion_matrix(y_val, y_predicted)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["sadness", "joy", "love", "anger", "fear", "surprise"], yticklabels=["sadness", "joy", "love", "anger", "fear", "surprise"])
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix Heatmap')
plt.tight_layout()
plt.show()
