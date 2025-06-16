import matplotlib.pyplot as plt
import numpy as np

from data_loader import load_data
from data_preprocessing import data_balancer

x_train, y_train, x_val, y_val, x_test, y_test = load_data()

label_mapping = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise',
}

x_train, y_train = data_balancer(x_train, y_train)
# Count label occurrences
label_counts = np.bincount(y_train)

# Map label numbers to text
label_names = [label_mapping[label] for label in range(len(label_counts))]

# Plot
plt.figure(figsize=(8, 5))
plt.bar(label_names, label_counts, color='skyblue')
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.title('Distribution of Emotions in Dataset')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()
