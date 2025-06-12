import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data/raw_data.csv', index_col=0)

print(data.head())

label_mapping = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise',
}

# Count label occurrences
label_counts = data['label'].value_counts().sort_index()

# Map label numbers to text
label_names = [label_mapping[label] for label in label_counts.index]

# Plot
plt.figure(figsize=(8, 5))
plt.bar(label_names, label_counts.values, color='skyblue')
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.title('Distribution of Emotions in Dataset')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()
