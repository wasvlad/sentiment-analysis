from pipeline import PipeLine

pl = PipeLine.load("data/model-first.pkl")
text = input("Enter text: ")
while text:
    prediction = pl.predict([text])
    import numpy as np
    pred_idx = np.argmax(prediction[0])
    pred_prob = prediction[0][pred_idx]
    label_map = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    pred_label = label_map[pred_idx]
    print(f"Prediction: {pred_label} (probability: {pred_prob:.4f})")
    text = input("Enter text: ")
