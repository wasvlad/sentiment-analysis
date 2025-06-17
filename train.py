from matplotlib import pyplot as plt

from pipeline import PipeLine

from data_loader import load_data
from data_preprocessing import s_bert, bag_of_words, data_balancer, s_bert_better, get_tokenizer, s_bert_best
from models import get_classification_model, get_model_regulized, get_model_with_embeddings


x_train, y_train, x_val, y_val, x_test, y_test = load_data()
pl = PipeLine(loader=load_data,
              preprocessor=s_bert_best,
              model_builder=get_model_regulized)


history = pl.train(x_train, y_train, x_val, y_val, epochs=1000)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.show()

pl.save("data/model.pkl")

print(f"Train f1-score: {pl.evaluate(x_train, y_train)}")
print(f"Val f1-score: {pl.evaluate(x_val, y_val)}")
print(f"Test f1-score: {pl.evaluate(x_test, y_test)}")