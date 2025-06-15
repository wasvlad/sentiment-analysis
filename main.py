from matplotlib import pyplot as plt

from pipeline import PipeLine

from data_loader import load_data
from data_preprocessing import s_bert, bag_of_words
from models import get_classification_model

pl = PipeLine(loader=load_data,
              preprocessor=bag_of_words,
              model_builder=get_classification_model)

history = pl.train()
plt.plot(history.history['val_loss'])
plt.show()

pl.save("data/model.pkl")
