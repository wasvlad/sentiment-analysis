class PipeLine:
    def __init__(self, loader, preprocessor, model_builder, data_augmenter=None):
        self.loader = loader
        self.preprocessor = preprocessor
        self.model_builder = model_builder
        self.model = None
        self.history = None
        self.data_augmenter = data_augmenter

    def preprocess(self, x, y, training=False):
        if self.data_augmenter and training:
            x, y = self.data_augmenter(x, y)
        preprocessed = self.preprocessor(x)
        import numpy as np
        if hasattr(preprocessed, 'detach') and hasattr(preprocessed, 'cpu'):
            x = preprocessed.detach().cpu().numpy()
        else:
            x = np.array(preprocessed)
        return x, y

    def train(self, x_train, y_train, x_val, y_val, epochs=100,
              class_weight=None):
        x_train, y_train = self.preprocess(x_train, y_train, training=True)
        x_val, y_val = self.preprocess(x_val, y_val, training=False)
        self.model = self.model_builder(x_train)
        from tensorflow import keras
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20,
                                                       restore_best_weights=True, mode='min')
        history = self.model.fit(x_train, y_train, validation_data=(x_val, y_val),
                                 class_weight=class_weight,
                                 epochs=epochs,
                                 callbacks=[early_stopping])
        self.history = history
        return history

    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def predict(self, data):
        preprocessed = self.preprocessor(data)
        import numpy as np
        if hasattr(preprocessed, 'detach') and hasattr(preprocessed, 'cpu'):
            x = preprocessed.detach().cpu().numpy()
        else:
            x = np.array(preprocessed)
        return self.model.predict(x)

    def evaluate(self, x, y):
        prediction = self.predict(x)
        import numpy as np
        y_predicted = np.argmax(prediction, axis=1)
        from sklearn.metrics import f1_score
        fscore = f1_score(y, y_predicted, average='weighted')
        return fscore

    @staticmethod
    def load(path) -> 'PipeLine':
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)
