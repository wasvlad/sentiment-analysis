class PipeLine:
    def __init__(self, loader, preprocessor, model_builder):
        self.loader = loader
        self.preprocessor = preprocessor
        self.model_builder = model_builder
        self.model = None
        self.history = None

    def train(self, epochs=100):
        data = self.loader()
        preprocessed = self.preprocessor(data['text'])
        import numpy as np
        if hasattr(preprocessed, 'detach') and hasattr(preprocessed, 'cpu'):
            x = preprocessed.detach().cpu().numpy()
        else:
            x = np.array(preprocessed)
        y = data['label'].values
        self.model = self.model_builder(embedding_dim=x.shape[1])
        from tensorflow import keras
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, mode='min')
        history = self.model.fit(x, y, epochs=epochs, validation_split=0.2, callbacks=[early_stopping])
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
        model = self.model_builder(embedding_dim=x.shape[1])
        return model.predict(x)

    @staticmethod
    def load(path) -> 'PipeLine':
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)
