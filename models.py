import tensorflow as tf

def get_classification_model(embedding_dim: int, num_classes: int = 6) -> tf.keras.Model:
    """
    Returns a TensorFlow classification model that takes embeddings as input.
    Args:
        embedding_dim (int): Dimension of the input embeddings.
        num_classes (int): Number of output classes. Default is 6.
    Returns:
        tf.keras.Model: Compiled classification model.
    """
    inputs = tf.keras.Input(shape=(embedding_dim,))
    x = tf.keras.layers.Dense(1024, activation='relu')(inputs)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    #x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model
