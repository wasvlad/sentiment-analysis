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
    x = tf.keras.layers.Dense(512, activation='relu')(inputs)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

def get_model_regulized(embedding_dim: int, num_classes: int = 6) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(embedding_dim,))
    x = tf.keras.layers.Dense(512, activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

def get_model_with_embeddings() -> tf.keras.Model:
    """
    Returns a TensorFlow model that trains embeddings.
    This is a placeholder function; actual implementation would depend on the specific embedding layer used.
    """
    # Example implementation, replace with actual embedding layer
    embedding_layer = tf.keras.layers.Embedding(input_dim=10000, output_dim=300, input_length=100)

    inputs = tf.keras.Input(shape=(100,))
    x = embedding_layer(inputs)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(6)(x)  # Assuming 6 classes for classification
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model