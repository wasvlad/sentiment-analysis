import tensorflow as tf

def get_classification_model(x, num_classes: int = 6) -> tf.keras.Model:
    """
    Returns a TensorFlow classification model that takes embeddings as input.
    Args:
        embedding_dim (int): Dimension of the input embeddings.
        num_classes (int): Number of output classes. Default is 6.
    Returns:
        tf.keras.Model: Compiled classification model.
    """
    inputs = tf.keras.Input(shape=(x.shape[1],))
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

def get_model_regulized(x, num_classes: int = 6) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(x.shape[1],))
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

def get_model_with_embeddings(x, num_classes: int = 6) -> tf.keras.Model:
    """
    Returns a TensorFlow model that trains embeddings.
    This is a placeholder function; actual implementation would depend on the specific embedding layer used.
    """
    # Example implementation, replace with actual embedding layer
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=x.shape[1], output_dim=1024),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model