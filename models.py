import tensorflow as tf

def get_classification_model(num_classes: int = 6) -> tf.keras.Model:
    """
    Returns a TensorFlow classification model that takes embeddings as input.
    The input shape will be inferred automatically when the model is built/trained.

    Args:
        num_classes (int): Number of output classes. Default is 6.

    Returns:
        tf.keras.Model: Compiled classification model.
    """
    inputs = tf.keras.Input(shape=(None,))  # shape is unspecified, will be set at build/train time
    x = tf.keras.layers.Dense(128, activation='relu')(inputs)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
