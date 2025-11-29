import tensorflow as tf
from tensorflow.keras import layers, models

# Construimos la CNN inicial para entrenar con CelebA.
def construir_cnn_base(input_shape=(224, 224, 3), num_classes=40):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        
        layers.Dense(num_classes, activation='sigmoid', name='celeba_output') ])
    return model
