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

# Construimos la función para colocar otro clasificador.
def construir_modelo_transferencia(modelo_base_entrenado):
    # Recuperamos la salida del último flatten.
    
    # Usamos el modelo hasta flatten.
    feature_extractor = models.Sequential(modelo_base_entrenado.layers[:-2]) 
    
    # Congelamos los pesos.
    feature_extractor.trainable = False 
    
    # Contruimos el nuevo modelo.
    model = models.Sequential([
        feature_extractor,
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')])
    
    return model
