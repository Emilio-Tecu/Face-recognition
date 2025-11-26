import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Creamos el generador de imágenes.
def obtener_generador_augmentado(ruta_imagenes, batch_size=32):
    # Configuración para crear variaciones de las fotos.
    datagen = ImageDataGenerator(
        rescale=1./255,            
        rotation_range=20,          
        width_shift_range=0.2,       
        height_shift_range=0.2,    
        shear_range=0.2,           
        zoom_range=0.2,             
        horizontal_flip=True,       
        fill_mode='nearest')

    generador = datagen.flow_from_directory(
        directory=ruta_imagenes,
        target_size=(224, 224),     
        batch_size=batch_size,
        class_mode='binary',      
        shuffle=True)
    
    return generador
