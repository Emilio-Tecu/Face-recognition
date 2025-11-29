Este proyecto implementa un sistema de autenticación biométrica diseñado para funcionar con un dataset personal limitado, se utiliza una CNN con Fine-Tuning para la extracción de características faciales y una SVM para la clasificación final. 
Esta arquitectura híbrida resuelve los problemas de baja confianza y overfitting que se obtuvieron durante los entrenamientos, logrando distinguir eficazmente al usuario autorizado de desconocidos y falsos positivos.

Para el entrenamiento de la CNN se utilizó el dataset de celebA, mientras que para el clasificador y el fine tunning se utilizaron dos carpetas, una con imagenes de personas, objetos y dibujos y la otra con un total de 60 fotos mías.
Al evaluar el fine tunning se obtuvieron las respuestas esperadas, no obstante su confianza era muy baja, es decir, a otras personas las evaluaba como que no soy yo a una confianza de aproximadamente 40% y a mí que sí lo soy con una confianza aproximada de 60%.
Por ello se quitó la última capa que decide si soy yo o no con una función de activación sigmoide, y se implementó un SVM, ya que este encuentra la separación más ancha posible entre ambos grupos.
Finalmente se obtuvo un resultado bastante bueno ya que la red tuvo más confianza para clasificar, el resultado final con fotos que nunca había visto fue el siguiente.
<img width="1421" height="690" alt="image" src="https://github.com/user-attachments/assets/666bc462-3ebe-4a48-b0f5-d9e380ed3df8" />
