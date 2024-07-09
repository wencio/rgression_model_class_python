# Modelo de Clasificación de una Sola Neurona

Este proyecto implementa un modelo de clasificación de una sola neurona desde cero utilizando NumPy. El modelo se entrena utilizando la función de pérdida de log-verosimilitud negativa (NLL) y se evalúa en términos de precisión de clasificación.

## Estructura del Proyecto

- **SingleNeuronModel**: Clase base para los modelos de clasificación y regresión.
- **SingleNeuronClassificationModel**: Subclase que implementa la función de activación sigmoide para tareas de clasificación.
- **Funciones de Entrenamiento y Evaluación**: Funciones para entrenar el modelo y evaluar su precisión.

## Requisitos

- Python 3.x
- NumPy
- Pandas (para manipulación de datos)

## Uso

### 1. Preparar el entorno

Instala las dependencias necesarias:
```bash
pip install numpy pandas

Descripción del Código
Clase SingleNeuronModel
__init__(self, in_features): Inicializa los pesos y el sesgo con valores pequeños distribuidos de forma normal.
forward(self, x): Calcula la preactivación z, aplica la función de activación y devuelve la salida activada a.
update(self, grad_w, grad_w_0, learning_rate): Actualiza los pesos y el sesgo basándose en los gradientes y la tasa de aprendizaje.
Clase SingleNeuronClassificationModel
activation(self, z): Aplica la función de activación sigmoide.
gradient(self, x, errors): Calcula el gradiente de la pérdida respecto a los pesos y el sesgo.
Funciones de Entrenamiento y Evaluación
train_model_NLL_loss(model, input_data, labels, learning_rate, epochs): Entrena el modelo utilizando la pérdida de log-verosimilitud negativa.
evaluate_classification_accuracy(model, input_data, labels): Evalúa la precisión del modelo en un conjunto de datos.
Notas
Asegúrate de ajustar input_columns y output_columns según las columnas de tu DataFrame.
