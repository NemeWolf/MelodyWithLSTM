import tensorflow.keras as keras
from preprocessing import generate_training_sequences, SEQUENCE_LENGTH # importa metodo de preprocesamiento y la longitud de la secuencia

# Hyperparameters
OUTPU_UNIT = 41 # cantidad de neuronas en la capa de salida (en este caso cantidad de notas musicales)
NUM_UNITS = [256] # cantidad de neuronas en la capa oculta (es una lista porque se pueden agregar mas capas ocultas) [256, 256]
LOSS = "sparse_categorical_crossentropy" # funcion de perdida
LEARNING_RATE = 0.001   # tasa de aprendizaje
EPOCHS = 50 
BATCH_SIZE = 64
SAVE_MODEL_PATH = "model.h5"

# Model

def build_model(output_units, num_units, loss, learning_rate):
    """Builds and compiles model

    :param output_units (int): Num output units
    :param num_units (list of int): Num of units in hidden layers
    :param loss (str): Type of loss function to use
    :param learning_rate (float): Learning rate to apply

    :return model (tf model): Where the magic happens :D
    """
    
    # create the model architecture - API funcional (flexible y explicita) 
    
    # API Funcional permite crear modelos mas complejos que la API secuencial (ejemplo: modelos con multiples entradas y salidas)  
    
    # Input layer    
    input = keras.layers.Input(shape=(None, output_units)) # None: longitud de la secuencia, output_units: cantidad de notas musicales
    
    # Al ser la longitud de la secuencia None, el modelo puede recibir secuencias de cualquier longitud
    
    # Construccion de las capas ocultas (add nodes)      
    x = keras.layers.LSTM(num_units[0])(input) # capa LSTM con 256 neuronas - pasa la salida de la capa anterior a la siguiente capa
    x = keras.layers.Dropout(0.2)(x) # capa de dropout para evitar overfitting 
    
    # Output layer    
    output = keras.layers.Dense(output_units, activation="softmax")(x) # capa densa con 41 neuronas 
    
    # create the model    
    model = keras.Model(input, output) # solo se necesita la capa de entrada y la capa de salida para crear el modelo
    
    # compile model - define the loss, the optimizer and the metrics
    model.compile(
        loss = loss,
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate),
        metrics = ["accuracy"])
    
    # print model summary
    model.summary()
    
    return model

def train(output_units=OUTPU_UNIT, num_units=NUM_UNITS, loss=LOSS, learning_rate=LEARNING_RATE):
    """Train and save TF model.

    :param output_units (int): Num output units
    :param num_units (list of int): Num of units in hidden layers
    :param loss (str): Type of loss function to use
    :param learning_rate (float): Learning rate to apply
    """
    
    # generate the training sequences
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH) # input -> (362402, 64, 41), target -> (362402, 41)
    # lo ideal para los datos es dividirlos en 3 conjuntos: entrenamiento, validacion y test
    
    # build the network
    model = build_model(output_units, num_units, loss, learning_rate) 
    # (neuronas en la capa de salida, neuronas en la capa oculta, funcion de perdida, tasa de aprendizaje)
    
    # train model
    model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE) 
    # epochs -> cantidad de veces que se entrena el modelo
    # batch_size -> cantidad de muestras que se utilizan para calcular el gradiente
    
    #save the model - guarda el modelo en un archivo .h5
    model.save(SAVE_MODEL_PATH)
    
if __name__ == "__main__":
    train()
                               
        