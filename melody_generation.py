import json
import numpy as np
import tensorflow.keras as keras
from preprocessing import SEQUENCE_LENGTH, MAPPING_PATH

class MelodyGeneration:

    def __init__(self, model_path="model.h5"):
        
        self.model_path = model_path
        self.model = keras.models.load_model(model_path) # carga el modelo entrenado

        # Load mappings (vocabularies)
        with open(MAPPING_PATH, "r") as fp:
            self._mappings = json.load(fp)

        # recordar que "/" separa las canciones en el dataset y es del tamaño de la longitud de la secuencia
        self._start_symbols = ["/"] * SEQUENCE_LENGTH
        
    def generate_melody(self, seed, num_steps, max_sequence_length, temperature):
        """Generates a melody using the DL model and returns a midi file.

        :param seed (str): Melody seed with the notation used to encode the dataset
        :param num_steps (int): Number of steps to be generated
        :param max_sequence_len (int): Max number of steps in seed to be considered for generation
        :param temperature (float): Float in interval [0, 1]. Numbers closer to 0 make the model more deterministic.
            A number closer to 1 makes the generation more unpredictable.

        :return melody (list of str): List with symbols representing a melody
        """
        # create seed with start symbols
        seed = seed.split()
        melody = seed        
        seed = self._start_symbols + seed
        
        # map seed to integers
        seed = [self._mappings[symbol] for symbol in seed]
        
        for _ in range(num_steps):
            # itera sobre el numero de pasos
            
            # limit the seed to max_sequence_length
            seed = seed[-max_sequence_length:]
            
            # one-hot encode the seed
            onehot_seed = keras.utils.to_categorical(seed, num_classes=len(self._mappings))
            # (1, max_sequence_length, num of symbols in the vocabulary)            
            # keras expects a batch size, so we add an extra dimension to the seed (batch_size=1)
            
            onehot_seed = onehot_seed[np.newaxis, ...] # agrega una dimension extra al principio
                        
            # make a prediction
            probabilities = self.model.predict(onehot_seed)[0]
            # [0] es para obtener solo un array de probabilidades. Si batch_size > 1, se obtiene un array de arrays de probabilidades
            # [0.1, 0.2, 0.1, 0.6] -> 1

            output_int = self._sample_with_temperature(probabilities, temperature)
            
            # update seed
            seed.append(output_int)
            
            # map in to our encoding
            output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]
            
            # check if we're at the end of the melody
            if output_symbol == "/":
                break
            
            # update melody
            melody.append(output_symbol)
            
        return melody
            
    def _sample_with_temperature(self, probabilities, temperature):
        """Samples an index from a probability array reapplying softmax using temperature

        :param predictions (nd.array): Array containing probabilities for each of the possible outputs.
        :param temperature (float): Float in interval [0, 1]. Numbers closer to 0 make the model more deterministic.
            A number closer to 1 makes the generation more unpredictable.

        :return index (int): Selected output symbol
        """
        
        # we want explore the space of probabilities
        
        prediction = np.log(probabilities) / temperature
        
        # Se aplica el logaritmo natural a las probabilidades para suavizar las diferencias entre ellas.
        # Luego, se divide por la temperatura. Esto ajusta la "nitidez" de la distribución de probabilidades:
        # Temperatura baja (cercana a 0): Las diferencias entre las probabilidades se amplifican, haciendo que la salida sea más determinista.
        # Temperatura alta (cercana a 1): Las diferencias entre las probabilidades se reducen, haciendo que la salida sea más aleatoria.
        
        # resample the probabilities distribution
        
        probabilities = np.exp(prediction) / np.sum(np.exp(prediction)) # softmax function
        
        #Se aplica la exponencial a las predicciones ajustadas para devolverlas al espacio de probabilidades.
        #Luego, se normalizan las probabilidades dividiendo por la suma de todas las exponenciales.
        
        choice = range(len(probabilities)) # [0, 1, 2, 3]
        index = np.random.choice(choice, p=probabilities)
        
        return index
        
if __name__ == "__main__":
    mg = MelodyGeneration()
    seed = "55 _ _ _ 60 _ _ _ 55 _ _ _ 55 _"
    melody = mg.generate_melody(seed, 500, SEQUENCE_LENGTH, 0.5)
    
    print(melody)