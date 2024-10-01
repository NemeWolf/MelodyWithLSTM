import os
import music21 as m21
import json
"""
This script preprocesses the dataset of folk songs in kern format.
First, load the dataset of folk songs in kern format.
Second, convert the songs from kern format to music21 score object.
Third, filter out songs that have non-acceptable durations.
Fourth, transpose songs to a common key.
Fifth, encode songs with music time series representation.
Sixth, save the encoded songs to text file.
Seventh, create a dataset that contains all the encoded songs in a single file.
"""

"""
music21 is a Python library for computer-aided musicology.
It's a package that enable to manipulate symbolic music data, 
and enable load and convert music symbolics files to other formats
(kern, MIDI, MusicXML, Humdrum, ABC, MuseData, Musedata, Lilypond, ... --> ker, MIDI, ...)
Enable analysis of music data and generation of music data. 
Usefull for representation of music data
"""

KERN_DATASET_PATH = "G:\\Mi unidad\\Cursos\\Valerio_Velardo\\Melody_generation_with_RNN-LSTM\\dataset_preprocessing\\deutschl\\test"
SAVE_DIR = "G:\\Mi unidad\\Cursos\\Valerio_Velardo\\Melody_generation_with_RNN-LSTM\\dataset"
SINGE_FILE_DATASET = "G:\\Mi unidad\\Cursos\\Valerio_Velardo\\Melody_generation_with_RNN-LSTM\\file_dataset"
MAPPINNG_PATH = "G:\\Mi unidad\\Cursos\\Valerio_Velardo\\Melody_generation_with_RNN-LSTM\\mapping.json"

SEQUENCE_LENGTH = 64 # Same amount that LSTM elements that sequences fixed length

# durations are expressed in quarter length
ACCEPTABLE_DURATIONS = [
    0.25, # 16th note
    0.5, # 8th note
    0.75,
    1.0, # quarter note
    1.5,
    2, # half note
    3,
    4 # whole note
]

def load_song_in_kern(dataset_path):
    """Loads all kern pieces in dataset using music21.

    :param dataset_path (str): Path to dataset
    :return songs (list of m21 streams): List containing all pieces
    """
    
    songs = []
    # go through all the fils in the dataset and load them with music21
    for path, subdirs, files in os.walk(dataset_path):
        for file in files:            
            if file[-3:]=="krn":  
                # Conviert the krn file to a music21 
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)
    return songs
           
def has_acceptable_durations(song, acceptable_durations):
    """Boolean routine that returns True if the song has only acceptable durations, False otherwise.
    
    param song: music21 stream object
    param acceptable_durations: list of acceptable duration in terms of number of quarter notes
    return: bool
    """
    
    # note.flatten().notesAndRests: flaten the song structure and get all the notes and rests in a list
    for note in song.flatten().notesAndRests:
        
        # note.duration.quarterLenght: get the duration of the note in terms of number of quarter notes
        if note.duration.quarterLength not in acceptable_durations:
            return False
            # we want simply the duration of the note in terms of number of quarter notes for the model        
    return True   
            
def transpose(song):
    """Transposes song to C maj/A min

    :param piece (m21 stream): Piece to transpose
    :return transposed_song (m21 stream):
    """
    
    # get key drom de the song
    
    parts = song.getElementsByClass(m21.stream.Part)    # get all the parts of the song (list)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure) # get all the measures of the first part
    key = measures_part0[0][4] # get the key signature of the first measure of the first part
    
    # estimate key using music21 (in case key is not available in the kern file)
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")
        
    # get interval for transposition. E.g., Bmaj -> Cmaj is a semitone up, so interval is 1
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))
        
    # transpose song by calculated interval
    transposed_song = song.transpose(interval)
    
    return transposed_song
    
def encode_song(song, time_step = 0.25):
    """Converts a score into a time-series-like music representation. Each item in the encoded list represents 'min_duration'
    quarter lengths. The symbols used at each step are: integers for MIDI notes, 'r' for representing a rest, and '_'
    for representing notes/rests that are carried over into a new time step. Here's a sample encoding:

        ["r", "_", "60", "_", "_", "_", "72" "_"]

    :param song (m21 stream): Piece to encode
    :param time_step (float): Duration of each time step in quarter length
    :return:
    """
    # p = 60, d = 1.0 -> [60, "_", "_", "_"]  
    
    encoded_song = []
    
    for event in song.flatten().notesAndRests:
        # handle notes
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi #60
        
        # handle rests
        elif isinstance(event, m21.note.Rest):
            symbol = "r"
                        
        # convert the note/rest into time series representation        
        steps =int(event.duration.quarterLength / time_step)

        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")
                
    # cast encoded song to a str
    encoded_song = " ".join(map(str, encoded_song))
    
    return encoded_song

def preprocess(dataset_path):
    
    # load the folk song
    print("Loading songs...")
    songs = load_song_in_kern(KERN_DATASET_PATH)
    print(f"Loaded {len(songs)} songs.")
    
    for i, song in enumerate(songs):
        
        # filter out songs that have non-aceptable duration
        if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
            continue
        
        # transpose song to cmaj/Amin
        song = transpose(song)
        
        # encode songs with music time series representation
        encoded_song = encode_song(song)
        
        # save songs to text file
        save_path = os.path.join(SAVE_DIR, str(i))
        
        with open(save_path, "w") as fp:
            fp.write(encoded_song)
    
def load(file_path):
    with open(file_path, "r") as fp:
        song = fp.read()        
    return song

def create_single_file_dataset(dataset_path,file_dataset_path, sequence_length):
    """ Create a file dataset from the dataset in the dataset_path
    param dataset_path: path to the dataset
    param file_dataset_path: path to save the file
    param sequence_length: number of time steps to be consider in each sequence
    return songs: string that contains all dataset
    """
    
    new_song_delimiter = "/ " * sequence_length
    songs = ""
    # load encoded songs and add delimeter
    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)
            songs = songs + song + " " + new_song_delimiter
    
    songs = songs[:-1]
    
    # save string that contains all datasets
    
    with open(file_dataset_path, "w") as fp:
        fp.write(songs)
        
    return songs
    
def create_mapping(songs, mapping_path):
    """Creates a json file that maps the symbols in the song dataset onto integers

    :param songs (str): String with all songs
    :param mapping_path (str): Path where to save mapping
    :return:
    """
    mappings = {}
    
    # identify the vocabulary
    songs = songs.split()
    vocabulary = list(set(songs))
    
    # create mappings (map all symbols to an integer)
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i   
    
    # save vocabulary to JSON file
    with open(mapping_path, "w") as fp:
        json.dump(mappings, fp, indent=4)
    
def main():
    preprocess(KERN_DATASET_PATH)
    songs = create_single_file_dataset(SAVE_DIR, SINGE_FILE_DATASET, SEQUENCE_LENGTH)
    create_mapping(songs, MAPPINNG_PATH)

if __name__==  "__main__":
    main()
    
    