import os
import music21 as m21

"""
This script preprocesses the dataset of folk songs in kern format.
First, load the dataset of folk songs in kern format.
Second, convert the songs from kern format to music21 score object.
Third, filter out songs that have non-acceptable durations.
Fourth, transpose songs to a common key.
"""

"""
music21 is a Python library for computer-aided musicology.
It's a package that enable to manipulate symbolic music data, 
and enable load and convert music symbolics files to other formats
(kern, MIDI, MusicXML, Humdrum, ABC, MuseData, Musedata, Lilypond, ... --> ker, MIDI, ...)
Enable analysis of music data and generation of music data. 
Usefull for representation of music data
"""

KERN_DATASET_PATH = "G:\\My Drive\\Cursos\\Valerio_Velardo\\Melody_generation_with_RNN-LSTM\\dataset_preprocessing\\deutschl\\test"
ACCEPTABLE_DURATIONS = [
    0.25, 
    0.5, 
    0.75, 
    1.0, 
    1.5, 
    2, 
    3, 
    4
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
    # get key drom de the song
    
    parts = song.getElementsByClass(m21.stream.Part)    # get all the parts of the song (list)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure) # get all the measures of the first part
    key = measures_part0[0][4] # get the key signature of the first measure of the first part
    
    # estimate key using music21 (in case key is not available in the kern file)
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")
    
    print(f"Original key: {key}")
    
    # get interval for transposition. E.g., Bmaj -> Cmaj is a semitone up, so interval is 1
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))
        
    # transpose song by calculated interval
    transposed_song = song.transpose(interval)
    
    return transposed_song
    
def preprocess(dataset_path):
    pass

    # load the folk song
    print("Loading songs...")
    songs = load_song_in_kern(KERN_DATASET_PATH)
    print(f"Loaded {len(songs)} songs.")
    
    for song in songs:
        
        # filter out songs that have non-aceptable duration
        if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
            continue
        
        # transpose song to cmaj/Amin
        song = transpose(song)
        
        # encode songs with music time series representation
        
        # save songs to text file
    
if __name__==  "__main__":
    
    songs = load_song_in_kern(KERN_DATASET_PATH)
    print(f"Loaded {len(songs)} songs.")
    song = songs[0]
    
    print(f"Has acceptable duration? {has_acceptable_durations(song, ACCEPTABLE_DURATIONS)}")
    
    transposed_song = transpose(song)
    
    song.show() # .show is a method of music21 that shows the music score
    transposed_song.show()
    
    