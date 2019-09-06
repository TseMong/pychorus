from os import path
from pydub import AudioSegment

# files                                                                         
src = "./7016317_1.mp3"
dst = "./7016317_1.wav"

# convert wav to mp3                                                            
sound = AudioSegment.from_mp3(src)
sound.export(dst, format="wav")