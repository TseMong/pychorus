from madmom.features.drums import *
# madmom install from http://ifs.tuwien.ac.at/~vogl/dafx2018/     ## 2 models
# avg time use: 10min

test_file = './test_data/7016317_1.mp3'
processor = CRNNDrumProcessor()
act = processor(test_file)
proc = DrumPeakPickingProcessor(fps=100) # NOTE: need a true fps 
                                         # fps : float, optional
                                         # Frames per second used for conversion of timings.
print(proc(act))
# output -> test_data/test_drums_output.txt