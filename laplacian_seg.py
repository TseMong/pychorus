from __future__ import print_function

import numpy as np
import scipy
import matplotlib.pyplot as plt

import sklearn.cluster

import librosa
import librosa.display
import os

################################################################################
# load file
def load_file():
    file_path = os.path.join(os.path.abspath('.'), "data", "songs", "爱.mp2")
    y, sr = librosa.load(file_path)
    return y, sr

################################################################################
# compute and plot a log-power CQT
def compute_CQT(y, sr):
    BINS_PER_OCTAVE = 12 * 3
    N_OCTAVES = 7
    C = librosa.amplitude_to_db(np.abs(librosa.cqt(y=y, sr=sr,
                                            bins_per_octave=BINS_PER_OCTAVE,
                                            n_bins=N_OCTAVES * BINS_PER_OCTAVE)),
                                ref=np.max)

    ################################################################################
    # To reduce dimensionality, we’ll beat-synchronous the CQT
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    Csync = librosa.util.sync(C, beats, aggregate=np.median)

    # For plotting purposes, we'll need the timing of the beats
    # we fix_frames to include non-beat frames 0 and C.shape[1] (final frame)
    beat_times = librosa.frames_to_time(librosa.util.fix_frames(beats,
                                                                x_min=0,
                                                                x_max=C.shape[1]),
                                        sr=sr)

    ################################################################################
    # Let’s build a weighted recurrence matrix using beat-synchronous CQT (Equation 1) width=3 
    # prevents links within the same bar mode=’affinity’ here implements S_rep (after Eq. 8)
    R = librosa.segment.recurrence_matrix(Csync, width=3, mode='affinity',
                                        sym=True)

    ################################################################################
    # Enhance diagonals with a median filter (Equation 2)
    df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
    Rf = df(R, size=(1, 7))

    ################################################################################
    # Now let’s build the sequence matrix (S_loc) using mfcc-similarity
    # Rpath[i,i±1]=exp(−∥Ci−Ci±1∥2/σ2)
    # Here, we take σ to be the median distance between successive beats.
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    Msync = librosa.util.sync(mfcc, beats)

    path_distance = np.sum(np.diff(Msync, axis=1)**2, axis=0)
    sigma = np.median(path_distance)
    path_sim = np.exp(-path_distance / sigma)

    R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)

    ################################################################################
    # And compute the balanced combination (Equations 6, 7, 9)
    deg_path = np.sum(R_path, axis=1)
    deg_rec = np.sum(Rf, axis=1)

    mu = deg_path.dot(deg_path + deg_rec) / np.sum((deg_path + deg_rec)**2)

    A = mu * Rf + (1 - mu) * R_path

    ################################################################################
    # Now let’s compute the normalized Laplacian
    L = scipy.sparse.csgraph.laplacian(A, normed=True)


    # and its spectral decomposition
    evals, evecs = scipy.linalg.eigh(L)


    # We can clean this up further with a median filter.
    # This can help smooth over small discontinuities
    evecs = scipy.ndimage.median_filter(evecs, size=(9, 1))


    # cumulative normalization is needed for symmetric normalize laplacian eigenvectors
    Cnorm = np.cumsum(evecs**2, axis=1)**0.5

    # If we want k clusters, use the first k normalized eigenvectors.
    # Fun exercise: see how the segmentation changes as you vary k

    k = 5

    X = evecs[:, :k] / Cnorm[:, k-1:k]

    ################################################################################
    # Let’s use these k components to cluster beats into segments (Algorithm 1)
    KM = sklearn.cluster.KMeans(n_clusters=k)

    seg_ids = KM.fit_predict(X)


    ################################################################################
    # Locate segment boundaries from the label sequence
    bound_beats = 1 + np.flatnonzero(seg_ids[:-1] != seg_ids[1:])

    # Count beat 0 as a boundary
    bound_beats = librosa.util.fix_frames(bound_beats, x_min=0)

    # Compute the segment label for each boundary
    bound_segs = list(seg_ids[bound_beats])

    # Convert beat indices to frames
    bound_frames = beats[bound_beats]

    # Make sure we cover to the end of the track
    bound_frames = librosa.util.fix_frames(bound_frames,
                                        x_min=None,
                                        x_max=C.shape[1]-1)


    ################################################################################
    # plot the final segmentation over original CQT
    # sphinx_gallery_thumbnail_number = 5


    bound_times = librosa.frames_to_time(bound_frames)
    freqs = librosa.cqt_frequencies(n_bins=C.shape[0],
                                    fmin=librosa.note_to_hz('C1'),
                                    bins_per_octave=BINS_PER_OCTAVE)

    return bound_beats, bound_frames, bound_segs, bound_times


if __name__ == "__main__":
    y, sr = load_file()
    bound_beats, bound_frames, bound_segs, bound_times = compute_CQT(y, sr)
    np.set_printoptions(suppress=True)
    print(bound_times)