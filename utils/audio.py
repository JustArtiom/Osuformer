import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def waveform_to_spectrogram(waveform: np.ndarray, frame_length: int, frame_step: int) -> np.ndarray:
    """
    Convert a waveform into its spectrogram representation.

    Params:
        waveform (np.ndarray): The input waveform.
        frame_length (int): The length of each frame in samples.
        frame_step (int): The number of samples to step between frames.

    Returns:
        np.ndarray: The spectrogram representation of the input waveform.
    """
    spectrogram = tf.signal.stft(waveform, frame_length, frame_step)

    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    spectrogram = spectrogram.numpy()

    return spectrogram

def update_waveform_bpm(waveform: np.ndarray, from_bpm: int, to_bpm: int) -> np.ndarray:
    """
    Update a waveform's BPM by stretching it to match the target BPM.

    Params:
        waveform (np.ndarray): The waveform to be updated.
        from_bpm (int): The original BPM (beats per minute) of the waveform.
        to_bpm (int): The target BPM (beats per minute) to which the waveform will be stretched.

    Returns:
        np.ndarray: The updated waveform with the tempo adjusted to match the target BPM.
    """
    return librosa.effects.time_stretch(waveform, rate=to_bpm / from_bpm)