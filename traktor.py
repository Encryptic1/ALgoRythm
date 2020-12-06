import librosa
import librosa.display
import matplotlib.pyplot as plt
from playsound import playsound
import numpy as np
import sklearn
import pandas as pd
##
audio_data = 'H:/1217145478/CloZee_Koto.wav'
y , sr = librosa.load(audio_data)
class Spectroanalize():
    def spectro():
        print('*-- Spectrograph Init --*')
        #<class 'numpy.ndarray'> <class 'int'>print(x.shape, sr)#(94316,) 22050
        plt.figure(figsize=(12, 8))
        D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
        amplitude_to_db(np.abs(S))
        plt.subplot(4, 2, 1)
        librosa.display.specshow(D, y_axis='linear')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Linear-frequency power spectrogram')

        # Or on a logarithmic scale

        plt.subplot(4, 2, 2)
        librosa.display.specshow(D, y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Log-frequency power spectrogram')

        # Or use a CQT scale

        CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=sr), ref=np.max)
        plt.subplot(4, 2, 3)
        librosa.display.specshow(CQT, y_axis='cqt_note')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Constant-Q power spectrogram (note)')

        plt.subplot(4, 2, 4)
        librosa.display.specshow(CQT, y_axis='cqt_hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Constant-Q power spectrogram (Hz)')

        # Draw a chromagram with pitch classes

        C = librosa.feature.chroma_cqt(y=y, sr=sr)
        plt.subplot(4, 2, 5)
        librosa.display.specshow(C, y_axis='chroma')
        plt.colorbar()
        plt.title('Chromagram')

        # Force a grayscale colormap (white -> black)

        plt.subplot(4, 2, 6)
        librosa.display.specshow(D, cmap='gray_r', y_axis='linear')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Linear power spectrogram (grayscale)')

        # Draw time markers automatically

        plt.subplot(4, 2, 7)
        librosa.display.specshow(D, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Log power spectrogram')

        # Draw a tempogram with BPM markers

        plt.subplot(4, 2, 8)
        Tgram = librosa.feature.tempogram(y=y, sr=sr)
        librosa.display.specshow(Tgram, x_axis='time', y_axis='tempo')
        plt.colorbar()
        plt.title('Tempogram')
        plt.tight_layout()

        # Draw beat-synchronous chroma in natural time

        plt.figure()
        tempo, beat_f = librosa.beat.beat_track(y=y, sr=sr, trim=False)
        beat_f = librosa.util.fix_frames(beat_f, x_max=C.shape[1])
        Csync = librosa.util.sync(C, beat_f, aggregate=np.median)
        beat_t = librosa.frames_to_time(beat_f, sr=sr)
        ax1 = plt.subplot(2,1,1)
        librosa.display.specshow(C, y_axis='chroma', x_axis='time')
        plt.title('Chroma (linear time)')
        ax2 = plt.subplot(2,1,2, sharex=ax1)
        librosa.display.specshow(Csync, y_axis='chroma', x_axis='time',
                                x_coords=beat_t)
        plt.title('Chroma (beat time)')
        plt.tight_layout()
        plt.show()
#imp = input('Checkpoint reached....')
    def databalace():
        #### deep spec
        print('Databalance & Feature extract')
        spectral_centroids = librosa.feature.spectral_centroid(y, sr=sr)[0]
        spectral_centroids.shape
        (775,)
        # Computing the time variable for visualization
        frames = range(len(spectral_centroids))
        #plt.figure(str(frames),figsize=(12, 4))
        t = librosa.frames_to_time(frames)
        #plt.show()
        # Normalising the spectral centroid for visualisation
        def normalize(y, axis=0):
            return sklearn.preprocessing.minmax_scale(y, axis=axis)
        #Plotting the Spectral Centroid along the waveform
        plt.figure(figsize=(16, 9))
        ##cpy

        ##
        plt.subplot(6, 1, 1)
        librosa.display.waveplot(y, sr=sr, alpha=0.4)
        plt.plot(t, normalize(spectral_centroids), color='b')
        spectral_rolloff = librosa.feature.spectral_rolloff(y+0.01, sr=sr)[0]
        plt.plot(t, normalize(spectral_rolloff), color='r')
        plt.legend(('Centeroids', 'Rolloff'))
        plt.title('Spectral Center and Rolloff')
        #plt.show()
        #spectral Analysis
        spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(y+0.01, sr=sr)[0]
        spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(y+0.01, sr=sr, p=3)[0]
        spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(y+0.01, sr=sr, p=4)[0]
        #plt.figure(figsize=(19, 6))
        plt.subplot(6, 1, 2)
        librosa.display.waveplot(y, sr=sr, alpha=0.4)
        plt.plot(t, normalize(spectral_bandwidth_2), color='r')
        plt.plot(t, normalize(spectral_bandwidth_3), color='g')
        plt.plot(t, normalize(spectral_bandwidth_4), color='y')
        plt.legend(('p = 2', 'p = 3', 'p = 4'))
        plt.title('Deep Spectral Bandwidth partitioning')
        #plt.show()
        ## deeper dive on percussive sounds
        print('Deep Percussive Analysis...')
        #Plot the signal:
        #plt.figure(figsize=(19, 6))
        plt.subplot(5, 1, 3)
        librosa.display.waveplot(y, sr=sr)
        plt.title('Raw signal')
        #plt.show()
        #plt.figure(figsize=(19, 6))
        zero_crossings = librosa.zero_crossings(y, pad=False)
        print(sum(zero_crossings))#16
        mfccs = librosa.feature.mfcc(y, sr=sr)
        print(mfccs.shape)
        (20, 97)
        #Displaying  the MFCCs:
        hop_length = 512
        #plt.figure(figsize=(19, 6))
        plt.subplot(5, 1, 4)
        librosa.display.specshow(mfccs, sr=sr, x_axis='time')
        plt.title('MFCCs')
        #plt.show()
        chromagram = librosa.feature.chroma_stft(y, sr=sr, hop_length=hop_length)
        #plt.figure(figsize=(19, 6))
        plt.subplot(5, 1, 5)
        librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')
        plt.title('Chromagram')
        plt.tight_layout()
        plt.show()