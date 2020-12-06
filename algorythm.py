import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import pathlib
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import keras
from keras import layers
from keras.models import Sequential, save_model, load_model
import warnings
warnings.filterwarnings('ignore')
import time
import pydot
import graphviz
###
genres = ['bass' ,'chillstep' ,'country', 'dubstep','electrofunk','funk','rap','tronhop']
class dataextractor():
    def gengraphs():
        print('cleaning file names')
        for g in genres:
            for filename in os.listdir(f'./drive/genres/{g}'):
                newname = filename.replace(" - ","_").replace(" ","_").replace("’","").replace("'",'').replace(",","")
                print(filename)
                print(newname)
                #wait = input('checkpoint..')
                os.rename(f'./drive/genres/{g}/'+ filename,f'./drive/genres/{g}/'+ newname)
        print('generating spectrographs')
        # grab raw audio and gen spectrograph
        cmap = plt.get_cmap('inferno')
        plt.figure(figsize=(16,9))
        for g in genres:
            pathlib.Path(f'img_data/{g}').mkdir(parents=True, exist_ok=True)
            for filename in os.listdir(f'./drive/genres/{g}'):
                songname = f'./drive/genres/{g}/{filename}'
                y, sr = librosa.load(songname, mono=True, duration=5)
                plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB')
                plt.axis('off')
                plt.savefig(f'img_data/{g}/{filename[:-3].replace(".", "")}.png')
                plt.clf()
        print('*-- Gen spectro done --*')

    def gencsv():
        print('converting to csv')
        #Gen csv of spectro data
        header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
        for i in range(1, 21):
            header += f' mfcc{i}'
        header += ' label'
        header = header.split()
        file = open('dataset.csv', 'w', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(header)
        for g in genres:
            for filename in os.listdir(f'./drive/genres/{g}'):
                songname = f'./drive/genres/{g}/{filename}'
                y, sr = librosa.load(songname, mono=True, duration=30)
                rmse = librosa.feature.rms(y=y)
                chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
                spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
                spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                zcr = librosa.feature.zero_crossing_rate(y)
                mfcc = librosa.feature.mfcc(y=y, sr=sr)
                to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
                for e in mfcc:
                    to_append += f' {np.mean(e)}'
                to_append += f' {g}'
                file = open('dataset.csv', 'a', newline='')
                with file:
                    writer = csv.writer(file)
                    writer.writerow(to_append.split())
        print('*-- CSV output done --*')
    
    def preprocess():
        print('\n', 'Preprocessing phase started')
        data = pd.read_csv('dataset.csv')
        data.head()# Dropping unneccesary columns
        data = data.drop(['filename'],axis=1)#Encoding the Labels
        genre_list = data.iloc[:, -1].astype(str)
        encoder = LabelEncoder()
        y = encoder.fit_transform(genre_list)#Scaling the Feature columns
        scaler = StandardScaler()
        X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))#Dividing data into training and Testing set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        print('\n', '*-- preprocess done --*')
        print('\n', "Incubating ALgoRythm 0 State")
        model = Sequential()
        model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print('\n', 'Initiating neuron ingition')
        time.sleep(1)
        print('\n', '.....')
        time.sleep(1)
        time.sleep(1)
        print('\n', 'Intelegence waking at 0 State')
        classifier = model.fit(X_train, y_train, epochs=100, batch_size=128)
        # Save the model
        modelpath = './saved_model'
        save_model(model, modelpath)
        print('\n', 'ALgoRythm: "boy howdy Lets rock and roll"')
    
    def modeleval():
        ##redifine for non img
        print('\n', 'evaluating...')
        filepath = './saved_model'
        model = load_model(filepath, compile=True)
        #save model graph
        keras.utils.plot_model(
                model,
                to_file="dot_model.png",
                show_shapes=False,
                show_layer_names=True,
                rankdir="TB",
                expand_nested=False,
                dpi=96,
                )
        print('model loaded')
        print('layer1: ',model.layers[0].output)
        print('layer2: ',model.layers[1].output)
        print('layer3: ',model.layers[2].output)
        print('layer4: ',model.layers[3].output)
        
        #make eval csv
        header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
        for i in range(1, 21):
            header += f' mfcc{i}'
        header += ' label'
        header = header.split()
        file = open('evalset.csv', 'w', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(header)
        # convert samples
        use_samples = []
        samples_to_predict = []
        print('\n', 'cleaning file names')
        for filename in os.listdir(f'./test/'):
            newname = filename.replace(" - ","_").replace(" ","_").replace("’","").replace("'",'').replace(",","")
            print(filename)
            print(newname)
            #wait = input('checkpoint..')
            os.rename(f'./test/'+ filename,f'./test/'+ newname)
        print('\n', 'Loading test files')
        for filename in os.listdir(f'./test/'):
            songname = f'./test/{filename}'
            use_samples.append(filename)
            y, sr = librosa.load(songname, mono=True, duration=30)
            rmse = librosa.feature.rms(y=y)
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
            for e in mfcc:
                to_append += f' {np.mean(e)}'
            #to_append += f' {g}'
            file = open('evalset.csv', 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())
                samples_to_predict.append(to_append.split())
        print('*-- CSV output done --*')
        #load samples
        print('Eval phase started')
        data = pd.read_csv('evalset.csv')
        data.head()# Dropping unneccesary columns
        data = data.drop(['filename'],axis=1)#Encoding the Labels
        print(data)
        #what = input('checkpoint...')
        fortransform = np.array(data.iloc[:, :-1], dtype = float)
        print('fortransform: ',fortransform)
        scaler = StandardScaler()
        X = scaler.fit_transform(fortransform)#Dividing data into training and Testing set
        gen = model.predict(X)
        print('***___--- Raw Tensors ---___***')
        print(gen)
        print('***___--- Raw Tensors ---___***')
        # Generate arg maxes for predictions
        classes = gen.argmax(axis = 1)
        print('Prediction Class: ',classes)
        uselen = len(use_samples)
        i = 0
        for song in use_samples:
            predicted_genre = sorted(genres)[classes[i]]
            print('Final Predicted Genere: for '+ song +' = ',predicted_genre)
            i += 1
        # Generate plots for samples

        