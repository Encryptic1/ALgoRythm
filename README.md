# ALgoRythm
Music Spectral Analyzer &amp; ANN genre classifier (full sequence)

Goal:
  1. to be able to input any song for full spectral analysis (complete)
  2. use the features extracted to train an ANN classifier for genres (complete)
  3. Train model on mass data.
    - Have yet to complete this you can test new songs on the model i have saved in the saved_model folder but it will not work for much as         the training set is small and relates only to the genres i created. I suggest you build a larger training set and your own genres             under./drive/genres/ to produce a more robust model
  
Install & Config:
---------------------------------------------------------------------------------------------------------------------
  1. Download or Clone
  ```
  git clone https://github.com/Encryptic1/ALgoRythm.git
  ```
  2. Install Requirements
  ```
  cd {main dir}
  pip install requirements.txt
  ```
  3. Get audio files (suggest spotdl)
  ```
  pip install spotdl
  ```
  4. Create your genres in the ./drive/genres/ folder. these folder names need to reflect in algorythm.py line 20. 
  These wil be the defining features used to train the network. be careful you select the genre accordingly.
  ```
  genres = ['bass' ,'chillstep' ,'country', 'dubstep','electrofunk','funk','rap','tronhop']
  ```
  5. Place any .mp3 files for predictions in the test folder ./test/
  6. Place an .mp3 file for the analizer in ./analize/
  7. Place your training  .mp3's from step 3 into their respective genre folders ./drive/genres/{x}

Usage:
---------------------------------------------------------------------------------------------------------------------
  ```
  python head.py
  ```
  Select:
    a for | Analizer to run specrography on file in ./analize/
    t for | Training the model from files in ./drive/genres/{*}
    p for | Predict the genres of the files in ./test/

Extra Notes:
---------------------------------------------------------------------------------------------------------------------
You may run into issues either training or predicting with the model. If the input shape error comes up its likely some weird symbol in the
.mp3 filename causing an improper load of the data. I added funtions to clean these names but some issues may persist.
You'll see some lines in algorythm.py like:
```
filename.replace(x)
```
Where x is ',-$ and some other symbols. this is where i would check first in the ./drive/genres/{*}.mp3 filenames
