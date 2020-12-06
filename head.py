import traktor
import algorythm
#import wmv
##

x = input('Analize sound = a | Train = t | Predict = p')

TEST = 0
if TEST == 1:
    print('testing')
    #wmv.genwmv()
else:
    if x == 'a':
        ## launch spectrograph
        traktor.Spectroanalize.spectro()
        ## launch ANN feature extraction
        traktor.Spectroanalize.databalace()
    if x == 't':
        # create spectro dataset
        algorythm.dataextractor.gengraphs()
        # drop spectograph data to csv for neural net
        algorythm.dataextractor.gencsv()
        # data preprocessing & training
        algorythm.dataextractor.preprocess()
    if x == 'p':
        #evaluate the model and predict
        algorythm.dataextractor.modeleval()


