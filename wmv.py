from converter import Converter
import os
##
conv = Converter()
def genwmv():
    
    for filename in os.listdir(f'./temp/'):
        songname = f'./temp/{filename}'
        songnew = F'./test/' + filename.split('.')[0] + '.wmv'
        print(songnew)
        info = conv.probe(songname)
        print(info)
        waht = input('Checkpoint...')
        convert = conv.convert(songname, songnew, {
            'format': 'wmv',
            'audio': {
                'codec': 'aac',
                'samplerate': 11025,
                'channels': 2
            }})

        for timecode in convert:
            print(f'\rConverting ({timecode:.2f}) ...')