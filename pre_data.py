from getRoI import getRoi
import cv2
import os
from pathlib import Path

names = ['salmon5_tests']
for name in names:
    loadPath = str( str(Path(__file__).parent) + '/frames_salmones/' + name)
    savepath = str( str(Path(__file__).parent) + '/rois/'+name)

    Bin = '/binario/'
    Original = '/original/'

    #Crear carpeta
    try:
        os.mkdir(savepath)
    except OSError as error:
        print(error)

    archivos = os.listdir(str(loadPath+Bin))

    for arch in archivos:
        frameBin = cv2.imread(str(loadPath+Bin+arch))
        frameOriginal = cv2.imread(str(loadPath+Original+arch))
        frameBin = cv2.cvtColor(frameBin,cv2.COLOR_BGR2GRAY)
        try:
            result = getRoi(frameOriginal, frameBin, debug=False)
        except OSError as error:
            print(error)
            print(str(loadPath+Bin+arch))
            continue
        try:
            cv2.imwrite( os.path.join(savepath , str(savepath+'/'+arch)),result )
        except OSError as error:
            print(error)
        #cv2.imshow('result', result)
        #cv2.waitKey(0)
