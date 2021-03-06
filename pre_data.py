from getRoI import getRoi
import cv2
import os
from pathlib import Path

names = ['salmon1_tests'] #['salmon1_tests','salmon2_tests','salmon3_tests','salmon4_tests','salmon5_tests','salmon6_tests','salmon7_tests','salmon8_tests','salmon9_tests','salmon10_tests','salmon11_tests']
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
