import cv2 as cv
import numpy as np
from skimage.feature import hog
import sklearn.metrics as metrics
from os import listdir
import matplotlib.pyplot as plt
import seaborn as sns

orientations = 10
ppc = (8,8)					# pixels per cell
cell_size = (2,2)
normalized_size = (64,64)
roi_size = (180, 80)
off = 16		# 0 <= off <= 20

def show_confusion_matrix(cm, labels):
    plt.figure(figsize=(10, 9))
    sns.heatmap(cm, xticklabels=labels, yticklabels=labels,annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()

# calcula el HoG de la lista de imagenes de template
def getTemplateHoG(list_img):

	result_HoG = list()
	for img in list_img:
		img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
		roi = img_gray[off:off+roi_size[1], off:off+roi_size[0]]
		roi_resize = cv.resize(roi, normalized_size, interpolation=cv.INTER_CUBIC)
		histogram = hog(roi_resize, orientations, ppc, cell_size, visualize=False)
		result_HoG.append(histogram)
	
	result_HoG = np.array(result_HoG)
	return result_HoG

# evalua con una lista de template
def evaluateHoG(img, templateHoG):

	img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	step = 1
	result_mtx = list()

	for offset in range(0, 20, step):
		roi = img_gray[offset:offset+roi_size[1], offset:offset+roi_size[0]]
		roi_resize = cv.resize(roi, normalized_size, interpolation=cv.INTER_CUBIC)
		histogram = hog(roi_resize, orientations, ppc, cell_size, visualize=False)
		histogram = np.tile(histogram, [len(templateHoG), 1])
		norm_result = np.linalg.norm(histogram-templateHoG, ord=2, axis=1)
		min_val = np.amin(norm_result)
		idx_class = np.where(norm_result == min_val)
		result_mtx.append( [int(idx_class[0][0]), min_val])
	
	result_mtx = np.array(result_mtx)
	min_global = np.amin(result_mtx[:,1])
	idx_class = np.where(result_mtx[:,1] == min_global)
	idx_class = int(idx_class[0][0])

	return int(result_mtx[idx_class,0])

# evalua 1 imagen con dos listas de template y toma el minimo entre ambas
def evaluateHoG2(img, templateHoG1, templateHoG2):

	img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	step = 1
	result_mtx = list()

	# Escaneo de la RoI
	for offset in range(0, 20, step):
		roi = img_gray[offset:offset+roi_size[1], offset:offset+roi_size[0]]
		roi_resize = cv.resize(roi, normalized_size, interpolation=cv.INTER_CUBIC)

		histogram = hog(roi_resize, orientations, ppc, cell_size, visualize=False)
		histogram = np.tile(histogram, [len(templateHoG1), 1])		# replica el histograma para el mismo numero de clases, y realizar operaciones matriciales
		
		norm_result1 = np.linalg.norm(histogram-templateHoG1, ord=2, axis=1)
		norm_result2 = np.linalg.norm(histogram-templateHoG2, ord=2, axis=1)

		#norm_result = (norm_result1 + norm_result2)/2
		norm_result = np.minimum(norm_result1, norm_result2)

		min_val = np.amin(norm_result)							# obtiene el valor minimo
		idx_class = np.where(norm_result == min_val)			# obtiene la clase que genero el valor minimo, row == clase
		result_mtx.append( [int(idx_class[0][0]), min_val])		# agrega el par (clase, valor_min), esto lo hace para las 20/step iteraciones
	
	result_mtx = np.array(result_mtx)		# matriz resultante de (20/step) x 2

	# minimo global
	min_global = np.amin(result_mtx[:,1])
	idx_class = np.where(result_mtx[:,1] == min_global)
	idx_class = int(idx_class[0][0])

	return int(result_mtx[idx_class,0])

def main(debug=False):
	
	# Templates
	class_names = ['salmon1','salmon2','salmon3','salmon4','salmon5','salmon6','salmon7','salmon8','salmon9','salmon10','salmon11']

	list_img = list()	# lista imagenes template1
	list_img2 = list()	# lista imagenes template2

	if (debug):		# escoge imagenes manualmente
		file_names = ['00351','00051','00161','00061','00121','00031','00091','00046','00266','00086','00076']
		file_names2 = ['00401','00101','00141','00101','00016','00076','00076','00061','00226','00126','00176']
		for i, file in enumerate(file_names):
			list_img.append( cv.imread('rois_HoG/'+class_names[i]+'/scene'+file+'.png') )
			list_img2.append( cv.imread('rois_HoG/'+class_names[i]+'/scene'+file_names2[i]+'.png') )

	else:			# escoge imagenes desde la carpeta 'template'
		for i in class_names:
			files = listdir('rois_HoG/'+i+'/template')
			list_img.append( cv.imread('rois_HoG/'+i+'/template/'+files[0]) )
			list_img2.append( cv.imread('rois_HoG/'+i+'/template/'+files[1]) )

	# Calculo de los HoG de referencia
	HoG = getTemplateHoG(list_img)
	HoG2 = getTemplateHoG(list_img2)

	# Testing Set
	testing_set = list()	# groundtruth imagenes
	y = list()				# groundtruth clase

	for i in range(0, len(class_names)):
		files = listdir('rois_HoG/'+class_names[i])
		files.remove('template')
		
		if (debug):
			files.remove('scene'+file_names[i]+'.png')
			files.remove('scene'+file_names2[i]+'.png')

		for f in files:
			testing_set.append( cv.imread('rois_HoG/'+class_names[i]+'/'+f) )
			y.append(i)

	# Evaluate
	y_hat = list()		# predictions
	for img in testing_set:
		y_hat.append( evaluateHoG2(img, HoG, HoG2) )	# evaluar con 2 templates

	accuracy = metrics.accuracy_score(y, y_hat)

	print('Accuracy = %.3f%%' % (accuracy*100))
	print(metrics.classification_report(y, y_hat, digits=3))

	confusion_mtx = metrics.confusion_matrix(y, y_hat)
	show_confusion_matrix(confusion_mtx, class_names)

if __name__ == '__main__':
	main()