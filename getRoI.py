import cv2 as cv
import numpy as np

def getRoi(img, img_mask, debug=False):

	# --- Obtener contorno ---
	border, hierarchy = cv.findContours(img_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
	border = border[0]		# solo el contorno exterior (?)
	height, width = img_mask.shape

	# Obtener la orientacion de la img
	rotRect = cv.minAreaRect(border)
	angle = rotRect[-1]
	if rotRect[1][0] < rotRect[1][1]:
		angle = angle - 90

	# Rotar la img segun angulo
	center = (int(width/2), int(height/2))
	bg_color = tuple(np.int0(np.average(img, axis=(0,1))))
	rotMtx = cv.getRotationMatrix2D(center, angle, 1.0)
	nwidth = int( height*np.abs(rotMtx[0,1]) + width*np.abs(rotMtx[0,0]) )
	nheight = int( height*np.abs(rotMtx[0,0]) + width*np.abs(rotMtx[0,1]) )
	rotMtx[0,2] += nwidth/2 - center[0]
	rotMtx[1,2] += nheight/2 - center[1]
	result = cv.warpAffine(img, rotMtx, (nwidth, nheight), flags=cv.INTER_LINEAR, borderValue=(int(bg_color[0]), int(bg_color[1]), int(bg_color[2])))
	result_mask = cv.warpAffine(img_mask, rotMtx, (nwidth, nheight), flags=cv.INTER_LINEAR)
	th, result_mask = cv.threshold(result_mask, 128, 255, cv.THRESH_BINARY)		# interpolacion genera 'grises'

	# --- Obtener nueva region ya rotada ---
	new_border, hierarchy = cv.findContours(result_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
	new_border = new_border[0]
	rect = cv.minAreaRect(new_border)

	# Coordenadas de la nueva region (x,y)
	box = cv.boxPoints(rect)
	p_ul = [ min(np.int0(box)[:,0]), min(np.int0(box)[:,1])]		# upper-left
	p_br = [ max(np.int0(box)[:,0]), max(np.int0(box)[:,1])]		# bottom-right

	# cv.drawContours(img, [np.int0(cv.boxPoints(rotRect))], 0, (0,0,255),2)
	# cv.imshow('asdasd',img)
	# cv.waitKey(0)

	first_roi = result[p_ul[1]:p_br[1], p_ul[0]:p_br[0]]
	first_roi_gray = cv.cvtColor(first_roi, cv.COLOR_BGR2GRAY)
	first_roi_mask = result_mask[p_ul[1]:p_br[1], p_ul[0]:p_br[0]]
	fish_length = first_roi.shape[1]

	# Determinar si el pez mira a la derecha o izquierda
	mid = int(fish_length/2)
	left_area = cv.countNonZero(first_roi_mask[:,0:mid])
	right_area = cv.countNonZero(first_roi_mask[:,mid+1:])
	# Orientacion: True = mira izq. , False = mira der.
	#print(left_area, right_area)
	look_left = True if left_area >= right_area else False

	# --- Upper fin calculation ---
	# sliding window
	for i in np.arange(0.2, 0.5, 0.01):
		if look_left:
			x1 = int(i*fish_length)
			x2 = int((i+0.2)*fish_length)
		else:
			x1 = int((1-i-0.2)*fish_length)
			x2 = int((1-i)*fish_length)
		fin_roi = first_roi_mask[0:int((p_br[1]-p_ul[1])/2),x1:x2]
		try:
			if look_left and (np.where(fin_roi[:,-1] == 255)[0][0] < 1):
				break
			elif not look_left and (np.where(fin_roi[:,0] == 255)[0][0] < 1):
				break
		except:
			break


	m1 = 0; b1 = 0; m2 = 0; b2 = 0
	fin_roi_w = x2 - x1;	fin_roi_w_half = int((x2-x1)/2)
	#cv.imshow('asd',fin_roi)
	# Recta 1
	m1 = (np.where(fin_roi[:,fin_roi_w_half-1] == 255)[0][0] - np.where(fin_roi[:,0] == 255)[0][0])/fin_roi_w_half
	b1 = np.where(fin_roi[:,0] == 255)[0][0] + 1 - m1
	# Recta 2
	m2 = (np.where(fin_roi[:,fin_roi_w-1] == 255)[0][0] - np.where(fin_roi[:,fin_roi_w_half] == 255)[0][0])/fin_roi_w_half
	b2 = np.where(fin_roi[:,fin_roi_w-1] == 255)[0][0] + 1 - m2*fin_roi_w
	#print("R1: y = %.3fx %s %.3f , R2: y = %.3fx %s %.3f" % (m1,'+' if b1 > 0 else '-',np.abs(b1),m2,'+' if b2 > 0 else '-',np.abs(b2)))

	M = np.array([[m1, -1], [m2, -1]])
	B = np.array([-b1, -b2])
	x_sol = np.linalg.solve(M,B)
	UP = (x1 + int(x_sol[0]) - 1, int(x_sol[1]) - 1)
	#print('UP:',UP)

	# --- Eye coordinates calculation ---
	c = 0.20#0.12		# 0.07 perilla
	th = 85#50
	if look_left:	# pez mirando a derecha, toma primeros pixeles para buscar ojo
		a = 0
		b = int(c*fish_length)
	else:
		a = int((1-c)*fish_length)
		b = fish_length

	eye_roi = first_roi_gray[:, a:b]
	th, eye = cv.threshold(eye_roi, th, 255, cv.THRESH_BINARY_INV)			# 60 perilla
	#cv.imshow('as',eye)
	#cv.waitKey(0)
	# Calculo centroide
	# eye_x, eye_y = 0, 0
	# n_px = 0
	# for y in range(0, eye.shape[0]):
	# 	for x in range(0, eye.shape[1]):
	# 		if eye.item(y,x) > 0:
	# 			eye_x += (x + a)
	# 			eye_y += y
	# 			n_px += 1
	# EP = ( int(eye_x/n_px), int(eye_y/n_px) )
	moments = cv.moments(eye)
	if moments['m00']==0:
		final_roi = first_roi
		return final_roi
	cx = int(moments['m10']/moments['m00']) + a
	cy = int(moments['m01']/moments['m00'])
	EP = (cx, cy)
	#print('EP:', EP)

	# --- Belly point ---
	bp_y = np.where(first_roi_mask[:,UP[0]] == 255)[0][-1]
	BP = ( UP[0], int(0.65*(bp_y - UP[1])) + UP[1])
	#print('BP:',BP)

	# --- Final RoI ---
	final_roi = first_roi.copy()
	final_roi = final_roi[UP[1]:BP[1], min(EP[0],UP[0]):max(EP[0],UP[0])]
	final_roi = cv.resize(final_roi, (200,100), interpolation=cv.INTER_CUBIC)

	# --- Visual debug ---
	if debug:
		#box = [p_ul, p_ur, p_br, p_bl]
		box = cv.boxPoints(rect)
		box = np.int64(box)

		box2 = [ [p_ul[0]+EP[0],p_ul[1]+UP[1]], [p_ul[0]+EP[0],p_ul[1]+BP[1]], [p_ul[0]+BP[0],p_ul[1]+BP[1]], [p_ul[0]+UP[0],p_ul[1]+UP[1]] ]
		box2 = np.int64(box2)
		#print(center,angle)
		#print(rotMtx)
		#print(box)
		#cv.drawContours(result, new_border, -1, (0,0,255), 1)
		cv.drawContours(result, [box], 0, (0,0,255), 2)
		cv.drawContours(result, [box2], 0, (255,0,0), 2)
		result = cv.circle(result, (p_ul[0]+EP[0], p_ul[1]+EP[1]), radius=3, color=(0,255,0), thickness=-1)
		result = cv.circle(result, (p_ul[0]+UP[0], p_ul[1]+UP[1]), radius=3, color=(0,255,0), thickness=-1)
		result = cv.circle(result, (p_ul[0]+BP[0], p_ul[1]+BP[1]), radius=3, color=(0,255,0), thickness=-1)

		cv.imshow('debug', result)

	return final_roi#first_roi#final_roi

def main():
	salmon = 'frames_salmones/salmon8'
	scene = '00071'
	img = cv.imread(salmon+'/original/scene'+scene+'.png')

	img_mask = cv.imread(salmon+'/binario/scene'+scene+'.png')
	img_mask = cv.cvtColor(img_mask, cv.COLOR_BGR2GRAY)		# para dejarlo de 1 canal
	r, img_mask = cv.threshold(img_mask, 127, 255, cv.THRESH_BINARY)

	result = getRoi(img, img_mask, debug=False)

	cv.imshow('result', result)
	cv.waitKey(0)


if __name__ == '__main__':
	main()
