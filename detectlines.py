import cv2
import PIL
from PIL import Image
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
areaa =[]
area5=[]
global crop_characters
crop_characters = []
crop_characters1 = []
global w1,h1,w2

def sort_contours(cnts,reverse = False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts
def is_plate(c):
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	# the contour is 'bad' if it is not a rectangle
	return  len(approx) == 4  

def main():
	xmin,ymin,xmax,ymax=236,291,450,361
	path ='/home/dell/flaskprojects/9.jpg'		
	#path ='/home/dell/Downloads/Data-Images/Plates/37.jpg'
	image = cv2.imread(path)
	
	img_lp = cv2.imread(path)
	print(image.shape)
	
	mask = np.ones(img_lp.shape[:2], dtype="uint8") * 255
	#img_lp = cv2.resize(img, (150, 75))
	h1,w1,c =img_lp.shape
	print(w1,h1,c)
	area1 = w1*h1
	if (h1/w1) < 2.5:
		cv2.imshow("imm" , image[int((h1/2)-15):int(h1),:])
		image1 = image[0:int(h1/2),:]
		image2= image[int((h1/2)-15):int(h1),:]
		img_gray_lp1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
		img_gray_lp2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
		img_gray_lp1 = cv2.medianBlur(img_gray_lp1, 5)
		img_gray_lp2 = cv2.medianBlur(img_gray_lp2, 5)
		_, img_binary_lp = cv2.threshold(img_gray_lp1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		#_, img_binary_lp = cv2.threshold(img_gray_lp1,127,255,cv2.THRESH_TOZERO_INV)
		#img_binary_lp = cv2.adaptiveThreshold(img_gray_lp1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
		img_binary_lp1 = cv2.erode(img_binary_lp, (3, 3))
		img_binary_lp1 = cv2.dilate(img_binary_lp1, (3,3))
		bin_img1 = cv2.bitwise_not(img_binary_lp1)
		#cv2.imshow("img1",bin_img)
		bin_img1 = cv2.medianBlur(bin_img1, 3) 
		#bin_img = cv2.GaussianBlur(bin_img,(3,3),0)
		#print(bin_img)
		cv2.imshow("imgg",bin_img1)
		#cont, _  = cv2.findContours(bin_img, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		cont, _  = cv2.findContours(bin_img1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		#############################################################################
		#img_binary_lp = cv2.adaptiveThreshold(img_gray_lp2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
		_, img_binary_lp = cv2.threshold(img_gray_lp2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		img_binary_lp2 = cv2.erode(img_binary_lp, (3, 3))
		img_binary_lp2 = cv2.dilate(img_binary_lp2, (3,3))
		bin_img2 = cv2.bitwise_not(img_binary_lp2)
		#cv2.imshow("img1",bin_img)
		bin_img2 = cv2.medianBlur(bin_img2, 3) 
		#bin_img = cv2.GaussianBlur(bin_img,(3,3),0)
		#print(bin_img)
		cv2.imshow("img",bin_img2)
		#cont, _  = cv2.findContours(bin_img, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		contr, _  = cv2.findContours(bin_img2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
###################################################################################################33	
		contours = sorted(cont, key = cv2.contourArea, reverse = True)[:9]
		contours1 = sorted(contr, key = cv2.contourArea, reverse = True)[:6]
###################################################################################################
		contour = []
		cn=[]
		area5 =[]
		for c in contours:
		# if the contour is bad, draw it on the mask
			if is_plate(c):
				pass
			else:
				contour.append(c)

		w2=0
		for i,c in enumerate(contour):
			
			
			#if cv2.isContourConvex(c) == True:
			#print()
			
			x, y, w, h = cv2.boundingRect(c)
			
			#print(area,x, y, w, h)
			if w>(.30*(w1)):
				pass
	
			else:
				
				w2= w2+w
				cn.append(c)
				
		w2 = w2/len(contours)
		#print(w2)
	
		for c in cn:
			x, y, w, h = cv2.boundingRect(c)
			for c in cn:
				x1, y1, w4, h1 = cv2.boundingRect(c)
				if x1>x and x1+w4 < x+y:
					pass
				else:
					cv2.drawContours(mask, [c], -1, 0, -1)
					area = cv2.contourArea(c)
					areaa.append(area)
				
			
		#print(img_gray_lp.shape)
				
		#cv2.drawContours(image,contours,-1,(255,0,255),4)
		#cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
		#cv2.imshow('Contours',image)
		digit_w, digit_h = 30, 60
	
		
		for c in sort_contours(cont):
			#print(c)
			areaaa= cv2.contourArea(c)
			if areaaa  in areaa:
	
					cv2.drawContours(mask, [c], -1, 0, -1)
					
					x, y, w, h = cv2.boundingRect(c)
					if (w < (.30*(w1))) or (w>(.70*(w2))):
	
						cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
						curr_num = img_gray_lp1[y:y+h,x:x+w]
						#curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
						#cv2.imshow('Contour',image)
						#curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
						curr_num = cv2.bitwise_not(curr_num)
						#cv2.imshow("Afte", curr_num)
						crop_characters.append(curr_num)
						print(len(crop_characters))
		
####################################################################################################################################3		
		contur = []
		cnn=[]
		for c in contours1:
		# if the contour is bad, draw it on the mask
			if is_plate(c):
				pass
			else:
				contur.append(c)

		w3=0
		for i,c in enumerate(contur):
			
			
			#if cv2.isContourConvex(c) == True:
			#print()
			
			x, y, w, h = cv2.boundingRect(c)
			
			#print(area,x, y, w, h)
			if w>(.30*(w1)):
				pass
	
			else:
				
				w3= w3+w
				cnn.append(c)
				
		w3 = w3/len(contours1)
		#print(w2)
	
		for c in cnn:
			x, y, w, h = cv2.boundingRect(c)
			for c in cnn:
				x1, y1, w5, h6 = cv2.boundingRect(c)
				if x1>x and x1+w5 < x+y:
					pass
				else:
					cv2.drawContours(mask, [c], -1, 0, -1)
					area = cv2.contourArea(c)
					area5.append(area)
				
			
		#print(img_gray_lp.shape)
				
		cv2.drawContours(img_gray_lp2,contours,-1,(255,0,255),4)
		#cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
		cv2.imshow('Contourss',img_gray_lp2)
		digit_w, digit_h = 30, 60
	
		
		for c in sort_contours(contr):
			#print(c)
			areaaa= cv2.contourArea(c)
			if areaaa  in area5:
	
					cv2.drawContours(mask, [c], -1, 0, -1)
				
					x, y, w, h = cv2.boundingRect(c)
					if (w < (.30*(w1))) or (w>(.70*(w2))):
						
						#cv2.rectangle(image,(x,(y+15+(h1/2)),(x+w,(y+h+15+(h1/2)),(0,255,0),2)
						curr_num = img_gray_lp2[y:y+h,x:x+w]
						#curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
						#cv2.imshow('Contourrrr',image)
					#curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
						curr_num = cv2.bitwise_not(curr_num)
				#cv2.imshow("Afte", curr_num)
						crop_characters.append(curr_num)
		'''for i in crop_characters1:
			crop_character.append(i)'''
		image = cv2.bitwise_and(image, image, mask=mask)
		print("Detect {} letters...".format(len(crop_characters)))
		cv2.imshow("After", image)
		
		
	

	
main()
fig = plt.figure(figsize=(14,4))
grid = gridspec.GridSpec(ncols=len(crop_characters),nrows=1,figure=fig)

for i in range(len(crop_characters)):
    fig.add_subplot(grid[i])
    plt.axis(False)
    plt.imshow(crop_characters[i],cmap="gray")
cv2.waitKey(0) 
cv2.destroyAllWindows()
