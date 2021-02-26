import cv2
import pytesseract
import sys 
from pyocr import pyocr 
import PIL
from PIL import Image
from PIL import Image
import sys
from pyocr import pyocr
from pyocr import builders
#from wand.image import Image
#from PIL import Image as PI
import pyocr
import pyocr.builders
import io
dir_name = "inference/output"
import boto3
#voc = []
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
import numpy as np
  

def main():
	xmin,ymin,xmax,ymax=236,291,450,361
	path ='/home/dell/flaskprojects/numberplate.jpg'		
	#path ='/home/dell/Downloads/Data-Images/Plates/37.jpg'
	image = cv2.imread(path)
	#img = cv2.imread(path)
	#image = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	print(image.shape)
	#cv2.imshow("2",image)
	#Cropped = image[int(ymin):int(ymax),int(xmin):int(xmax) ]
	#cv2.imshow("1",Cropped)
	###PYtesseract######
	filterSize =(3, 3) 
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT,filterSize) 
	input_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
  
	# Applying the Top-Hat operation 
	tophat_img = cv2.morphologyEx(input_image,  cv2.MORPH_TOPHAT, kernel) 
	bophat_img = cv2.morphologyEx(input_image,  cv2.MORPH_BLACKHAT, kernel)
	image = gray -tophat_img  
	image2 = image+ bophat_img
	cv2.imshow("original", image2) 
	cv2.imshow("tophat", image) 
	text = pytesseract.image_to_string(image, config='--psm 8 -c tessedit_char_whitelist=0123456789')
	print("programming_fever's License Plate Recognition\n")
	print("Detected license plate Number is:",text)
	med_blur = cv2.medianBlur(image,1)
	blur = cv2.GaussianBlur(med_blur,(3,3),0)    
	bin_img = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 199, 5) 
	bin_img = cv2.bitwise_not(bin_img)
	print(bin_img)
	cv2.imshow("img",bin_img)
	image = Image.fromarray(bin_img)
	#cv2.imshow("imgg",image)
	#cv2.imshow('Cropped',Cropped)
	tools = pyocr.get_available_tools()
	tool = tools[0]
	#img = np.asarray(img) 
	plate_number = tool.image_to_string(image, lang='eng', builder=pyocr.builders.TextBuilder())  
	print("plate_number:", plate_number)


	cv2.waitKey(0) 
	cv2.destroyAllWindows()
main()
       
