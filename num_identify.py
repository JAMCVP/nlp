import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.layers import AveragePooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob
import numpy as np
# remove warning message
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# required library
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from local_utils import detect_lp
from os.path import splitext,basename
from keras.models import model_from_json
import glob
from google.colab.patches import cv2_imshow

# Load model architecture, weight and labels
json_file = open('MobileNets_character_recognition.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("License_character_recognition_weight.h5")
print("[INFO] Model loaded successfully...")

labels = LabelEncoder()
labels.classes_ = np.load('license_character_classes.npy')
print("[INFO] Labels loaded successfully...")

#############PLATE SEGMNETATION 3#########################################
import cv2
import PIL
from PIL import Image
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
areaa =[]


crop_characters = []
global w1,h1,w2

def sort_contours(cnts,reverse = False):
  i = 0
  boundingBoxes = [cv2.boundingRect(c) for c in cnts]
  (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),key=lambda b: b[1][i], reverse=reverse))
  return cnts
def is_plate(c):
	# approximate the contour
  peri = cv2.arcLength(c, True)
  approx = cv2.approxPolyDP(c, 0.02 * peri, True)
  # the contour is 'bad' if it is not a rectangle
  return  len(approx) == 4  

def main():
  #xmin,ymin,xmax,ymax=236,291,450,361
  path ='numberplate.jpg'		
  #path ='/home/dell/Downloads/Data-Images/Plates/37.jpg'
  image = cv2.imread(path)
  img_lp = cv2.imread(path)
  mask = np.ones(img_lp.shape[:2], dtype="uint8") * 255
  #img_lp = cv2.resize(img, (150, 75))
  h1,w1,c =img_lp.shape
  print(w1,h1,c)
  area1 = w1*h1
  img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
  img_gray_lp = cv2.medianBlur(img_gray_lp, 5)
  _, img_binary_lp = cv2.threshold(img_gray_lp, 100, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  img_binary_lp = cv2.erode(img_binary_lp, (3, 3))
  img_binary_lp = cv2.dilate(img_binary_lp, (3,3))
  bin_img = cv2.bitwise_not(img_binary_lp)
  bin_img = cv2.medianBlur(bin_img, 3) 
  #bin_img = cv2.GaussianBlur(bin_img,(3,3),0)
  #print(bin_img)
  #cv2.imshow("img",bin_img)
  cont, _  = cv2.findContours(bin_img, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  #cont, _  = cv2.findContours(bin_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

  #cv2.drawContours(img_lp,cont,-1,(255,0,255),4)
	#cv2.imshow('Contours',img_lp)
  contours = sorted(cont, key = cv2.contourArea, reverse = True)[:11]
	#print(contours)
  contour = []
  cn=[]
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
      cv2.drawContours(mask, [c], -1, 0, -1)
      w2= w2+w
			#print(w2)
			#print(contours[i])
      area = cv2.contourArea(c)
      areaa.append(area)
			#print(area,x, y, w, h)
      cn.append(c)
  w2 = w2/len(contours)
	#print(w2)		
	#print(img_gray_lp.shape)		
  cv2.drawContours(img_gray_lp,contours[0],-1,(255,0,255),4)
  cv2.rectangle(img_lp,(x,y),(x+w,y+h),(0,255,0),2)
  #cv2.imshow('Contours',img_gray_lp)
  digit_w, digit_h = 30, 60

  crop_characters = []
  for c in sort_contours(cont):
    #print(c)
    areaaa= cv2.contourArea(c)
    if areaaa  in areaa:
      cv2.drawContours(mask, [c], -1, 0, -1)
      x, y, w, h = cv2.boundingRect(c)
      if (w < (.30*(w1))) or (w>(.80*(w2))):
        cv2.rectangle(img_lp,(x,y),(x+w,y+h),(0,255,0),2)
        curr_num = img_gray_lp[y:y+h,x:x+w]
        curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
        #cv2.imshow('Contour',img_lp)
				#curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        curr_num = cv2.bitwise_not(curr_num)
			#cv2.imshow("Afte", curr_num)
        crop_characters.append(curr_num)
  image = cv2.bitwise_and(img_lp, img_lp, mask=mask)
  print("Detect {} letters...".format(len(crop_characters)))
  #cv2.imshow("After", img_lp)
  global crop
  crop = crop_characters
	
		
main()

fig = plt.figure(figsize=(14,4))
grid = gridspec.GridSpec(ncols=len(crop),nrows=1,figure=fig)

for i in range(len(crop)):
    fig.add_subplot(grid[i])
    plt.axis(False)
    plt.imshow(crop[i],cmap="gray")
    print(i) #cv2_imshow(crop_characters[i])
    #print("next")
    #print(crop_characters[i])

def predict_from_model(image,model,labels):
    image = cv2.resize(image,(80,80))
    image = np.stack((image,)*3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
    return prediction
    
fig = plt.figure(figsize=(15,3))
cols = len(crop)
grid = gridspec.GridSpec(ncols=cols,nrows=1,figure=fig)

final_string = ''
for i,character in enumerate(crop):
    fig.add_subplot(grid[i])
    title = np.array2string(predict_from_model(character,model,labels))
    plt.title('{}'.format(title.strip("'[]"),fontsize=20))
    final_string+=title.strip("'[]")
    plt.axis(False)
    plt.imshow(character,cmap='gray')

print("Achieved result: ", final_string)
#plt.savefig('final_result.png', dpi=300)
