import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('face.jpg')
rect = cv2.imread('face.jpg')

gray = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (int(x-x*0.1), int(y-y*0.1)), (int((x+w)*1.1), int((y+h)*1.1)), (255, 255, 0), 2)
    cv2.imwrite("rectangle_face.jpg", img) # Обводим в прямоугольник
    roi_color = rect[int(y-y*0.1):int((y+h)*1.1), int(x-x*0.1):int((x+w)*1.1)]
    cv2.imwrite("only_face.jpg",roi_color) # Только лицо с отступом 10%

#cv2.imshow('img', img)

#4 пункт
binary_img = cv2.imread('only_face.jpg')
binary_img = cv2.Canny(binary_img, 30, 150)
#cv2.imshow('img', binary_img)
cv2.imwrite("binary_img.jpg", binary_img)



#refresh
final_contors = []
contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    x,y,w,h = cv2.boundingRect(c)
    if w >= 10 and h >= 10:
        final_contors.append(c)
the_mask = np.zeros_like(binary_img)
cv2.drawContours(the_mask, final_contors, -1, (255, 255, 255), cv2.FILLED)
binary_img = cv2.bitwise_and(binary_img, binary_img, mask=the_mask)
cv2.imwrite("binary_img_refresh.jpg", binary_img)
#cv2.imshow("img",binary_img)



#5 пункт
img_morph = cv2.imread('binary_img.jpg',0)
kernel = np.ones((5,5),np.uint8)
erosion = cv2.dilate(img_morph,kernel,iterations=1)
cv2.imwrite("img_morph.jpg", erosion)
#cv2.imshow('img', erosion)



#6 пункт
img_gaus = cv2.imread('img_morph.jpg',0)
img_gaus = cv2.GaussianBlur(img_gaus,(5,5),0)
cv2.imwrite("img_gaus.jpg", img_gaus)

img_gaus = cv2.imread('img_gaus.jpg',0)

zeros_like1 = np.zeros_like(img_gaus)

img_gaus  = cv2.normalize(img_gaus, zeros_like1, 1, 0, cv2.NORM_MINMAX)

#cv2.imwrite("img_gaus_with_norma.jpg", img_gaus)
#cv2.imshow('img', img_gaus)


#7 пункт
img_bilat = cv2.imread('only_face.jpg')
img_bilat=cv2.bilateralFilter(img_bilat,50,50, cv2.BORDER_REFLECT)
cv2.imwrite("img_bilat.jpg", img_bilat)
#cv2.imshow('img', img_bilat)



#8 пункт
img_chetko = cv2.imread('only_face.jpg')
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
img_chetko=cv2.filter2D(img_chetko,-1,kernel)
cv2.imwrite("img_chetko.jpg", img_chetko)
#cv2.imshow('img', img_chetko)





#9 пункт
final=np.expand_dims(img_gaus, axis=2)
result = final * img_chetko + (1 - final) * img_bilat
#cv2.imshow('img', result)
cv2.imwrite('final.jpg', result)
cv2.waitKey()

