import numpy as np
import cv2

img1 = cv2.imread('img1.png')
img2 = cv2.imread('img2.jpg')

# Le quitamos el fondo a la imagen 1
img1Gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
_, thr1 = cv2.threshold(img1Gray, 250, 255, cv2.THRESH_BINARY_INV)
img1Cutted = cv2.bitwise_and(img1, img1, mask = thr1)

# Dimensiones de la imagen 1
rows,cols,channels = img1.shape

# Nos quedamos con el fondo no usado por la imagen 1 de la imagen de fondo (img2)
backMask = cv2.bitwise_not(thr1)
img2Roi = img2[0:rows, 0:cols]
img2Cutted = cv2.bitwise_and(img2Roi, img2Roi, mask = backMask)

# Agregamos la imagen cortada a la imagen de fondo original
img2[0:rows, 0:cols ] = cv2.add(img2Cutted, img1Cutted)

cv2.imshow('img1', img1)
cv2.imshow('img2', img2)

while(True):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()