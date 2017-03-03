import numpy as np
import cv2

# Define que porcentaje de la imagen debe estar activa para considerarse movimiento
MOVEMENT_THRESHOLD = 0.001

def movementPercentage(img):
    imgWidth, imgHeight = img.shape
    nonZero = cv2.countNonZero(img)
    return nonZero / float(imgWidth * imgHeight)


def isMovementInFrame(frame, mog2):
    # Aplicamos una reduccion de ruido
    frame = cv2.medianBlur(frame, 3)

    # Identificamos el movimiento
    movementResult = mog2.apply(frame)

    # Calculamos el porcentaje de movimiento en la imagen
    movPercentage = movementPercentage(movementResult)

    # Retornamos que hay moviento si superamos un thresshold
    return movPercentage > MOVEMENT_THRESHOLD


def getBlackFrame():
    return np.zeros((300, 300, 3), np.uint8)


def processFrame():
    cap = cv2.VideoCapture(0)

    # Creamos una imagen en negro para cuando no haya movimiento
    blank_image = getBlackFrame()

    # Creamos un substractor de MOG2
    mog2 = cv2.createBackgroundSubtractorMOG2()

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Detectamos si hay moviento en el frame
        isMovement = isMovementInFrame(frame, mog2)

        # Si hubo movimiento mostramos la imagen
        if isMovement:
            cv2.imshow('frame', frame)
        else:
            cv2.imshow('frame', blank_image)
        
        # Si apretan la "q" terminamos el programa
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


# Iniciamos el programa
processFrame()


