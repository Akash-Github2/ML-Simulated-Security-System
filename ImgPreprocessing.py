import cv2

def resizeImg(img, newWidth, newHeight):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print('Original Dimensions : ',img.shape)
    dim = (newWidth, newHeight)
    # resize image
    resizedImg = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)
    return resizedImg #returns the 2d pixel array --> can directly use it with tensorflow
