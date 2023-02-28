#Have this editor side by side with QuickTime streaming iphone's camera and it will work
import pyscreenshot as ps
from time import sleep
import ImgPreprocessing as ip
import ImgRecog as ir
import cv2
import os

#Training Model
mdl = ir.trainModel()

#Testing Model
cnt = 1

while True:
    img = ps.grab(bbox=(2400, 300, 2900, 1150)) #rescale from big to small
    img.save(f"TestingData/img{cnt}.png")
    print("Count:", cnt)

    temp = cv2.imread(f"TestingData/img{cnt}.png")
    newImg = ip.resizeImg(temp, ir.wid, ir.wid)
    isOpen = ir.makePrediction(mdl, newImg)
    print(cnt, ": isOpen = ", isOpen)
    if isOpen:
        os.system(f"mv TestingData/img{cnt}.png TestingData/Open/img{cnt}.png")
        os.system("afplay CloseDoor.mp3")
    else:
        os.system(f"mv TestingData/img{cnt}.png TestingData/Close/img{cnt}.png")

    cnt += 1
