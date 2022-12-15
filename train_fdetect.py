import numpy as np
import cv2
from cartoon import cartoon

maiorArea = 0

base=0
VIDFILE="scene.avi"
cap = cv2.VideoCapture(VIDFILE)
_,f = cap.read()
avg2 = np.float32(f)
### training ###
lFrames=[]
while(True):
    ret,f = cap.read()
    if ret != True:
        break

    cv2.accumulateWeighted(f,avg2,0.01)
    base = cv2.convertScaleAbs(avg2)
    lFrames.append(base)
cap.release()
c = cv2.VideoCapture(VIDFILE)
_,f = c.read()
i=0
while(True):
    ret,f = c.read()
    if ret == False:
        break
    f=lFrames[i]
    cv2.accumulateWeighted(f,avg2,0.005)

    base = cv2.convertScaleAbs(avg2)
    base = cv2.GaussianBlur(base,(3,3),0)
    i+=1

c.release()
cv2.imwrite('base.jpg',base)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg=cv2.createBackgroundSubtractorMOG2()

# ct=0
# while ct < 20:
#     frame=base
#     fgmask=fgbg.apply(frame)
#     bgmask=fgbg.apply(frame,-1)
#     fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
#     fgmask = cv2.medianBlur(fgmask,5)
#     fgmask = cv2.medianBlur(fgmask,5)
#     fgmask = cv2.medianBlur(fgmask,5)
#     # print "HI"
#     ct+=1




cap = cv2.VideoCapture(VIDFILE)
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     cv2.imshow("Webcam", frame)
# #     bkg=frame.copy()
# #     fundo = cv2.GaussianBlur(bkg,(3,3),0)
#     print("OK")
#     if cv2.waitKey(1) == 32:
#         cv2.destroyWindow("Webcam")
#         break

w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT ))
r=int(cap.get(cv2.CAP_PROP_FPS ))

### video recorder
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter("testA_output.avi", fourcc, r, (w, h))

while True:
    ret, frame = cap.read()
    # ret, frame = cap.read()
    # ret, frame = cap.read()
    # ret, frame = cap.read()
    # ret, frame = cap.read()
    # ret, frame = cap.read()
    # ret, frame = cap.read()
    # ret, frame = cap.read()
    # ret, frame = cap.read()
    # ret, frame = cap.read()
    # ret, frame = cap.read()
    # ret, frame = cap.read()
    # ret, frame = cap.read()
    # ret, frame = cap.read()
    # ret, frame = cap.read()
    # ret, frame = cap.read()
    # ret, frame = cap.read()
    # ret, frame = cap.read()
    # ret, frame = cap.read()
    # ret, frame = cap.read()

    # print ret
    imagem=frame.copy()
    imagem = cv2.GaussianBlur(imagem,(3,3),0)
    diff=imagem.copy()
    thresh=imagem.copy()
    #cv2.imshow("Webcam", imagem)
    imagem = cv2.GaussianBlur(imagem,(3,3),0)
    diff=cv2.absdiff(imagem,base)
    m1=diff.copy()
    m1[diff < 40]=0
    m1[diff >= 40]=255

    can=cv2.Canny(m1,40,200)
    gray = cv2.cvtColor(m1, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray,5)
    gray = cv2.medianBlur(gray,5)
    gray = cv2.GaussianBlur(gray,(5,5),0)
    gray[gray > 20]=255
    ret,thresh1 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((3,3),np.uint8)
    # dilated = cv2.dilate(thresh1,kernel,iterations = 20)
    # cinza = cv2.erode(dilated,kernel,iterations = 10)
    thresh=thresh1
    dilated=cv2.erode(thresh,kernel,iterations=5)
    dilated=cv2.dilate(dilated,kernel,iterations=10)
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    dilated=cv2.dilate(dilated,kernel,iterations=2)
    dilated=cv2.erode(dilated,kernel,iterations=5)

    frame=imagem
    fgmask=fgbg.apply(frame)
    bgmask=fgbg.apply(frame,-1)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    fgmask = cv2.medianBlur(fgmask,5)
    fgmask = cv2.medianBlur(fgmask,5)
    fgmask = cv2.medianBlur(fgmask,5)

    # fgmask[fgmask < 50]=0

    # mask=np.bitwise_or((fgmask> 170),(dilated > 0))
    # mask=(dilated > 0)
    # mask = dilated > 0
    mask=fgmask.copy()
    nMask=np.zeros((mask.shape[0],mask.shape[1])).astype(np.uint8)
    nMask[mask]=255
    nMask =cv2.dilate(nMask,kernel,iterations=4)
    nMask = cv2.morphologyEx(nMask, cv2.MORPH_CLOSE, kernel)
    nMask =cv2.erode(nMask,kernel,iterations=2)

    # _,cont,heir=cv2.findContours(nMask.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # for c in cont:
    # cv2.drawContours(cv2.convexHull(cont[0]),-1,255,cv2.FILLED)
    # c=cont

    # nH=[]
    # for c in cont:
    #     h=cv2.convexHull(c)
    #     h=cv2.approxPolyDP(c,0.1,True)
        # nH.append(h)

    nnMask=np.zeros((mask.shape[0],mask.shape[1]))
    # cv2.drawContours(nnMask,nH,-1,255,cv2.FILLED)

    #### reound2 ###
    img=frame.copy()
    img = cv2.GaussianBlur(img,(5,5),0)
    dMask2=nMask.copy()

    dMask1=dMask2.copy()
    dMask=cv2.dilate(dMask1,kernel,iterations=5)
    dMask = cv2.morphologyEx(dMask, cv2.MORPH_CLOSE, kernel)
    dMask=cv2.dilate(dMask,kernel,iterations=10)
    dMask = cv2.morphologyEx(dMask, cv2.MORPH_CLOSE, kernel)

    # img[dMask == 0]=0
    # diffMask=dMask - dMask2
    # img=img.astype(np.uint8)
    # diffMask=cv2.bitwise_not(diffMask)
    # diffMask = cv2.morphologyEx(diffMask, cv2.MORPH_OPEN, kernel)
    #
    #
    # _,cont,heir=cv2.findContours(nMask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # diffMask=np.zeros((diffMask.shape[0],diffMask.shape[1]))
    # diffMask=cv2.drawContours(diffMask,cont,-1,255,cv2.FILLED)
    # diffMask=cv2.dilate(diffMask,kernel,iterations=8)
    # diffMask = cv2.morphologyEx(diffMask, cv2.MORPH_CLOSE, kernel)


    ### blending ###
    nMask=np.zeros((gray.shape[0],gray.shape[1]))
    bits=np.bitwise_or((fgmask> 0),(gray> 0))
    nMask[bits]=255
    nMask=cv2.erode(nMask,kernel,iterations=5)
    nMask=cv2.dilate(nMask,kernel,iterations=8)
    nMask = cv2.morphologyEx(nMask, cv2.MORPH_CLOSE, kernel)
    car=cartoon(frame.copy())
    img[nMask==255]=car[nMask==255]
    img[nMask==0]=frame[nMask==0]


    # cv2.imshow("AbDiff", diff)
    # cv2.imshow("Diff1", m1)
    # cv2.imshow("Gray", gray)
    # cv2.imshow("Webcam", imagem)
    # cv2.imshow("Dilated", dilated )
    # cv2.imshow("Motion", fgmask)
    # cv2.imshow("Mask", nMask)
    # cv2.imshow("New Mask",img)
    # cv2.imshow("Base",base)
    # cv2.imshow("Canny",can)

    video_writer.write(img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    

### close video ###
cap.release()
video_writer.release()
cv2.destroyAllWindows()