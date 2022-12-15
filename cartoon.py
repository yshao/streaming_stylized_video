import numpy as np
import cv2

def cartoon(srcColor):
    srcGray=cv2.cvtColor(srcColor,cv2.COLOR_BGR2GRAY)
    print srcGray.shape,srcColor.shape
    cv2.medianBlur(srcGray,5,srcGray)

    mask=srcGray.copy().astype(np.uint8)
    edges=srcGray.copy().astype(np.uint8)

    ### sketch detection
    cv2.Laplacian(srcGray,cv2.CV_8U,edges,5)
    cv2.threshold(edges,60,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU,mask)
    outImg=srcColor.copy()
    tmp=outImg.copy()

    ### bilateral filtering ###
    rep=10
    for i in xrange(rep):
        size=9;sigmaColor=9;sigmaSpace=7

        cv2.bilateralFilter(outImg,size,sigmaColor,sigmaSpace,tmp)
        cv2.bilateralFilter(tmp,size,sigmaColor,sigmaSpace,outImg)

    output=cv2.bitwise_and(srcColor,srcColor,mask=mask)
    cv2.edgePreservingFilter(output,output)

    return output



if __name__ == '__main__':
    ""
    cv2.imwrite('cartoon.jpg',cartoon(cv2.imread('test.jpg')))