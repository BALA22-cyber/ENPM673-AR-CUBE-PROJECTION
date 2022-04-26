import cv2 as cv
import numpy as np
# from google.colab.patches import cv2_imshow
from matplotlib import pyplot as plt
import math
from utils import *
# from Homography import 

video = cv.VideoCapture("/home/kb2205/Desktop/ENPM 673/project 1/output.avi")
testudo = cv.imread("/home/kb2205/Desktop/ENPM 673/project 1/testudo.png")
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('3D projection.avi', fourcc, 30.0, (800,400))
## RESIZING TESTUDO    
testudo = cv.resize(testudo, (160,160))

### Intrinsic Camera Parameters

K = np.array([[1346.100595,0,932.1633975],
              [0,1355.933136,654.8986796],
                    [0,0,1]             ])
    
while True:
    isTrue, frame = video.read()
    TAG_VIDEO = frame.copy() 
    # if not isTrue: break
    gray = cv.cvtColor(frame , cv.COLOR_BGR2GRAY)
    # gray = np.float32(gray)
    blur = cv.GaussianBlur(gray,(7,7),cv.BORDER_DEFAULT)
    # cv.imshow("blur",blur)
    retval,threshed = cv.threshold(blur,150,200,cv.THRESH_BINARY)
    # cv.imshow('OG',frame)
    # cv.imshow('threshed',threshed)
    
    #######          FFT AND INVERSE FOURIER TRANSFORMS
    
    dft = cv.dft(np.float32(threshed), flags=cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    magnitude_spectrum = 20 * np.log(cv.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    rows, cols = threshed.shape
    crow, ccol = int(rows / 2), int(cols / 2)  # center

    #######           MASKING      
    # ####    Circular HPF mask, center circle is 0, remaining all ones

    mask = np.ones((rows, cols, 2), np.uint8)
    r = 80
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 0 #value 0 for high pass filter

    # apply mask in frequency domain
    fshift = dft_shift * mask
    fshift_mask_mag = 2000 * np.log(cv.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv.idft(f_ishift)
    img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    
    # cv.imshow("inversefft",img_back) 
       
       
       
    ### Corner detection using GOOD FEATURES TO TRACK
       
    corners = cv.goodFeaturesToTrack(img_back,20,0.01,10)       
    corners= (np.int0(corners[:,0]))            
    x,y = corners.T
    i=y.argmax()
    j=x.argmin()
    k=y.argmin()
    l=x.argmax()
    
    c0, c1, c2, c3 = corners[(i,j,k,l),] 
    #cn is the centre of the tag 
    cn = (c0+c1+c2+c3)/4
    dt = math.dist(cn, c0)
    tag_corners = []
    for pt in corners:
        
        if 0.25*dt < math.dist(pt, cn) < 0.58*dt :
            tag_corners.append(pt)
    
    
    
    for x, y in tag_corners:

        cv.circle(frame, (x, y), 2, [0,0,255], -1)
    
    tag_corners = np.array(tag_corners)
    
    m,n = tag_corners.T
   
    I0 = m.argmax()
    I1 = n.argmin()
    I2 = m.argmin()
    I3 = n.argmax() 
    I0, I1, I2, I3 = tag_corners[(I0,I1,I2,I3),]
    Icorner_points= np.array([I0,I1,I2,I3])

    
    cv.putText(frame, 'P0', I0, cv.FONT_HERSHEY_PLAIN, 1,((209, 80, 0, 255)),1 )
    cv.putText(frame, 'P1', I1, cv.FONT_HERSHEY_PLAIN, 1,((209, 80, 0, 255)),1 )
    cv.putText(frame, 'P2', I2, cv.FONT_HERSHEY_PLAIN, 1,((209, 80, 0, 255)),1 )
    cv.putText(frame, 'P3', I3, cv.FONT_HERSHEY_PLAIN, 1,((209, 80, 0, 255)),1 )
    # cv.imshow('corners', frame)
    
    
    ###### HOMOGRAPHY and WARPING of Tag corners
    
    # dstpnt = np.array([[0,0],[160,0],[160,160],[0,160]])
    # TAG_H = Homograph(Icorner_points,dstpnt)
    # TAG_WARPING = warpPerspective(TAG_VIDEO,TAG_H,(160,160))
    # cv.imshow("warpPews",TAG_WARPING)

    # inner_grid,a,b,c,d= decode_tag(TAG_WARPING)
    # cv.imshow("decoded",inner_grid)
    
    # ######   WARPING THE IMAGE
    
    w,h = testudo.shape[:2]
    p1 =np.float32([[0,0],[w,0],[w,h],[0,h]])
    p2=np.float32(Icorner_points)
    h1, w1 = gray.shape 
    H = Homograph(p1,p2)  
    warped = warpPerspective(testudo,H,(w1,h1))
    p2=np.int32(p2)
    cv.fillConvexPoly(frame,p2,(0,0,0))
    frame = frame + warped
    
    # cv.imwrite("blurred.jpg",blur)
    # cv.imwrite("thresholded Image.jpg",threshed)
    # cv.imwrite("inverse FFT.jpg",img_back)
    # cv.imwrite("Inner Corners in frame.jpg",frame)
    # cv.imwrite("SuperImposed Testudo.jpg",frame)
    
    cv.imshow("testudowarp",frame)

   ## writing cubes
    cube_position = np.array([[0,0,0,1],[0,160,0,1],[160,160,0,1],
                    [160,0,0,1],[160,0,-80,1],[0,0,-80,1],[0,160,-80,1],
                    [160,160,-80,1]])
    
    P = projectionMatrix(H, K)
    new_cube = P @ cube_position.T
    new_cube = np.int0(new_cube[:2]/new_cube[2])
    x, y  = new_cube
    
    cv.line(frame,(int(x[0]),int(y[0])),(int(x[5]),int(y[5])), (0,0,255), 2)
    cv.line(frame,(int(x[1]),int(y[1])),(int(x[6]),int(y[6])), (0,0,255), 2)
    cv.line(frame,(int(x[2]),int(y[2])),(int(x[7]),int(y[7])), (0,0,255), 2)
    cv.line(frame,(int(x[3]),int(y[3])),(int(x[4]),int(y[4])), (0,0,255), 2)

    cv.line(frame,(int(x[0]),int(y[0])),(int(x[1]),int(y[1])), (0,0,255), 2)
    cv.line(frame,(int(x[1]),int(y[1])),(int(x[2]),int(y[2])), (0,0,255), 2)
    cv.line(frame,(int(x[2]),int(y[2])),(int(x[3]),int(y[3])), (0,0,255), 2)
    cv.line(frame,(int(x[3]),int(y[3])),(int(x[0]),int(y[0])), (0,0,255), 2)
    
    cv.line(frame,(int(x[4]),int(y[4])),(int(x[5]),int(y[5])), (0,0,255), 2)
    cv.line(frame,(int(x[4]),int(y[4])),(int(x[7]),int(y[7])), (0,0,255), 2)
    cv.line(frame,(int(x[5]),int(y[5])),(int(x[6]),int(y[6])), (0,0,255), 2)
    cv.line(frame,(int(x[6]),int(y[6])),(int(x[7]),int(y[7])), (0,0,255), 2)
    
    
   
    cv.imshow("cube projection",frame)    
    
    out.write(frame)

    
    
    
    plt.subplot(2, 2, 1), plt.imshow(threshed, cmap='gray')
    plt.title('Input Image (Threshed)'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('After FFT'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(fshift_mask_mag, cmap='gray')
    plt.title('FFT + Mask'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(img_back, cmap='gray')
    plt.title('After FFT Inverse'), plt.xticks([]), plt.yticks([])
    
    if cv.waitKey(3) & 0xFF==ord("k"):
        break
video.release()
out.release()
plt.show()
cv.destroyAllWindows()

plt.show()
cv.destroyAllWindows()

