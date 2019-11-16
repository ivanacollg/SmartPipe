import cv2 as cv
import numpy as np

video_name = 'SEQ1'
#video_name = 'SEQ_0516_stab'
cap = cv.VideoCapture('../../../IMT_09/Vision/Proyecto/videos/SEQ1.avi')

if (cap.isOpened()== False): 
    print("Error opening video stream or file")

imageSize   = (320, 240)
thres_limit = 60

element_erode   = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11,11))
element_dilate  = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11,11))
blur_kernel = (5,5)

ret, frame1  = cap.read()
frame1       = frame1[10:frame1.shape[0] - 10, 10:frame1.shape[1] - 10]
frame1       = cv.resize(frame1, imageSize)
ret, frame1  = cv.threshold(frame1,thres_limit,255,cv.THRESH_BINARY_INV)
prvs         = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
hsv          = np.zeros_like(frame1)
hsv[...,0]   = 1
hsv[...,1]   = 0

frame_count = 1

while(1):
    ret, frame2 = cap.read()

    if ret == True:
        frame_count += 1
        frame2       = frame2[10:frame2.shape[0] - 10, 10:frame2.shape[1] - 10]
        frame2       = cv.resize(frame2, imageSize)
        nextF        = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
        frame_orig   = nextF
        flow         = cv.calcOpticalFlowFarneback(prvs,nextF, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        mag, ang     = cv.cartToPolar(flow[...,0], flow[...,1])
        
        hsv[...,2]   = cv.normalize(mag, None, 0, 255,cv.NORM_MINMAX)
        
        h, s, v = cv.split(hsv)
        blur = cv.GaussianBlur(v,blur_kernel,0)
        
        eroded = cv.erode(blur, element_erode)
        dilated = cv.dilate(eroded, element_dilate)
        
        mean, stddev = cv.meanStdDev(dilated)
        epsilon = 0.1
        stable = cv.subtract(dilated, mean + epsilon * stddev)
        alpha = 2
        ret, stable_thresh = cv.threshold(stable, stddev[0] + alpha, 255, cv.THRESH_TOZERO)
        
        erd_thres = cv.erode(stable_thresh, element_erode)
        dil_thres = cv.dilate(erd_thres, element_dilate)

        cv.putText(v, 'Avg: ' + str(mean), (10,20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        cv.putText(v, 'Stdev: ' + str(stddev), (10,40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

        _, contours, h = cv.findContours(dil_thres, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        rectangles = []
        contours_filtered = []
        approx_contours = []
        epsilon_scale = [0.1, 0.01, 0.001]
        colors = {epsilon_scale[0] : (0,0,255),
                  epsilon_scale[1] : (255,0,0),
                  epsilon_scale[2] : (0,255,0)}

        for c in contours:
            area = cv.contourArea(c)
            if area > 200:
                M = cv.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                x,y,w,h = cv.boundingRect(c)
                rectangles.append([x,y,w,h])
                cv.circle(frame2, (cX, cY), 1, (255, 255, 255), -1)
                rect = cv.rectangle(frame2, (x,y), (x+w,y+h), (255,255,255), 1)
                for i in epsilon_scale:
                    epsilon = i*cv.arcLength(c,True)
                    approx = cv.approxPolyDP(c,epsilon,True)
                    cv.drawContours(frame2, [approx], -1, colors[i], 1)
        

        #cv.drawContours(frame2, contours_filtered, -1, (0,255,0), 3)
        
        '''
        if np.shape(stable) != ():
            cv.imshow('stable',stable)
        '''
        if np.shape(dil_thres) != ():
            dil_thres = cv.resize(dil_thres, (640, 480))
            cv.imshow('stable_thresh',dil_thres)
        '''
        if np.shape(dilated) != ():
            cv.imshow('dilated',dilated)
     
        if np.shape(v) != ():
            cv.imshow('blur',blur)
        '''
        if np.shape(frame2) != ():
            frame2 = cv.resize(frame2, (640, 480))
            #frame_orig = cv.resize(frame_orig, (640, 480))
            cv.imshow('VideoStream',frame2)
            cv.imshow('VideoStream2',frame_orig)
        
        if cv.waitKey(35) & 0xFF == ord('q'):
            break
        elif cv.waitKey(35) & 0xFF == ord('s'):
            for j in range(len(rectangles)):
                i = rectangles[j]
                file_name = 'rois/' + video_name + '_' + str(frame_count) + '_roi_' + str(j) + '.jpg'
                file_name_edges = 'rois/' + video_name + '_' + str(frame_count) + '_roi_edges_' + str(j) + '.jpg'
                roi = frame_orig[i[1]:i[1]+i[3], i[0]:i[0]+i[2]]
                roi = cv.resize(roi, (100,100))
                th2 = cv.adaptiveThreshold(roi,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,11,2)
                #edges = cv.Canny(roi,100,150)
                cv.imwrite(file_name, roi)
                cv.imwrite(file_name_edges, th2)
    else: 
    
        break
    
    prvs = nextF

cap.release() 
cv.destroyAllWindows()