import cv2
# import numpy as np
import imutils

def centroid_craters(crater_image, segmentation_mask, filename_output_satellite_image):

    image = cv2.imread(segmentation_mask)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # img = cv2.bitwise_not(img)
    thresh = cv2.threshold(image_gray, 110, 255, cv2.THRESH_BINARY)[1]

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    # cntsSorted = sorted(cnts, key=lambda x: cv2.contourArea(x))
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for c in cnts[:3]:

        # compute the center of the contour
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        print('x:', cX, ' y:', cY,' area:', cv2.contourArea(c))

        # draw the contour and center of the shape on the image
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        cv2.circle(image, (cX, cY), 3, (0, 0, 255), -1)
        cv2.putText(image,'('+str(cX)+','+str(cY)+')',(cX - 10, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)



    cv2.imwrite(filename_output_satellite_image, image)

    return



segmentation_mask = '../output/PredictedCraterMasks/AS16-M-0872-predicted_mask.jpg'
crater_image = '../output/BoundedCraters/AS16-M-0872-with_bboxes.jpg'
centroid_craters(crater_image, segmentation_mask,
                 '../output/CentroidCraterMasks/AS16-M-0872-predicted_mask_centroid.jpg')