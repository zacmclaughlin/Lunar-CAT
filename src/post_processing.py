import cv2
import numpy as np
import imutils


def centroid(crater_image, segment_image):

    image_gray = cv2.cvtColor(segment_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(image_gray, 200, 255, cv2.THRESH_BINARY)[1]
    contour_areas = cv2.findContours(thresh.copy(),
                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_areas = imutils.grab_contours(contour_areas)
    contour_areas = sorted(contour_areas,
                           key=cv2.contourArea,
                           reverse=True)
    print(crater_image.shape)
    print(segment_image.shape)
    crater_image = cv2.addWeighted(crater_image, 0.45, segment_image, 0.55, 0)

    if len(contour_areas) > 0: #need check to handle images without identified crater contours
        for c in contour_areas[:4]:
            # compute the center of the contour
            M = cv2.moments(c)

            if M["m00"]!= 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0


            print('x:', cX, ' y:', cY, ' area:', cv2.contourArea(c))

            # draw the contour and center of the shape on the image
            # cv2.drawContours(segment_image, [c], -1, (0, 255, 0), 2)
            cv2.circle(segment_image, (cX, cY), 3, (0, 0, 255), -1)
            # cv2.putText(segment_image,
            #             '(' + str(cX) + ',' + str(cY) + ')',
            #             (cX - 10, cY - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX,
            #             0.5,
            #             (255, 0, 0), 2)
            cv2.circle(crater_image, (cX, cY), 6, (0, 0, 255), -1)
    return crater_image, segment_image


def get_centroid_craters(crater_image, segmentation_mask, path_and_filename_to_save):
    crater_image = cv2.cvtColor(crater_image, cv2.COLOR_BGR2GRAY)
    #segmentation_mask = cv2.cvtColor(segmentation_mask, cv2.COLOR_BGR2GRAY)
    crater_image, segmentation_mask = centroid(crater_image, segmentation_mask)
    cv2.imwrite(path_and_filename_to_save, segmentation_mask)
    return crater_image, segmentation_mask


def centroid_craters_from_file_to_file(crater_image_path_and_filename,
                                       segmentation_mask_path_and_filename,
                                       path_and_filename_to_save):

    segment_image = cv2.imread(segmentation_mask_path_and_filename)
    crater_image = cv2.imread(crater_image_path_and_filename)
    # image_gray = cv2.cvtColor(seg_image, cv2.COLOR_BGR2GRAY)
    crater_image, segment_image = centroid(crater_image, segment_image)
    cv2.imwrite(path_and_filename_to_save, crater_image)


def centroid_one_crater():
    segmentation_mask = '../output/PredictedCraterMasks/AS16-M-0344-predicted_mask.jpg'
    crater_image = '../output/BoundedCraters/AS16-M-0344-with_bboxes.jpg'
    centroid_craters_from_file_to_file(crater_image, segmentation_mask,
                                       '../output/CentroidCraterMasks/plswork.jpg')

# segmentation_mask = '../output/PredictedCraterMasks/AS16-M-0343-predicted_mask.jpg'
# crater_image = '../output/BoundedCraters/AS16-M-0343-with_bboxes.jpg'
centroid_one_crater()