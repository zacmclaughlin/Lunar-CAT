import cv2
# import numpy as np
import imutils


def centroid(crater_image, segment_image):
    image_gray = cv2.cvtColor(segment_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(image_gray, 110, 255, cv2.THRESH_BINARY)[1]
    print(type(thresh))
    contour_areas = cv2.findContours(thresh.copy(),
                                     cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
    contour_areas = imutils.grab_contours(contour_areas)
    contour_areas = sorted(contour_areas,
                           key=cv2.contourArea,
                           reverse=True)

    for c in contour_areas[:3]:
        # compute the center of the contour
        M = cv2.moments(contour_areas)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        print('x:', cX, ' y:', cY, ' area:', cv2.contourArea(c))

        # draw the contour and center of the shape on the image
        cv2.drawContours(segment_image, [contour_areas], -1, (0, 255, 0), 2)
        cv2.circle(segment_image, (cX, cY), 3, (0, 0, 255), -1)
        cv2.putText(segment_image,
                    '(' + str(cX) + ',' + str(cY) + ')',
                    (cX - 10, cY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0), 2)
        cv2.circle(crater_image, (cX, cY), 3, (0, 0, 255), -1)

    return crater_image, segment_image


def get_centroid_craters(crater_image, segmentation_mask, path_and_filename_to_save):

    crater_image, segmentation_mask = centroid(crater_image, segmentation_mask)
    cv2.imwrite(path_and_filename_to_save, segmentation_mask)
    return crater_image, segmentation_mask


def centroid_craters_from_file_to_file(crater_image_path_and_filename,
                                       segmentation_mask_path_and_filename,
                                       path_and_filename_to_save):

    segment_image = cv2.imread(segmentation_mask_path_and_filename)
    crater_image = cv2.imread(crater_image_path_and_filename)
    crater_image, segment_image = centroid(crater_image, segment_image)
    cv2.imwrite(path_and_filename_to_save, segment_image)


def centroid_one_crater():
    segmentation_mask = '../output/PredictedCraterMasks/AS16-M-0872-predicted_mask.jpg'
    crater_image = '../output/BoundedCraters/AS16-M-0872-with_bboxes.jpg'
    centroid_craters_from_file_to_file(crater_image, segmentation_mask,
                                       '../output/CentroidCraterMasks/AS16-M-0872-predicted_mask_centroid.jpg')
