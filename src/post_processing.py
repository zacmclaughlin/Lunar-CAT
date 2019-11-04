import cv2


def centroid_craters(crater_image, segmentation_mask):
    """
    :param crater_image: crater image (without bounding boxes)
    :param segmentation_mask: numpy array for a segmentation mask
    :return: crater with segment centroids superimposed on image
    """
    # convert the grayscale image to binary image
    ret, thresh = cv2.threshold(segmentation_mask, 127, 255, 0)

    # find contours in the binary image
    img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        # calculate moments for each contour
        M = cv2.moments(c)

    # calculate x,y coordinate of center
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
    cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
    cv2.putText(img, "centroid", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # display the image
    cv2.imshow("Image", img)
    cv2.waitKey(0)

    return
