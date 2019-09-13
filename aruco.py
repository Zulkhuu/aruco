import numpy as np
import cv2

# Aruco library
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
aruco_params =  cv2.aruco.DetectorParameters_create()

# Capture camera 0
cap = cv2.VideoCapture(0)

# Mask image to be used
mask = cv2.imread('mask.jpg')

def overlay_img(im_msk, im_dst, corners):
    """Overlay roi defined by corners in im_dst by im_msk"""
    # Four corners of detected marker
    pts = [(corner[0], corner[1]) for corner in corners]
    pts_dst = np.array(pts)

    # Four corners of input image
    h, w = im_msk.shape[0], im_msk.shape[1]
    pts_src = np.array([[0,0],
                        [w - 1, 0],
                        [w - 1, h-1],
                        [0, h - 1 ]],
                        dtype=float);

    # Find the perspective transformation between two planes
    h, _ = cv2.findHomography(pts_src, pts_dst)
    # Apply the perspective transformation to the mask image
    warped_mask = cv2.warpPerspective(im_msk, h, (im_dst.shape[1],im_dst.shape[0]))
    # Fill detected marker location with zeroes in input image
    cv2.fillConvexPoly(im_dst, pts_dst.astype(int), 0, 16);
    # Add warped mask image to input image
    im_dst = im_dst + warped_mask

    return im_dst

while(True):
    # Read frame
    ret, frame = cap.read()
    im_dst = frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform Aruco marker detection
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    # Replace markers with mask image
    if np.all(ids is not None):
        #display = aruco.drawDetectedMarkers(frame, corners)
        for i in range(len(ids)):
            corner = corners[i][0][:][:] # 4 corners of detected marker
            im_dst = overlay_img(mask, im_dst, corner)

    # Display result
    cv2.imshow('Display',im_dst)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
