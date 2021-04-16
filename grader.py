import cv2
import numpy as np
from imutils.perspective import four_point_transform
import imutils
from PIL import Image
import io


# image = cv2.imread('Image/bs1.jpg')
# image = cv2.imread('Image/bs2.png')
# image = cv2.imread('Image/bs3.png')
# image = cv2.imread('Image/bs4.jpg')
# image = cv2.imread('Image/bs5.jpg')
image = cv2.imread('Image/bs6.jpg')
# contours_img = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 50, 100)

contours, hierarchy = cv2.findContours(
    edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(image, contours, -1, (0, 255, 0), 10)
# cv2.imshow("Image", image)
area_cnt = [cv2.contourArea(cnt) for cnt in contours]
area_sort = np.argsort(area_cnt)[::-1]
# for cnt in contours:
#     with open("contours.txt", "a") as output:
#         output.write(str(cnt) + '\n contourArea ' + str(cv2.contourArea(cnt)) + '\n')

# print(np.argsort(area_cnt))
# print(area_cnt[:10])
cnts = contours[area_sort[0]]
p = cv2.arcLength(cnts, True)
r = cv2.approxPolyDP(cnts, 0.01*p, True)
# cv2.drawContours(image, [r], -1, (0, 0, 255), 3)
print(r.shape)
r = r.reshape(4, 2)

rect = np.zeros((4, 2), dtype='float32')
s = np.sum(r, axis=1)
rect[0] = r[np.argmin(s)]
rect[2] = r[np.argmax(s)]
diff = np.diff(r, axis=1)
rect[1] = r[np.argmin(diff)]
rect[3] = r[np.argmax(diff)]
(tl, tr, br, bl) = rect
width1 = np.sqrt((tl[0] - tr[0])**2 + (tl[1] - tr[1])**2)
width2 = np.sqrt((bl[0] - br[0])**2 + (bl[1] - br[1])**2)
Width = max(int(width1), int(width2))

height1 = np.sqrt((tl[0] - bl[0])**2 + (tl[1] - bl[1])**2)
height2 = np.sqrt((tr[0] - br[0])**2 + (tr[1] - br[1])**2)
Height = max(int(height1), int(height2))
new_rect = np.array([
    [-10, -10],
    [Width+10, -10],
    [Width+10, Height+10],
    [-10, Height+10]], dtype="float32")

M = cv2.getPerspectiveTransform(rect, new_rect)

output = cv2.warpPerspective(image, M, (Width, Height))
# cv2.imwrite('new_img.jpg', output)
# new_img = cv2.imread('new_img.jpg')
gray2 = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
# blurred2 = cv2.GaussianBlur(gray2, (5, 5), 0)
# edged2 = cv2.Canny(blurred2, 50, 100)
ret, thresh = cv2.threshold(
    gray2, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

contours2, hierarchy2 = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# img = cv2.drawContours(output.copy(), contours2, -1, (0, 255, 0), 3)

questionCnts = []
# loop over the contours
for c in contours2:
	# compute the bounding box of the contour, then use the
	# bounding box to derive the aspect ratio
	(x, y, w, h) = cv2.boundingRect(c)
	ar = w / float(h)
	# in order to label the contour as a question, region
	# should be sufficiently wide, sufficiently tall, and
	# have an aspect ratio approximately equal to 1
	if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
		questionCnts.append(c)
print(questionCnts)
f = open("questionCnts.txt", "a")
f.write(str(questionCnts))
cv2.imshow("Image", thresh)

# cv2.imshow("Output", thresh)


cv2.waitKey()
