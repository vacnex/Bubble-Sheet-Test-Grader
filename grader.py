import cv2
import numpy as np
from imutils.perspective import four_point_transform
import imutils
from imutils import contours
from PIL import Image
import io


# image = cv2.imread('Image/bs1.jpg')
# image = cv2.imread('Image/bs2.png')
image = cv2.imread('Image/bs3.png')
# image = cv2.imread('Image/bs4.jpg')
# image = cv2.imread('Image/bs5.jpg')
# image = cv2.imread('Image/bs6.jpg')
# contours_img = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 50, 100)

contours, hierarchy = cv2.findContours(
    edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
# cv2.imshow("Image", image)
# print('first contour', len(contours))
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
# print(r.shape)
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

contours2 = cv2.findContours(
    thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
# img = cv2.drawContours(output.copy(), contours2, -1, (0, 255, 0), 3)
# print(thresh)
# cv2.imshow('test2', edged)
# print('second contour', len(contours2))

questionCnts = []
for c in contours2:
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)
    if w >= 20 and h >= 20 and ar >= 0.8 and ar <= 1.2:
        questionCnts.append(c)
# print('question contour', len(questionCnts))
# print('cnts question contour', questionCnts)
# img = cv2.drawContours(output.copy(), questionCnts, -1, (0, 255, 0), 3)
# cv2.imshow('test', img)


def get_contour_precedence(contour, cols):
    origin = cv2.boundingRect(contour)
    return origin[1] * cols


def sort_contours(questionCnts,method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(qcnt) for qcnt in questionCnts]
    boundingBoxes = np.array([(x, y, x+w, y+h) for (x, y, w, h) in boundingBoxes])
    (questionCnts, boundingBoxes) = zip(*sorted(zip(questionCnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (questionCnts, boundingBoxes)


ANSWER_KEY = {
    0: 0,
    1: 1,
    2: 3,
    3: 0,
    4: 0,
    5: 1,
    }

questionCnts = sort_contours(questionCnts,method="top-to-bottom")[0]
correct = 0
for (q, i) in enumerate(np.arange(0, len(questionCnts), 4)):
    new = sort_contours(questionCnts[i:i + 4])[0]
    bubbled = None
    for (j, c) in enumerate(new):
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        total = cv2.countNonZero(mask)

        if bubbled is None or total > bubbled[0]:
            bubbled = (total, j)
    color = (0, 0, 255)
    k = ANSWER_KEY[q]
	# check to see if the bubbled answer is correct
    if k == bubbled[1]:
        color = (0, 255, 0)
        correct += 1
	# draw the outline of the correct answer on the test
    cv2.drawContours(output, [new[k]], -1, color, 3)
cv2.imshow("Exam", output)


# a = sorted(questionCnts, key=lambda x: get_contour_precedence(
#     x, thresh.shape[1]))

# for i in range(len(a)):
#     output = cv2.putText(output.copy(), str(i), cv2.boundingRect(a[i])[:2], cv2.FONT_HERSHEY_COMPLEX, 1, [125])
# cv2.imshow('test', output)

# boundingBoxes = [cv2.boundingRect(qcnt) for qcnt in questionCnts]
# boundingBoxes = np.array([(x, y, x+w, y+h) for (x, y, w, h) in boundingBoxes])
# print('boundingBoxes', boundingBoxes[:2])


# questionCnts.sort(key=lambda x: get_contour_precedence(x, thresh.shape[1]))



# for (idx, vl) in enumerate(np.arange(0, len(a), 4)):
# test = cv2.drawContours(output.copy(), a, -1, (0, 255, 0), 3)
# cv2.imshow('test', test)
    # print(idx,vl)
# test = cv2.drawContours(output.copy(), a[i:4], -1, (0, 255, 0), 3)
# cv2.imshow('test', test)
# for (q, i) in enumerate(np.arange(0, len(a),4)):
#     correct = 0
#     bubbled = None
#     for (j, c) in enumerate(a):
#         mask = np.zeros(thresh.shape, dtype="uint8")
#         cv2.drawContours(mask, [c], -1, 255, -1)
#         mask = cv2.bitwise_and(thresh, thresh, mask=mask)
#         total = cv2.countNonZero(mask)
#         if bubbled is None or total > bubbled[0]:
#             bubbled = (total, j)
#        	color = (0, 0, 255)
#         k = ANSWER_KEY[q]
#        	if k == bubbled[1]:
#             color = (0, 255, 0)
#             correct += 1
#        	cv2.drawContours(output, [a[k]], -1, color, 3)
# cv2.imshow("Exam", output)

# j la index c la data cua index do
# print('J va C',j,c)
# mask = np.zeros(thresh.shape, dtype="uint8")
# cv2.drawContours(mask, [c], -1, 255, -1)

# questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]


# for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
#     contours2 = contours.sort_contours(questionCnts[i:i + 5])[0]
#     bubbled=None
#     for (j, c) in enumerate(contours2):
#         mask = np.zeros(thresh.shape, dtype="uint8")
#         cv2.drawContours(mask, [c], -1, 255, -1)
#         mask = cv2.bitwise_and(thresh, thresh, mask=mask)
#         total = cv2.countNonZero(mask)
#         if bubbled is None or total > bubbled[0]:
#             bubbled = (total, j)
#             color = (0, 0, 255)
#             k = ANSWER_KEY[q]
#             # check to see if the bubbled answer is correct
#             if k == bubbled[1]:
#                 color = (0, 255, 0)
#                 correct += 1
#             # draw the outline of the correct answer on the test
#             cv2.drawContours(paper, [cnts[k]], -1, color, 3)


cv2.waitKey()
