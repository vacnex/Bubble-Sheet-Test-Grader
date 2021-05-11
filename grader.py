import cv2
import numpy as np
import imutils.perspective
import imutils
from PIL import Image

# 1: 1 col 4 ans
# 2: 1 col 5 ans
# 3: 2 col 4 ans
# 4: 2 col 5 ans

def main():
    preProcess_img = preProcess(7)
    biggestCnts = findBiggestCnts(preProcess_img)
    cropBigest = cropBiggestCnts(biggestCnts, preProcess_img[1])
    threshCropped = threshCroppedImage(cropBigest)
    threshCnts = findThreshCnts(threshCropped, cropBigest)
    questionCnts = findQuestionCnts(threshCnts, cropBigest)
    # h = cropBigest.shape[0]
    # w = cropBigest.shape[1]
    # cv2.imshow("cropped", cropBigest[0:h, 0: round(w / 2)])
    # cv2.imshow("cropped2", cropBigest[0:h, round(w/2):w])
    Debug(questionCnts, cropBigest)
    showResult(questionCnts, threshCropped, cropBigest)

# region ANSWER_TYPE
def ANSWER_TYPE():
    ANSWER_TYPE_1 = {
        0: 2,
        1: 4,
        2: 0,
        3: 2,
        4: 1,
    }
    ANSWER_TYPE_2 = {
        0: 2,
        1: 3,
        2: 0,
        3: 2,
        4: 1,
    }
    ANSWER_TYPE_6 = {
        0: 0,
        1: 3,
        2: 1,
        3: 0,
        4: 3,
        5: 0,
        6: 3,
        7: 0,
        8: 2,
        9: 3,
    }
    return ANSWER_TYPE_1, ANSWER_TYPE_2, ANSWER_TYPE_6
# endregion

# region showResult
def showResult(questionCnts, threshCropped, cropBigest, Type=1):
    correctAnswer = ANSWER_TYPE()
    if (Type == 1):
        drawCorectAnswer(
            correctAnswer[2], questionCnts, threshCropped, cropBigest)
    elif (Type == 2):
        drawCorectAnswer(
            correctAnswer[1], questionCnts, threshCropped, cropBigest,2)
# endregion

# region preProcess
def preProcess(image, view_Result=False):
    switcher = {
        1: 'Image/bs1.jpg',
        2: 'Image/bs2.png',
        3: 'Image/bs3.png',
        4: 'Image/bs4.jpg',
        5: 'Image/bs5.jpg',
        6: 'Image/bs6.png',
        7: 'Image/bs7.png'
    }
    sel_image = switcher.get(image, "No Image")
    org_img = cv2.imread(sel_image)
    gray_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    edged_img = cv2.Canny(blur_img, 50, 100)
    if view_Result:
        cv2.imshow("PreProcess_Result", edged_img)
        return edged_img, org_img
    else:
        return edged_img, org_img
# endregion

# region findBiggestCnts
def findBiggestCnts(image, view_Result=False):
    contours, hierarchy = cv2.findContours(
        image[0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_cnt = [cv2.contourArea(cnt) for cnt in contours]
    area_sort = np.argsort(area_cnt)[::-1]
    cnts = contours[area_sort[0]]
    p = cv2.arcLength(cnts, True)
    r = cv2.approxPolyDP(cnts, 0.01*p, True)
    if view_Result:
        cv2.drawContours(image[1], [r], -1, (0, 0, 255), 3)
        cv2.imshow("BigestContour_Result", image[1])
        return r
    else:
        return r
# endregion

# region cropBiggestCnts
def cropBiggestCnts(Biggestcnts, image, view_Result=False):
    r = Biggestcnts.reshape(4, 2)
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
    if view_Result:
        cv2.imshow("CroppedBigestContour_Result", output)
        return output
    else:
        return output
# endregion

# region threshCroppedImage
def threshCroppedImage(image, view_Result=False):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(
        gray_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if view_Result:
        cv2.imshow("CroppedBigestContour_Result", thresh)
        return thresh
    else:
        return thresh
# endregion

# region findThreshCnts
def findThreshCnts(thresh_Image, cropped_image,  view_Result=False):
    contours, hierarchy = cv2.findContours(
        thresh_Image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if view_Result:
        cv2.drawContours(cropped_image, contours, -1, (0, 0, 255), 3)
        cv2.imshow("BigestContour_Result", cropped_image)
        return contours
    else:
        return contours
# endregion

# region findQuestionCnts
def findQuestionCnts(thesh_cnts, cropped_image, view_Result=False):
    questionCnts = []
    for c in thesh_cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        if w >= 20 and h >= 20 and ar >= 0.8 and ar <= 1.2:
            questionCnts.append(c)
    if view_Result:
        cv2.drawContours(cropped_image, questionCnts, -1, (0, 255, 0), 3)
        cv2.imshow("QuestionCnts_Result", cropped_image)
        return questionCnts
    else:
        return questionCnts
# endregion

# region get_contour_precedence
def get_contour_precedence(contour, cols):
    tolerance_factor = 10
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]
# endregion

# region drawCorectAnswer
def drawCorectAnswer(ANSWER_KEY, question_Cnts, thresh_Image, org_Image, Exam_type=1):
    ans_num = 0
    col = 0
    if Exam_type == 1:
        ans_num = 4
        col = 1
    elif Exam_type == 2:
        ans_num = 5
        col = 1
    # elif Exam_type == 3:
    #     ans_num = 5
    #     col = 1
    # elif Exam_type == 4:
    #     ans_num = 5
    #     col = 2

    question_Cnts.sort(
        key=lambda x: get_contour_precedence(x, thresh_Image.shape[1]))
    correct = 0

    for (q, i) in enumerate(np.arange(0, len(question_Cnts), ans_num)):
        bubbled = None
        AnsCnts = question_Cnts[i:i + ans_num]
        for (j, c) in enumerate(AnsCnts):
            mask = np.zeros(thresh_Image.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_and(thresh_Image, thresh_Image, mask=mask)
            total = cv2.countNonZero(mask)
            if bubbled is None or total > bubbled[0]:
                bubbled = (total, j)
        color = (0, 0, 255)
        k = ANSWER_KEY[q]
        if k == bubbled[1]:
            color = (0, 255, 0)
            correct += 1
        cv2.drawContours(org_Image, [AnsCnts[k]], -1, color, 3)
    cv2.imshow('Result', org_Image)
# endregion

# region Debug
def Debug(Cnts, image, sort=True):
    img = image.copy()
    if (sort == False):
        for i in range(len(Cnts)):
            cv2.putText(img, str(i),
            cv2.boundingRect(Cnts[i])[:2],
            cv2.FONT_HERSHEY_COMPLEX, 1,[125])
            cv2.imshow("Deubug", img)
    else:
        Cnts.sort(
            key=lambda x: get_contour_precedence(x, image.shape[1]))
        for i in range(len(Cnts)):
            cv2.putText(img, str(i), cv2.boundingRect(Cnts[i])[
                :2], cv2.FONT_HERSHEY_COMPLEX, 1, [125])
            cv2.imshow("Debug", img)
# endregion

main()
cv2.waitKey()
