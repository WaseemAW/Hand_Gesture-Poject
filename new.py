import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import time, math, numpy as np
# import HandTrackingModule as htm
import mediapipe as mp
import pyautogui, autopy
from datetime import datetime


wCam, hCam = 640,480
frameR = 100 # Frame Reduction
cap = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture('ScreenSaver/Saver_2.mp4')
cap.set(3, wCam)
cap.set(4, hCam)
#cap1.set(3, wCam)
#cap1.set(4, hCam)
smoothening = 7
#########################
pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0
w, h = autopy.screen.size()


# cTime = 0
class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True, color=(255, 0, 255), z_axis=False):

        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                #   print(id, lm)
                h, w, c = img.shape
                if z_axis == False:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # print(id, cx, cy)
                    lmList.append([id, cx, cy])
                elif z_axis:
                    cx, cy, cz = int(lm.x * w), int(lm.y * h), round(lm.z, 3)
                    # print(id, cx, cy, cz)
                    lmList.append([id, cx, cy, cz])

                if draw:
                    cv2.circle(img, (cx, cy), 5, color, cv2.FILLED)

        return lmList


detector = handDetector(maxHands=1, detectionCon=0.85, trackCon=0.8)

tipIds = [4, 8, 12, 16, 20]
mode = ''
active = 0

pyautogui.FAILSAFE = False
flag1 = True
flag2 = False
flag_video = False
start_time = datetime.now()
diff = (datetime.now() - start_time).seconds # converting into seconds
frame_counter = 0
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    # print(lmList)
    fingers = []

    if len(lmList) != 0:
        start_time = datetime.now()
        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0 - 1]][1]:
            if lmList[tipIds[0]][1] >= lmList[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        elif lmList[tipIds[0]][1] < lmList[tipIds[0 - 1]][1]:
            if lmList[tipIds[0]][1] <= lmList[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        #  print(fingers)
        if (fingers == [0, 0, 0, 0, 0]) & (active == 0):
            mode = 'N'
        elif (fingers == [1, 1, 1, 1, 1]) & (active == 0):
            mode = 'Cursor'
            active = 1

    ############# Scroll 👇👇👇👇##############
        if mode == 'N':
            active = 0
            #   print(mode)
            putText(mode)
            cv2.rectangle(img, (200, 410), (245, 460), (255, 255, 255), cv2.FILLED)
            if len(lmList) != 0:
                if fingers == [0, 0, 0, 0, 0]:
                    putText(mode='Select', loc=(200, 455), color=(0, 255, 0))
                    pyautogui.click()
                    time.sleep(0.5)
                if fingers == [0, 1, 1, 0, 0]:
                    putText(mode='D', loc=(200, 455), color=(0, 0, 255))
                elif fingers == [0, 0, 0, 0, 0]:
                    active = 0
                    mode = 'N'
                    pyautogui.click()
                    #time.sleep(0.5)
    if mode == 'Cursor':
        active = 1
        putText(mode)
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                      (255, 0, 255), 2)
        if fingers[1:] == [0, 0, 0, 0]:  # thumb excluded
            active = 0
            mode = 'N'
            print(mode)
        else:
            if len(lmList) != 0:

                x1, y1 = lmList[8][1], lmList[8][2]

                x3 = np.interp(x1, (frameR, wCam - frameR), (0, w))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, h))
#                print("After min")
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening

                # 7. Move Mouse

                autopy.mouse.move(w - clocX, clocY)
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                plocX, plocY = clocX, clocY

                #  pyautogui.moveTo(X,Y)
                if fingers[0] == 0:
                    cv2.circle(img, (lmList[4][1], lmList[4][2]), 10, (0, 0, 255), cv2.FILLED)  # thumb
                    pyautogui.click()
                    time.sleep(0.5)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS:{int(fps)}', (480, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
    diff = (datetime.now() - start_time).seconds  # converting into seconds

#    cv2.imshow('Hand LiveFeed', img)
    if diff < 20:
        cv2.imshow('Hand LiveFeed', img)
        cv2.setWindowProperty('Hand LiveFeed', cv2.WND_PROP_TOPMOST, 1)
        flag1 = True
        if flag2:
            cv2.destroyWindow('Saver')
            flag2 = False
    else:
        if flag1:
            flag1 = False
        if frame_counter == cap1.get(cv2.CAP_PROP_FRAME_COUNT):
            frame_counter = 0
#            cap1 = cv2.VideoCapture('ScreenSaver/Saver.mp4')
            if (flag_video):
                cap1 = cv2.VideoCapture('ScreenSaver/Saver_2.mp4')
                flag_video = False
            else:
                cap1 = cv2.VideoCapture('ScreenSaver/Saver_1.mp4')
                flag_video = True
        #cv2.destroyWindow('Hand LiveFeed')
        success1, img1 = cap1.read()
        frame_counter += 1
        cv2.namedWindow('Saver', cv2.WINDOW_NORMAL)
        img1 = cv2.resize(img1, (1080, 1920))
        cv2.imshow('Saver', img1)
        cv2.setWindowProperty('Saver', cv2.WND_PROP_TOPMOST, 1)
        cv2.moveWindow('Saver', 5, 5)
        flag2 = True
        if cv2.waitKey(1) & 0xFF == ord('s'):
            #        break
            cv2.destroyWindow('Saver')
            start_time = datetime.now()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

    def putText(mode, loc=(250, 450), color=(0, 255, 255)):
        cv2.putText(img, str(mode), loc, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    3, color, 3)
