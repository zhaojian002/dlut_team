import cv2
import numpy as np
def check_license_plate(over_speed_num):
    ball_color = 'blue'

    color_dist = {'red': {'Lower': np.array([0, 60, 60]), 'Upper': np.array([6, 255, 255])},
                  'blue': {'Lower': np.array([100, 80, 46]), 'Upper': np.array([124, 255, 255])},
                  'green': {'Lower': np.array([35, 43, 35]), 'Upper': np.array([90, 255, 255])},
                  }

    img = cv2.imread("cut"+str(over_speed_num)+".jpg")
    frame = img

    cv2.namedWindow('camera', cv2.WINDOW_AUTOSIZE)

    if frame is not None:
        gs_frame = cv2.GaussianBlur(frame, (5, 5), 0)  # 高斯模糊
        hsv = cv2.cvtColor(gs_frame, cv2.COLOR_BGR2HSV)  # 转化成HSV图像
        erode_hsv = cv2.erode(hsv, None, iterations=2)  # 腐蚀 粗的变细
        inRange_hsv = cv2.inRange(erode_hsv, color_dist[ball_color]['Lower'], color_dist[ball_color]['Upper'])
        cnts = cv2.findContours(inRange_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        c = max(cnts, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        # cv2.drawContours(frame, [np.int0(box)], -1, (0, 255, 255), 2)

        cut_frame = frame[int(np.min(box[:, 1])):int(np.max(box[:, 1])), int(np.min(box[:, 0])):int(np.max(box[:, 0]))]
        cv2.imwrite('test.jpg', cut_frame)
        cv2.imshow('camera', frame)
        #cv2.waitKey(1)

    else:
        print("无画面")

    #cv2.waitKey(0)
    cv2.destroyAllWindows()

