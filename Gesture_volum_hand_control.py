import computer_vision.computer_vision_bodytracking_interactions as computer_vision_interaction
import cv2
import time
import os
import numpy as np
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

def main():
    #Adjusting the sccreen size of the laptop camera
    width_cam, height_cam = 640, 480
    cap = cv2.VideoCapture(0)
    cap.set(3, width_cam)
    cap.set(4, height_cam)
    previous_time = 0

    while True:
        success, img = cap.read()

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detactor = computer_vision_interaction.HandDetactor(img, imgRGB)

        img = detactor.hand_dots_and_lines()
        landmark_list = detactor.find_dots_mark_position(draw=False)

        volume_bar = 0
        volume_bar_screen = 400
        volume_bar_percentage = 0

        if(len(landmark_list) != 0):
            thumb_tip_x1, thumb_tip_y1 = (landmark_list[4][1], landmark_list[4][2])
            finger_tip_x1, finger_tip_y1 = (landmark_list[8][1], landmark_list[8][2])  
            center_x, center_y = (thumb_tip_x1 + finger_tip_x1)//2, (thumb_tip_y1 + finger_tip_y1)//2

            cv2.circle(img, (thumb_tip_x1, thumb_tip_y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (finger_tip_x1, finger_tip_y1), 15, (255, 0, 255), cv2.FILLED)
            #Create line inbetween the point 4 and 8
            cv2.line(img, (thumb_tip_x1, thumb_tip_y1), (finger_tip_x1, finger_tip_y1), (255, 0, 255), 3)
            cv2.circle(img, (center_x, center_y), 15, (255, 0, 255), cv2.FILLED)

            length = math.hypot((finger_tip_x1-thumb_tip_x1), (finger_tip_y1-thumb_tip_y1))

            if(length < 50):
                cv2.circle(img, (center_x, center_y), 15, (0, 165, 255), cv2.FILLED)
            minimum_volume, maximum_volume = detactor.draw_volume_bar()
            #Now our finger range is 50 - 300 and volume range is -65 - 0
            volume_bar = np.interp(length, [50,300], [minimum_volume, maximum_volume])
            volume_bar_screen = np.interp(length, [50,300], [400, 150])
            volume_bar_percentage = np.interp(length, [50,300], [0, 100])
            
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(
                IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = cast(interface, POINTER(IAudioEndpointVolume))
            volume.SetMasterVolumeLevel(volume_bar, None)
        
        cv2.rectangle(img, (50, 150), (85, 400), (0, 165, 255), 4)
        cv2.rectangle(img, (50, int(volume_bar_screen)), (85, 400), (0, 165, 255), 4)
        cv2.putText(img, f'FPS: {int(volume_bar_percentage)} %', (40, 450), cv2.FONT_HERSHEY_PLAIN, 3, (0, 165, 255), 2)

        current_time = time.time()
        fps = 1/(current_time - previous_time)
        previous_time = current_time 

        cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

        cv2.imshow("Img", img)
        k = cv2.waitKey(1)  
        # press 'q' to exit
        if k == ord('q'):
            break

if(__name__ == "__main__"):
    main()