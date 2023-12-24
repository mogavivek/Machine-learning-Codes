import cv2
import time 
import computer_vision_bodytracking_interactions as cv_tracking

cap = cv2.VideoCapture(0)
previous_time = 0
current_time = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    hand_tracking = cv_tracking.HandDetactor(img, imgRGB)
    img = hand_tracking.hand_dots_and_lines()    
    landmark_list = hand_tracking.find_dots_mark_position()
    hand_tracking.check_dots_showing_correct_value(4)
    if(len(landmark_list) != 0):
        print(landmark_list[4])

    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)

    cv2.imshow("Image", img)
    k = cv2.waitKey(1)
    # press 'q' to exit
    if k == ord('q'):
        break