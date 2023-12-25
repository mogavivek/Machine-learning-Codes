import computer_vision.computer_vision_bodytracking_interactions as computer_vision_interaction
import cv2
import time
import os

def main():
    '''This function open the computer vision webcam and run the finger counting project'''

    #Adjusting the sccreen size of the laptop camera
    width_cam, height_cam = 640, 480
    cap = cv2.VideoCapture(0)
    cap.set(3, width_cam)
    cap.set(4, height_cam)

    #Images path
    folder_path = "C:/Users/vivek/PycharmProjects/pythonProject/Vivekcode/Machine-learning-Codes/computer_vision/fingers"
    finger_list = os.listdir(folder_path)
    #Calling the images as per stored
    overlay_list = []
    for image_path in finger_list:
        image = cv2.imread(f'{folder_path}/{image_path}')
        overlay_list.append(image)
    print(len(overlay_list))
    previous_time = 0

    #The below list is all the tip number of the fingers and thumb
    tip_ids = [4, 8, 12, 16, 20]

    
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hand_interactions = computer_vision_interaction.HandDetactor(img, imgRGB)
        img = hand_interactions.hand_dots_and_lines(draw=False)
        land_mark_list = hand_interactions.find_dots_mark_position(draw=False)

        if(len(land_mark_list) != 0):
            fingers = []
            #For thumb (x-axis counted hence 1)
            if(land_mark_list[tip_ids[0]][1] > land_mark_list[tip_ids[0]-1][1]):
                fingers.append(1)
            else:
                fingers.append(0)
            #For fingers (y-axis counted hence 2)
            for id in range(1, 5):
                if(land_mark_list[tip_ids[id]][2] < land_mark_list[tip_ids[id]-2][2]):
                    fingers.append(1)
                else:
                    fingers.append(0)
            total_fingers = fingers.count(1)
        
            #print(fingers)    
            height, width, channel = overlay_list[total_fingers].shape
            img[0:height, 0:width] = overlay_list[total_fingers]

            cv2.putText(img, str(total_fingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 5, (0, 165, 255), 10)

        current_time = time.time()
        fps = 1/(current_time - previous_time)
        previous_time = current_time 

        cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

        cv2.imshow("Image", img)
        k = cv2.waitKey(1)  
        # press 'q' to exit
        if k == ord('q'):
            break

if __name__ == "__main__":
    main()




         