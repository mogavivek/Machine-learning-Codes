import cv2
import mediapipe as mp
import time 


class HandDetactor():

    def __init__(self, webcam_image, webcam_rgb_image, mode=False, max_hands=2, model_complexity = 1, detection_concentration=0.5, track_contentration=0.5):
        """
        This class interact all hand tracking activities functions or objects\n
        
        Argument:
            webcam_image : Apply input variable as a webcam image or video
            webcam_rgb_image : Apply input of rgb color image or video from the webcame
            mode : Whether to treat the input images as a batch of static
            max_hands : It takes the input of the number of hands require
            model_complexity : Complexity of the hand landmark model: 0 or 1.
            detection_concentration : Minimum confidence value ([0.0, 1.0]) for hand detection to be considered successful
            track_contentration : Minimum confidence value ([0.0, 1.0]) for the hand landmarks to be considered tracked successfully
        """
        self.webcam_image = webcam_image
        self.webcam_rgb_image = webcam_rgb_image 
        self.mode = mode
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.detection_concentration = detection_concentration
        self.track_contentration = track_contentration

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max_hands, self.model_complexity, self.detection_concentration, self.track_contentration)
        self.mp_draw = mp.solutions.drawing_utils 
    
    def hand_dots_tracking_on_hand(self, draw=True):
        '''This function draw the 21 dots on hands which are visible in webcame\n
        Argument:
            draw : If draw is True then it will show the dots on the hands\n
        Returns:
            webcam_image : It returns the webcame image
        '''
        try:
            self.results = self.hands.process(self.webcam_rgb_image)
            if(self.results.multi_hand_landmarks):
                for hand_landmarks in self.results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(self.webcam_image, hand_landmarks)
            return self.webcam_image
                
        except:
            return Exception("Failed to load the hand tracking process")

    def hand_dots_and_lines(self, draw = True):
        '''This function draw the lines of 21 dots in hands which are visible in webcame\n
        Argument:
            draw : If draw is True then it will show the dots and lines on the hands\n
        Returns:
            webcam_image : It returns the webcame image
        '''
        try:
            self.results = self.hands.process(self.webcam_rgb_image)
            if(self.results.multi_hand_landmarks):
                for hand_landmarks in self.results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(self.webcam_image, hand_landmarks, self.mpHands.HAND_CONNECTIONS)
            return self.webcam_image
        except:
            return Exception("Failed to load the hand tracking process")
    
    def find_dots_mark_position(self, hand_numebr=0, draw=True):
        '''
        This function take the positions value with id numbers and return the list\n
        Arguments:
            hand_numebr : Provide the number of hands
            draw : If draw is True then it will show the dots and lines on the hands
        Return:
            land_mark_list : It return the landmark list of the all 21 dots with its id number, center_x and center_y
        '''
        land_mark_list = []
        try:
            if(self.results.multi_hand_landmarks):
                my_hand = self.results.multi_hand_landmarks[hand_numebr]

                for id, land_mark in enumerate(my_hand.landmark):
                    height, width, channel = self.webcam_image.shape
                    center_x, center_y = int(land_mark.x*width), int(land_mark.y*height)
                    land_mark_list.append([id, center_x, center_y])
            return land_mark_list
        except:
            return Exception("Failed to detect the positions of the dots")        

    def check_dots_showing_correct_value(self, provide_dot_number):
        '''
        This function takes the value of the hand tracking all dots with its number\n
        Then checks that the whether it is showing correct value
        
        Argument:
            provide_dot_number : It takes the input from number 1 to 21 (because we have 21 tracking dots on hand)
        '''
        try:
            my_hand = self.results.multi_hand_landmarks
            for id, land_mark in enumerate(my_hand.landmark):
                height, width, channel = self.webcam_image.shape
                center_x, center_y = int(land_mark.x*width), int(land_mark.y*height)
            if id == provide_dot_number:
                cv2.circle(self.webcam_image, (center_x, center_y), 20, (255, 0, 255), cv2.FILLED)
        except:
            return Exception("Failed to locate the circle of selected point on hand")


