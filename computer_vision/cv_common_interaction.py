import computer_vision.computer_vision_bodytracking_interactions as gesture_tracking


class ComputerVisionCommonInteractions():

    def __init__(self, webcam_image, webcam_rgb_image):
        self.webcam_image = webcam_image
        self.webcam_rgb_image = webcam_rgb_image

        self.hand_tracking_interaction = gesture_tracking.HandDetactor(self.webcam_image, self.webcam_rgb_image)
