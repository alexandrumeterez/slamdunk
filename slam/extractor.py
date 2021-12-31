import cv2

class Frame(object):
    def __init__(self, frame, detector, matcher):
        self.frame = frame
        self.detector = detector
        self.matcher = matcher
        self.keypoints = None
        self.descriptors = None
        self.__extract_keypoints()

    def __extract_keypoints(self):
        kp, des = self.detector.detectAndCompute(self.frame, None)
        self.keypoints = kp
        self.descriptors = des

    def get_matches_with(self, other):
        matches = self.matcher.knnMatch(self.descriptors, other.descriptors, k=2)
        return matches

