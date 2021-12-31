import cv2
import sys
from extractor import Frame
from geometry import ratio_test, get_R_t
if __name__ == '__main__':
    sift = cv2.SIFT_create()
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)

    cap = cv2.VideoCapture(sys.argv[1])
    if (cap.isOpened() == False):
        print("Error opening cap")
    
    frame_count = 0
    matches = None


    while(cap.isOpened()):
        ret, frame = cap.read()
        frame_count += 1
        f = Frame(frame, sift, flann)
        if frame_count > 1:
            matches = f.get_matches_with(prev_frame)
            pts1, pts2 = ratio_test(matches, f.keypoints, prev_frame.keypoints, 0.8)
            F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
            R, t = get_R_t(F)
            print(R)
            print(t)


        frame = cv2.drawKeypoints(frame, f.keypoints, None, color=(0, 255, 0))

        if ret == True:
            cv2.imshow("Frame", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        prev_frame = f


    cap.release()
    cv2.destroyAllWindows()