import cv2
import mediapipe as mp
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose


pyautogui.FAILSAFE = False


FIST_THRESHOLD = 0.7
LEFT_THUMB_UP_THRESHOLD = 0.8
RIGHT_THUMB_UP_THRESHOLD = 0.8
CONTROL_MAP = {
    'w': mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
    's': mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
    'a': mp.solutions.pose.PoseLandmark.LEFT_HIP,
    'd': mp.solutions.pose.PoseLandmark.RIGHT_HIP,
    'space': mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
    'shift': mp.solutions.pose.PoseLandmark.LEFT_WRIST,
}

pose_model = mp_pose.Pose(min_detection_confidence=0.5,
min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.2, min_tracking_confidence=0.2) as pose_model:
    with mp_hands.Hands(min_detection_confidence=0.2, min_tracking_confidence=0.2, max_num_hands=1) as hand_model:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose_model.process(image)

            pose_results = pose_model.process(image)
            hand_results = hand_model.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            if hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

            if hand_results.multi_hand_landmarks:
                hand_landmarks = hand_results.multi_hand_landmarks[0]
                thumb_up = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y < hand_landmarks.landmark[
                    mp_hands.HandLandmark.INDEX_FINGER_TIP].y
                if thumb_up and hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x:
                    pyautogui.press('space')
                elif thumb_up and hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x:
                    pyautogui.press('shift')
     
            for key, value in CONTROL_MAP.items():
                if results.pose_landmarks and results.pose_landmarks.landmark[value].visibility > 0.5:
                    pyautogui.keyDown(key)
                else:
                    pyautogui.keyUp(key)

            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imshow('Fall Guys Controller',cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(10) == ord('q'):
                break


cv2.destroyAllWindows()
cap.release()
cv2.waitKey('q')