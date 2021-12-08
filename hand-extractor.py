import cv2
vidcap = cv2.VideoCapture('A.mp4')
success,image = vidcap.read() # read video
length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) #get the number of frames in the video
print( length )
vidcap.set(1, length/2) #set video variable at middle frame
ret, frame = vidcap.read() #capture middle frame
cv2.imwrite("middle-frame.jpg", frame) #save middle frame


import mediapipe as mp
import math
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=True,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = hands.process(imgRGB)
print('Handedness:', results.multi_handedness)
if(results.multi_hand_landmarks):
    image_height, image_width, _ = frame.shape
    annotated_image = frame.copy()
    for hand_landmarks in results.multi_hand_landmarks:
        print(hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_MCP].x * image_width)
        wrist_x = hand_landmarks.landmark[mpHands.HandLandmark.WRIST].x * image_width
        wrist_y = hand_landmarks.landmark[mpHands.HandLandmark.WRIST].y * image_height
        print("wrist-x:",wrist_x)
        print("wrist-y:",wrist_y)
        print("wrist-x raw:",hand_landmarks.landmark[mpHands.HandLandmark.WRIST].x)
        print("wrist-y raw:",hand_landmarks.landmark[mpHands.HandLandmark.WRIST].y)
        print("image height:", image_height)
        print("image width:", image_width)
        crop_image_x_left = max(0,math.ceil(wrist_x) - 300)
        print('crop_image_x_left:',crop_image_x_left)
        crop_image_x_right = min(image_width,math.ceil(wrist_x) + 300)
        print('crop_image_x_right',crop_image_x_right)
        crop_image_y_bottom = max(0,math.ceil(wrist_y) - 550)
        print('crop_image_y_bottom',crop_image_y_bottom)
        crop_image_y_top = min(image_height,math.ceil(wrist_y) + 200)
        print('crop_image_y_top',crop_image_y_top)
        cropped_image = frame[crop_image_y_bottom:crop_image_y_top, crop_image_x_left:crop_image_x_right]
        print(cropped_image)
        cv2.imwrite('cropped_image.jpg', cv2.flip(cropped_image, 1))
        mpDraw.draw_landmarks(annotated_image,hand_landmarks,mpHands.HAND_CONNECTIONS,mp_drawing_styles.get_default_hand_landmarks_style(),mp_drawing_styles.get_default_hand_connections_style())
    cv2.imwrite('annotataed_image.jpg', cv2.flip(annotated_image, 1))
