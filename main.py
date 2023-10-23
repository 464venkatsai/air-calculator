
import mediapipe as mp
import cv2
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
# model = joblib.load('digit_classifier.h5')
cap = cv2.VideoCapture(0)

hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)


def distance(point1, point2):
    return ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2) ** 0.5


def fingers_landmarks(hand):
    thumbpoint1 = hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumbpoint2 = hand.landmark[mp_hands.HandLandmark.THUMB_IP]
    thumbpoint3 = hand.landmark[mp_hands.HandLandmark.THUMB_MCP]
    thumbpoint4 = hand.landmark[mp_hands.HandLandmark.THUMB_CMC]
    indexpoint1 = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    indexpoint2 = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
    indexpoint3 = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    indexpoint4 = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middlepoint1 = hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middlepoint2 = hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
    middlepoint3 = hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    middlepoint4 = hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ringpoint1 = hand.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ringpoint2 = hand.landmark[mp_hands.HandLandmark.RING_FINGER_DIP]
    ringpoint3 = hand.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
    ringpoint4 = hand.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    littlepoint1 = hand.landmark[mp_hands.HandLandmark.PINKY_TIP]
    littlepoint2 = hand.landmark[mp_hands.HandLandmark.PINKY_DIP]
    littlepoint3 = hand.landmark[mp_hands.HandLandmark.PINKY_PIP]
    littlepoint4 = hand.landmark[mp_hands.HandLandmark.PINKY_MCP]
    wrist = hand.landmark[mp_hands.HandLandmark.WRIST]
    return [
        thumbpoint1,
        thumbpoint2,
        thumbpoint3,
        thumbpoint4,
        indexpoint1,
        indexpoint2,
        indexpoint3,
        indexpoint4,
        middlepoint1,
        middlepoint2,
        middlepoint3,
        middlepoint4,
        ringpoint1,
        ringpoint2,
        ringpoint3,
        ringpoint4,
        littlepoint1,
        littlepoint2,
        littlepoint3,
        littlepoint4,
        wrist,
    ]


def check(list_name, symbol):
    return (
        all(
            [
                True if dist <= list_name[i] else False
                for i, dist in enumerate(distances)
            ]
        )
        if symbol == "<="
        else all(
            [
                True if dist >= list_name[i] else False
                for i, dist in enumerate(distances)
            ]
        )
    )

def Draw_Line(records):
    i = 0
    while True:
        if i>= len(records)-1:
            break
        else:
            cv2.line(image, (records[i][0], records[i][1]),(records[i+1][0], records[i+1][1]), (0, 0, 255), 2)
        i=i+1
    return     

def erase_points(records, center, key):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, center, (65, 50), 190, 0, 360, (170,170,170), -1)
    cv2.ellipse(image, center, (65, 50), 190, 0, 360, (170, 170, 170), -1)
    return [point for point in records if mask[point[1], point[0]] == 0], True

def Crop_Rectangle(point1,point2):
    cv2.rectangle(image, point1, point2, (0, 255, 0), 2)
    croped_frame = cv2.cvtColor(cv2.resize(image[y1:y2, x1:x2], (28, 28)), cv2.COLOR_BGR2GRAY)
    croped_frame = np.expand_dims(croped_frame, axis=-1)
    return np.expand_dims(croped_frame, axis=0)

fist_close = [0.38, 0.25, 0.2, 0.19, 0.2]
fist_open = [0.24, 0.39, 0.43, 0.41, 0.35]
# point1  = (0,0)
# point2 = (0,0)

x1, y1 ,x2 ,y2 = 25, 25, 250, 250
recorded_handmarks = {'1':[],'2':[]}
end = False
key = '1'

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame,(0,0),fx=1.25,fy=1.25)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.flip(image, 1)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for num, hand in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)
            fingers = fingers_landmarks(hand)
            point1 = (int(fingers[4].x * image.shape[1]), int(fingers[4].y * image.shape[0]))
            distances = [distance(fingers[20], fingers[i]) for i in range(0,20,4)]
            if check(fist_close, "<="):
                # prediction = model.predict(rectangle1)
                # print([index for index,ele in enumerate(prediction[0]) if int(ele)==1][0])
                pass
            elif check(fist_open, ">="):
                recorded_handmarks[key],end = erase_points(records=recorded_handmarks[key],center=(int(fingers[12].x * image.shape[1]), int(fingers[12].y * image.shape[0])),key=key)
            else:
                recorded_handmarks[key].append(point1)

    Draw_Line(records=recorded_handmarks[key])
    Draw_Line(records=recorded_handmarks['1'])
    # rectangle1 = Crop_Rectangle(point1=(x1,y1),point2=(x2,y2))

    if cv2.waitKey(50) == ord("q"):
        break
    cv2.imshow("Hand Tracking", image)

cap.release()
cv2.destroyAllWindows()         