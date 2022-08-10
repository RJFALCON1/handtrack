import mediapipe as mp
import cv2
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8,
                       min_tracking_confidence=0.5)
                
def drawHandmarks(img,handPositions) :
    if handPositions:
        for x in handPositions:
            mp_drawing.draw_landmarks(img,x,mp_hands.HAND_CONNECTIONS)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    allHands = hands.process(img)
    handPositions = allHands.multi_hand_landmarks
    drawHandmarks(img,handPositions)
    cv2.imshow('hand',img)
    if (cv2.waitKey(25)==32) :
        break
cv2.destroyAllWindows()