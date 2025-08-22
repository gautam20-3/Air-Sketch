# ================== Air Sketch Project ==================
# Imports
import cv2
import numpy as np
import mediapipe as mp
from collections import deque

def main():
    # Deques to handle color points
    bpoints = [deque(maxlen=1024)]
    cpoints = [deque(maxlen=1024)]
    gpoints = [deque(maxlen=1024)]
    opoints = [deque(maxlen=1024)]

    # Index counters for colors
    black_index, blue_index, green_index, orange_index = 0, 0, 0, 0

    # Kernel for morphological operations (not used heavily here but good for expansion)
    kernel = np.ones((5, 5), np.uint8)

    # Define colors: Black, Cyan, Green, Orange
    colors = [(0, 0, 0), (150, 75, 0), (0, 255, 0), (0, 165, 255)]
    colorIndex = 0

    # ================== Sketch Setup ==================
    paintWindow = np.zeros((471, 636, 3), dtype=np.uint8) + 255
    paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), cv2.FILLED)
    paintWindow = cv2.rectangle(paintWindow, (160, 1), (255, 65), (0, 0, 0), cv2.FILLED)
    paintWindow = cv2.rectangle(paintWindow, (275, 1), (370, 65), (150, 75, 0), cv2.FILLED)
    paintWindow = cv2.rectangle(paintWindow, (390, 1), (485, 65), (0, 255, 0), cv2.FILLED)
    paintWindow = cv2.rectangle(paintWindow, (505, 1), (600, 65), (0, 165, 255), cv2.FILLED)

    cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "BLACK", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "CYAN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "GREEN", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "ORANGE", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

    # ================== Mediapipe Initialization ==================
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.6)
    mpDraw = mp.solutions.drawing_utils

    # ================== Webcam Initialization ==================
    cap = cv2.VideoCapture(0)
    ret = True

    while ret:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Draw menu buttons on frame
        frame = cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), cv2.FILLED)
        frame = cv2.rectangle(frame, (160, 1), (255, 65), (0, 0, 0), cv2.FILLED)
        frame = cv2.rectangle(frame, (275, 1), (370, 65), (150, 75, 0), cv2.FILLED)
        frame = cv2.rectangle(frame, (390, 1), (485, 65), (0, 255, 0), cv2.FILLED)
        frame = cv2.rectangle(frame, (505, 1), (600, 65), (0, 165, 255), cv2.FILLED)

        cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "BLACK", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "CYAN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "GREEN", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "ORANGE", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2, cv2.LINE_AA)

        # Mediapipe hand tracking
        result = hands.process(framergb)

        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    lmx, lmy = int(lm.x * 640), int(lm.y * 480)
                    landmarks.append([lmx, lmy])

                # Draw hand landmarks
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            fore_finger = (landmarks[8][0], landmarks[8][1])
            center = fore_finger
            thumb = (landmarks[4][0], landmarks[4][1])
            cv2.circle(frame, center, 3, (0, 255, 0), -1)

            # Gesture for new stroke
            if (thumb[1] - center[1] < 30):
                bpoints.append(deque(maxlen=512))
                black_index += 1
                cpoints.append(deque(maxlen=512))
                blue_index += 1
                gpoints.append(deque(maxlen=512))
                green_index += 1
                opoints.append(deque(maxlen=512))
                orange_index += 1

            elif center[1] <= 65:  # Menu selection
                if 40 <= center[0] <= 140:  # Clear
                    bpoints, cpoints, gpoints, opoints = [deque(maxlen=512)], [deque(maxlen=512)], [deque(maxlen=512)], [deque(maxlen=512)]
                    black_index, blue_index, green_index, orange_index = 0, 0, 0, 0
                    paintWindow[67:, :, :] = 255
                elif 160 <= center[0] <= 255:
                    colorIndex = 0  # Black
                elif 275 <= center[0] <= 370:
                    colorIndex = 1  # Cyan
                elif 390 <= center[0] <= 485:
                    colorIndex = 2  # Green
                elif 505 <= center[0] <= 600:
                    colorIndex = 3  # Orange
            else:
                if colorIndex == 0:
                    bpoints[black_index].appendleft(center)
                elif colorIndex == 1:
                    cpoints[blue_index].appendleft(center)
                elif colorIndex == 2:
                    gpoints[green_index].appendleft(center)
                elif colorIndex == 3:
                    opoints[orange_index].appendleft(center)

        else:  # If no hand detected, add new deques
            bpoints.append(deque(maxlen=512))
            black_index += 1
            cpoints.append(deque(maxlen=512))
            blue_index += 1
            gpoints.append(deque(maxlen=512))
            green_index += 1
            opoints.append(deque(maxlen=512))
            orange_index += 1

        # Draw lines on frame & canvas
        points = [bpoints, cpoints, gpoints, opoints]
        for i in range(len(points)):
            for j in range(len(points[i])):
                for k in range(1, len(points[i][j])):
                    if points[i][j][k - 1] is not None and points[i][j][k] is not None:
                        cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                        cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

        # Show windows
        cv2.imshow("Output", frame)
        cv2.imshow("Paint", paintWindow)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

