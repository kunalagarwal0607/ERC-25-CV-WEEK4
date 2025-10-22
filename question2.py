import cv2
import numpy as np
import mediapipe as mp

# MediaPipe Hands initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

canvas = None
drawing = False
pen_color = (0, 255, 0)  # Green
pen_thickness = 5
eraser_mode = False
line_style = cv2.LINE_AA  # Default line style
eraser_radius = 30         # Eraser size

def switch_color(key):
    # R, G, B, Black
    return {
        ord('r'): (0, 0, 255),    # Red
        ord('g'): (0, 255, 0),    # Green
        ord('b'): (255, 0, 0),    # Blue
        ord('k'): (0, 0, 0)       # Black
    }.get(key, pen_color)

cap = cv2.VideoCapture(0)
prev_x, prev_y = None, None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    if canvas is None:
        canvas = np.zeros_like(frame)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    h, w, _ = frame.shape

    # Track index fingertip
    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]
        lm = handLms.landmark[8]
        x, y = int(lm.x * w), int(lm.y * h)
        if drawing:
            if eraser_mode:
                # Erase by drawing a circle with the background color (black)
                cv2.circle(canvas, (x, y), eraser_radius, (0, 0, 0), -1)
                prev_x, prev_y = None, None  # Don't draw lines in eraser mode
            else:
                if prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), (x, y), pen_color, pen_thickness, line_style)
                prev_x, prev_y = x, y
        else:
            prev_x, prev_y = None, None
        mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
    else:
        prev_x, prev_y = None, None

    # Blend drawing with live frame
    blended = cv2.addWeighted(frame, 0.7, canvas, 0.7, 0)

    # Instructions
    cv2.putText(blended, 'Hold "p" to draw | "e" for eraser | r/g/b/k for color | "c" clear | "q" quit',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 255), 2, line_style)

    cv2.imshow('Drawing Pad', blended)
    key = cv2.waitKey(1) & 0xFF

    # Controls
    if key == ord('q'):
        break
    elif key == ord('p'):
        drawing = True
        eraser_mode = False
    elif key == ord('e'):
        drawing = True
        eraser_mode = True
    elif key in [ord('r'), ord('g'), ord('b'), ord('k')]:
        pen_color = switch_color(key)
        eraser_mode = False
    elif key == ord('c'):
        canvas[:] = 0
        drawing = False

    if not key in [ord('p'), ord('e')]:
        drawing = False  # Only draw when button held

cap.release()
cv2.destroyAllWindows()
