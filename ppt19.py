import cv2
import mediapipe as mp
import pyautogui
import os
import time
from datetime import datetime
import pygetwindow as gw

# Path to your PowerPoint file
ppt_path = r"C:\Users\jagad\OneDrive\Documents\bayesian inference.ppt.pptx"

# Open the PowerPoint presentation
os.startfile(ppt_path)
time.sleep(5)  # Allow time for PowerPoint to open

# Initialize MediaPipe Hands and Drawing
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Set up the webcam
cap = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()  # Screen size of the monitor

# Function to check if four fingers are raised
def are_four_fingers_raised(hand_landmarks):
    finger_tips = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]
    finger_mcp = [
        mp_hands.HandLandmark.INDEX_FINGER_MCP,
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
        mp_hands.HandLandmark.RING_FINGER_MCP,
        mp_hands.HandLandmark.PINKY_MCP
    ]

    # Count raised fingers by checking the y-coordinates
    raised_fingers = 0
    for tip, mcp in zip(finger_tips, finger_mcp):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[mcp].y:
            raised_fingers += 1
    return raised_fingers == 4

# Function to check if only the index and little fingers are raised
def is_index_and_little_finger_raised(hand_landmarks):
    return (
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y and
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y and
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y and
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y
    )

# Function to check if the thumb is raised
def is_thumb_raised(hand_landmarks):
    return (
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y
    )

# Function to check if the thumb is down
def is_thumb_down(hand_landmarks):
    return (
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y
    )

# Function to check if index and middle fingers are raised for zoom in
def is_index_and_middle_finger_raised(hand_landmarks):
    index_tip = mp_hands.HandLandmark.INDEX_FINGER_TIP
    middle_tip = mp_hands.HandLandmark.MIDDLE_FINGER_TIP
    index_mcp = mp_hands.HandLandmark.INDEX_FINGER_MCP
    middle_mcp = mp_hands.HandLandmark.MIDDLE_FINGER_MCP

    return (
        hand_landmarks.landmark[index_tip].y < hand_landmarks.landmark[index_mcp].y and
        hand_landmarks.landmark[middle_tip].y < hand_landmarks.landmark[middle_mcp].y
    )

# Function to check if middle and ring fingers are raised for zoom out
def is_middle_and_ring_finger_raised(hand_landmarks):
    middle_tip = mp_hands.HandLandmark.MIDDLE_FINGER_TIP
    ring_tip = mp_hands.HandLandmark.RING_FINGER_TIP
    middle_mcp = mp_hands.HandLandmark.MIDDLE_FINGER_MCP
    ring_mcp = mp_hands.HandLandmark.RING_FINGER_MCP

    return (
        hand_landmarks.landmark[middle_tip].y < hand_landmarks.landmark[middle_mcp].y and
        hand_landmarks.landmark[ring_tip].y < hand_landmarks.landmark[ring_mcp].y
    )

# Function to take a screenshot of the PowerPoint presentation
def take_ppt_screenshot():
    ppt_window = None

    # Get the PowerPoint window
    for window in gw.getWindowsWithTitle("PowerPoint"):
        ppt_window = window
        break

    if ppt_window:
        ppt_window.activate()
        time.sleep(0.2)  # Ensure the window is active
        screenshot = pyautogui.screenshot(region=(ppt_window.left, ppt_window.top, ppt_window.width, ppt_window.height))
        filename = f"ppt_screenshots/screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        screenshot.save(filename)
        print(f"Screenshot saved as {filename}")
    else:
        print("PowerPoint window not found.")

# Function to check if three fingers are raised
def are_three_fingers_raised(hand_landmarks):
    finger_tips = [
        mp_hands.HandLandmark.PINKY_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
    ]
    finger_mcps = [
        mp_hands.HandLandmark.PINKY_MCP,
        mp_hands.HandLandmark.RING_FINGER_MCP,
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
    ]

    raised_fingers = 0
    for tip, mcp in zip(finger_tips, finger_mcps):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[mcp].y:  # Check if tip is above MCP
            raised_fingers += 1
    return raised_fingers == 3  # All three fingers raised

# Main loop
while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Flip the image for mirror effect
    image = cv2.flip(image, 1)
    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # If only the index finger is raised, move the mouse pointer
            if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y:
                x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w)
                y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h)
                pyautogui.moveTo(screen_width / w * x, screen_height / h * y)

            if are_four_fingers_raised(hand_landmarks):
                pyautogui.press("right")
                time.sleep(3)

            if is_index_and_little_finger_raised(hand_landmarks):
                pyautogui.press("left")
                time.sleep(3)

            if is_thumb_raised(hand_landmarks):
                pyautogui.hotkey("alt", "f5")
                time.sleep(0.5)

            if is_thumb_down(hand_landmarks):
                pyautogui.press("esc")
                time.sleep(0.5)

            if is_index_and_middle_finger_raised(hand_landmarks):
                pyautogui.hotkey("ctrl", "+")  # Zoom in
                time.sleep(1)

            if is_middle_and_ring_finger_raised(hand_landmarks):
                pyautogui.hotkey("ctrl", "-")  # Zoom out
                time.sleep(2)

            # Take screenshot
            if is_thumb_raised(hand_landmarks) and hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y:
                take_ppt_screenshot()

        # Check if the gesture is made with both hands (three fingers raised on both hands)
        if len(results.multi_hand_landmarks) == 2:
            left_hand = results.multi_hand_landmarks[0]
            right_hand = results.multi_hand_landmarks[1]

            if are_three_fingers_raised(left_hand):
                pyautogui.hotkey("alt", "f4")  # Exit presentation
                print("Exiting presentation.")
                time.sleep(0.5)  # Avoid multiple triggers

        # Draw landmarks
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show the webcam feed
    cv2.imshow('Gesture Control', image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
