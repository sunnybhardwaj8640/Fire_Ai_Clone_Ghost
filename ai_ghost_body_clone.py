
import cv2
import mediapipe as mp
import pygame
import numpy as np
import time
from noise import pnoise1

# Setup MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Pygame init
pygame.init()
width, height = 1280, 480
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("AI Ghost Body Clone")

clock = pygame.time.Clock()
cap = cv2.VideoCapture(0)

# Fire color gradient
def get_fire_color(intensity):
    r = 255
    g = max(50, 255 - intensity * 10)
    b = max(0, 80 - intensity * 8)
    return (r, g, b)

# Trails for both hands
trail_right = []
trail_left = []

running = True
while running:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    h, w, _ = frame.shape

    # Webcam feed on left side
    frame_surface = pygame.image.frombuffer(frame.tobytes(), frame.shape[1::-1], "BGR")
    screen.blit(pygame.transform.scale(frame_surface, (width // 2, height)), (0, 0))

    # Clear right side (black background)
    pygame.draw.rect(screen, (0, 0, 0), (width // 2, 0, width // 2, height))

    joints = []
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            x, y = int(lm.x * (width // 2)), int(lm.y * height)
            joints.append((x, y))

        # Get left/right wrists
        rwrist = joints[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        lwrist = joints[mp_pose.PoseLandmark.LEFT_WRIST.value]

        t = time.time()
        for wrist, trail in zip([rwrist, lwrist], [trail_right, trail_left]):
            offset_x = int(pnoise1(t * 0.8 + (trail == trail_left)) * 20)
            offset_y = int(pnoise1(t * 0.8 + 50 + (trail == trail_left)) * 20)
            fx, fy = wrist[0] + offset_x, wrist[1] + offset_y
            trail.append((fx, fy))
            if len(trail) > 25:
                trail.pop(0)
    else:
        trail_right.clear()
        trail_left.clear()

    # Draw fire trails on ghost clone side
    for trail in [trail_right, trail_left]:
        for i, pt in enumerate(trail):
            intensity = len(trail) - i
            color = get_fire_color(intensity)
            pygame.draw.circle(screen, color, (pt[0] + width // 2, pt[1]), 10)

    # Draw ghost skeleton
    if joints:
        ghost_surface = pygame.Surface((width // 2, height), pygame.SRCALPHA)
        for joint in joints:
            pygame.draw.circle(ghost_surface, (255, 255, 255, 100), (joint[0], joint[1]), 5)
        for a, b in mp_pose.POSE_CONNECTIONS:
            try:
                pt1 = joints[a]
                pt2 = joints[b]
                pygame.draw.line(ghost_surface, (255, 255, 255, 80), pt1, pt2, 2)
            except IndexError:
                continue
        screen.blit(ghost_surface, (width // 2, 0))

    # Handle quit
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.flip()
    clock.tick(30)

cap.release()
pygame.quit()
