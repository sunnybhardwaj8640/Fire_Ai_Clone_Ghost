
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
pygame.display.set_caption("Full-Body Ghost Clone")

clock = pygame.time.Clock()
cap = cv2.VideoCapture(0)

# Color fade
def get_trail_color(intensity):
    r = 255
    g = max(100, 255 - intensity * 10)
    b = max(0, 100 - intensity * 5)
    return (r, g, b)

# Pose trail history
ghost_trails = []

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

    # Clear right side
    pygame.draw.rect(screen, (0, 0, 0), (width // 2, 0, width // 2, height))

    joints = []
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            x, y = int(lm.x * (width // 2)), int(lm.y * height)
            joints.append((x, y))

        if len(joints) == 33:
            ghost_trails.append(joints.copy())
            if len(ghost_trails) > 15:
                ghost_trails.pop(0)
    else:
        ghost_trails.clear()

    # Draw ghost trails
    for t_idx, trail in enumerate(ghost_trails):
        alpha = int(255 * (t_idx + 1) / len(ghost_trails))
        ghost_surface = pygame.Surface((width // 2, height), pygame.SRCALPHA)
        for joint in trail:
            pygame.draw.circle(ghost_surface, (255, 255, 255, alpha // 2), (joint[0], joint[1]), 5)
        for a, b in mp_pose.POSE_CONNECTIONS:
            try:
                pt1 = trail[a]
                pt2 = trail[b]
                pygame.draw.line(ghost_surface, (255, 255, 255, alpha // 3), pt1, pt2, 2)
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
