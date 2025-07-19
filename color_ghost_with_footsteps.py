import cv2
import mediapipe as mp
import pygame
import numpy as np
import time
import math

# Setup MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Pygame init
pygame.init()
width, height = 1280, 480
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Ghost Clone with Footstep Trails & Color Shift")

clock = pygame.time.Clock()
cap = cv2.VideoCapture(0)

ghost_trails = []
footstep_trails = []

# Get dynamic color
def shifting_color(t):
    r = int((math.sin(t) + 1) * 127)
    g = int((math.sin(t + 2) + 1) * 127)
    b = int((math.sin(t + 4) + 1) * 127)
    return (r, g, b)

running = True
while running:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    h, w, _ = frame.shape

    # Webcam feed on left
    frame_surface = pygame.image.frombuffer(frame.tobytes(), frame.shape[1::-1], "BGR")
    screen.blit(pygame.transform.scale(frame_surface, (width // 2, height)), (0, 0))

    # Clear right canvas
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

            # Save heel positions
            lheel = joints[mp_pose.PoseLandmark.LEFT_HEEL.value]
            rheel = joints[mp_pose.PoseLandmark.RIGHT_HEEL.value]
            footstep_trails.append((lheel, rheel))
            if len(footstep_trails) > 50:
                footstep_trails.pop(0)
    else:
        ghost_trails.clear()
        footstep_trails.clear()

    # Draw ghost trails
    t = time.time()
    for i, trail in enumerate(ghost_trails):
        alpha = int(255 * (i + 1) / len(ghost_trails))
        ghost_surface = pygame.Surface((width // 2, height), pygame.SRCALPHA)
        color = shifting_color(t + i * 0.3)
        for joint in trail:
            pygame.draw.circle(ghost_surface, (*color, alpha // 2), (joint[0], joint[1]), 5)
        for a, b in mp_pose.POSE_CONNECTIONS:
            try:
                pt1 = trail[a]
                pt2 = trail[b]
                pygame.draw.line(ghost_surface, (*color, alpha // 3), pt1, pt2, 2)
            except IndexError:
                continue
        screen.blit(ghost_surface, (width // 2, 0))

    # Draw footsteps
    for i, (lh, rh) in enumerate(footstep_trails):
        f_alpha = int(255 * (i + 1) / len(footstep_trails))
        pygame.draw.circle(screen, (100, 255, 100, f_alpha), (lh[0] + width // 2, lh[1]), 8)
        pygame.draw.circle(screen, (100, 255, 255, f_alpha), (rh[0] + width // 2, rh[1]), 8)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.flip()
    clock.tick(30)

cap.release()
pygame.quit()
