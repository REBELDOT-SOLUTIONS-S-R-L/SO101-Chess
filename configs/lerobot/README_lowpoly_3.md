---
task_categories:
  - robotics
tags:
  - lerobot
  - imitation-learning
  - isaac-lab
  - robotics
pretty_name: SO-101 Chess Balanced Lowpoly 3
---

# SO-101 Chess Balanced Lowpoly 3

This is a LeRobot v3 dataset of 1,000 successful synthetic SO-101 chess
manipulation episodes generated in Isaac Lab. Episodes are balanced across pawn,
rook, knight, bishop, queen, and king moves.

The policy task string is:

`Move the chess piece from the red square to the green square`

Each frame contains the six robot joint positions, the six commanded joint
targets, a 640x480 top-camera video, and a 640x480 wrist-camera video at 60 Hz.
