# main.py
# This script is the core of the basketball analytics application.
# It uses a custom YOLOv8 model to detect players and referees,
# and BoT-SORT for tracking to maintain consistent IDs for each person.
# It then calculates and displays real-time statistics on the video feed.

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import utils

# Configuration 
MODEL_PATH = 'best.pt'
INPUT_VIDEO_PATH = 'game_input.mp4'
OUTPUT_VIDEO_PATH = 'output_video.mp4'
CONFIDENCE_THRESHOLD = 0.5

#CLASS_NAMES based on your .yaml file
CLASS_NAMES = {
    0: 'Ball', 
    1: 'Hoop', 
    2: 'Period', 
    3: 'Player', 
    4: 'Ref', 
    5: 'Shot Clock', 
    6: 'Team Name', 
    7: 'Team Points', 
    8: 'Time Remaining'
}

# --- Custom Color Mapping for Different Classes ---
# BGR format: (Blue, Green, Red)
# NOTE: To have different colors for two teams, you would need to train your model
# with separate classes like 'player_team_A' and 'player_team_B'.
# For now, all 'Player' objects will have the same color.
COLOR_MAP = {
    'Player': (255, 0, 0),   # Blue for Players
    'Ref': (0, 255, 255),    # Yellow for Referees
    'Ball': (0, 0, 255),     # Red for the Ball
    'default': (0, 255, 0)   # Green for anything else
}


# --- Initialization ---
# Load the custom YOLOv8 model
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

# Open the input video file
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video file at {INPUT_VIDEO_PATH}")
    exit()

# Get video properties for the output file
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

# --- Data Storage for Analytics ---
track_history = defaultdict(lambda: [])
player_distance = defaultdict(float)
last_positions = {}
ball_possession_frames = defaultdict(int) # Changed name for clarity
player_with_ball = None

# --- Main Processing Loop ---
frame_number = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("End of video or failed to read frame.")
        break

    # Perform object tracking on the frame
    try:
        results = model.track(frame, persist=True, tracker="botsort.yaml", conf=CONFIDENCE_THRESHOLD, verbose=False)
    except Exception as e:
        print(f"Error during tracking: {e}")
        continue

    # Make a copy of the frame to draw on
    annotated_frame = frame.copy()

    try:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        clss = results[0].boxes.cls.int().cpu().tolist()
        confs = results[0].boxes.conf.cpu().tolist()
    except (AttributeError, IndexError):
        out.write(annotated_frame)
        cv2.imshow("Basketball Analytics", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    # --- Analytics & Manual Drawing ---
    player_positions = {}
    ball_position = None

    # Find the ball's position first
    for i, cls_id in enumerate(clss):
        if CLASS_NAMES.get(cls_id) == 'Ball':
            x, y, _, _ = boxes[i]
            ball_position = (int(x), int(y))
            break

    # Process and draw each detected object
    for box, track_id, cls_id, conf in zip(boxes, track_ids, clss, confs):
        x, y, w, h = box
        center_x, center_y = int(x), int(y)
        class_name = CLASS_NAMES.get(cls_id, 'Unknown')
        
        # Store player positions for possession calculation
        if class_name == 'Player':
            player_positions[track_id] = (center_x, center_y)

        # Add current position to history for tracking lines
        track_history[track_id].append((center_x, center_y))

        # --- Manual Drawing Logic ---
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        
        # Get color from the color map
        box_color = COLOR_MAP.get(class_name, COLOR_MAP['default'])
        
        # Draw the bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
        
        # Create and draw the label (ID, class, confidence)
        label = f"ID:{track_id} {class_name} {conf:.2f}"
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(annotated_frame, (x1, y1 - label_height - baseline), (x1 + label_width, y1), box_color, -1)
        cv2.putText(annotated_frame, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # --- Calculate and Display Stats for Players and Refs ---
        if class_name in ['Player', 'Ref']:
            if track_id in last_positions:
                dist_moved = utils.calculate_distance(last_positions[track_id], (center_x, center_y))
                player_distance[track_id] += dist_moved
                speed = dist_moved * fps
                
                # Display speed and distance
                cv2.putText(annotated_frame, f"D: {player_distance[track_id]:.1f}px", 
                            (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"S: {speed:.1f}px/s", 
                            (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            last_positions[track_id] = (center_x, center_y)

        # If you want to draw the tracking lines un-comment the following part of code 
       # points = np.array(track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
       # cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

    # --- Ball Possession Logic ---
    if ball_position:
        player_with_ball = utils.get_player_with_ball(player_positions, ball_position)
    
    if player_with_ball:
        # Instead of 'team_A', we track possession by any player.
        ball_possession_frames['Player'] += 1

    # --- Display Statistics on Frame ---
    possession_seconds = ball_possession_frames['Player'] / fps
    cv2.putText(annotated_frame, f"Ball Possession: {possession_seconds:.2f}s", 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    out.write(annotated_frame)
    cv2.imshow("Basketball Analytics", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --- Cleanup ---
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Processing complete. Output video saved to {OUTPUT_VIDEO_PATH}")