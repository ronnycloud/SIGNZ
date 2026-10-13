import os
import cv2
import numpy as np
import mediapipe as mp
from glob import glob
from tqdm import tqdm
from src.features.extractor import extract_keypoints

mp_holistic = mp.solutions.holistic

def process_video(video_path, output_path, holistic_model):
    cap = cv2.VideoCapture(video_path)
    frames_data = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic_model.process(image)
        
        # Using the optimized extractor
        keypoints = extract_keypoints(results)
        frames_data.append(keypoints)
        
    cap.release()
    
    # Save as numpy array
    if len(frames_data) > 0:
        np.save(output_path, np.array(frames_data))

def run_pipeline(raw_dir, out_dir):
    \"\"\"
    Ingests raw videos and converts them to numpy arrays of landmarks.
    \"\"\"
    os.makedirs(out_dir, exist_ok=True)
    classes = os.listdir(raw_dir)
    classes = [c for c in classes if os.path.isdir(os.path.join(raw_dir, c))]
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for cls in classes:
            os.makedirs(os.path.join(out_dir, cls), exist_ok=True)
            videos = glob(os.path.join(raw_dir, cls, "*.mp4"))
            
            for vid in tqdm(videos, desc=f"Processing {cls}"):
                vid_name = os.path.basename(vid).replace('.mp4', '.npy')
                out_path = os.path.join(out_dir, cls, vid_name)
                # Skip if already processed (resumable)
                if not os.path.exists(out_path):
                    process_video(vid, out_path, holistic)

if __name__ == "__main__":
    # Example usage:
    # run_pipeline("dataset/raw_videos", "dataset/MP_Data")
    pass
