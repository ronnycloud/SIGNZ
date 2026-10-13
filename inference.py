import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn.functional as F
import collections

from src.models.transformer import SignTransformer
from src.features.extractor import extract_keypoints
from src.nlp.grammar_corrector import GrammarCorrector

mp_holistic = mp.solutions.holistic

class RealTimePredictor:
    def __init__(self, model_path="checkpoints/best_transformer.pt", sequence_length=60, threshold=0.7):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sequence_length = sequence_length
        self.threshold = threshold
        
        # Load Model checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.classes = checkpoint['classes']
        num_classes = len(self.classes)
        
        self.model = SignTransformer(num_classes=num_classes, input_dim=258)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.frames = collections.deque(maxlen=sequence_length)
        self.grammar_corrector = GrammarCorrector()
        
        # Smoothing variables
        self.recent_predictions = collections.deque(maxlen=5)
        self.sentence = []

    def process_frame(self, frame, holistic_model):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic_model.process(image)
        
        keypoints = extract_keypoints(results)
        self.frames.append(keypoints)
        
        predicted_word = None
        confidence = 0.0
        
        if len(self.frames) == self.sequence_length:
            input_tensor = torch.FloatTensor(np.array(self.frames)).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(input_tensor)
                probs = F.softmax(output, dim=1)
                conf, pred_idx = torch.max(probs, 1)
                
                confidence = conf.item()
                if confidence > self.threshold:
                    raw_word = self.classes[pred_idx.item()]
                    self.recent_predictions.append(raw_word)
                    
                    # Temporal Smoothing: Require same prediction 5 frames in a row
                    if len(set(self.recent_predictions)) == 1:
                        predicted_word = raw_word
                        # Avoid repeated words in sentence
                        if not self.sentence or self.sentence[-1] != predicted_word:
                            self.sentence.append(predicted_word)

        return predicted_word, confidence, self.grammar_corrector.correct(self.sentence)

def run_webcam():
    cap = cv2.VideoCapture(0)
    predictor = RealTimePredictor()
    
    print("Starting Webcam Inference...")
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            word, conf, formatted_sentence = predictor.process_frame(frame, holistic)
            
            # Display outputs
            cv2.rectangle(frame, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(frame, f"Sentence: {formatted_sentence}", (10,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                        
            if word:
                cv2.putText(frame, f"Word: {word} ({conf:.2f})", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow('SignVision Real-Time', frame)
            
            # Press 'r' to reset sentence, 'q' to quit
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                predictor.sentence = []
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam()
