import cv2
import numpy as np
import os

# Define the emotions we want to detect
emotions = ['angry', 'happy', 'sad', 'surprise', 'neutral']

# Paths to emoji images - update these paths to match your emoji dataset location
emoji_images = {
    'angry': r'C:\Users\harsh\OneDrive\Desktop\emoji\angry.jpg',
    'happy': r'C:\Users\harsh\OneDrive\Desktop\emoji\happy.jpg',
    'sad': r'C:\Users\harsh\OneDrive\Desktop\emoji\sad.jpg',
    'surprise': r'C:\Users\harsh\OneDrive\Desktop\emoji\surprise.jpg',
    'fear':r'C:\Users\harsh\OneDrive\Desktop\emoji\fear.jpg',
    'disgust':r'C:\Users\harsh\OneDrive\Desktop\emoji\disgust.jpg',
    'neutral': r'C:\Users\harsh\OneDrive\Desktop\emoji\neutral.jpg'
}

# Load emoji images
loaded_emojis = {}
for emotion, path in emoji_images.items():
    if os.path.exists(path):
        loaded_emojis[emotion] = cv2.imread(path, -1)  # <- This is the correct line
    else:
        print(f"Warning: Emoji image for {emotion} not found at {path}")
        # Create a placeholder colored square if image not found
        placeholder = np.zeros((100, 100, 4), dtype=np.uint8)
        if emotion == 'angry':
            placeholder[:, :, :3] = [0, 0, 255]
        elif emotion == 'happy':
            placeholder[:, :, :3] = [0, 255, 255]
        elif emotion == 'sad':
            placeholder[:, :, :3] = [255, 0, 0]
        elif emotion == 'surprise':
            placeholder[:, :, :3] = [0, 255, 0]
        else:
            placeholder[:, :, :3] = [200, 200, 200]
        placeholder[:, :, 3] = 180
        loaded_emojis[emotion] = placeholder


# Function to overlay emoji on frame
def overlay_emoji(frame, emoji_img, x, y, w, h, scale_factor=1.5):
    # Calculate the size for the emoji (larger than the face)
    emoji_size = int(max(w, h) * scale_factor)
    
    # Resize emoji to the calculated size
    if emoji_img is not None:
        emoji_resized = cv2.resize(emoji_img, (emoji_size, emoji_size))
        
        # Calculate position to center emoji on face
        emoji_x = max(0, x + (w - emoji_size) // 2)
        emoji_y = max(0, y + (h - emoji_size) // 2)
        
        # Handle case where emoji might go beyond frame boundaries
        emoji_w = min(emoji_size, frame.shape[1] - emoji_x)
        emoji_h = min(emoji_size, frame.shape[0] - emoji_y)
        
        # If emoji has an alpha channel (PNG)
        if emoji_img.shape[2] == 4 and emoji_h > 0 and emoji_w > 0:
            # Extract RGB and alpha channels
            emoji_rgb = emoji_resized[:emoji_h, :emoji_w, 0:3]
            emoji_alpha = emoji_resized[:emoji_h, :emoji_w, 3] / 255.0
            
            # Reshape alpha for broadcasting
            alpha = emoji_alpha.reshape(emoji_h, emoji_w, 1)
            
            # Extract the region of interest from the frame
            roi = frame[emoji_y:emoji_y+emoji_h, emoji_x:emoji_x+emoji_w]
            
            # Overlay emoji on frame using alpha blending
            blended = (1-alpha) * roi + alpha * emoji_rgb
            frame[emoji_y:emoji_y+emoji_h, emoji_x:emoji_x+emoji_w] = blended.astype(np.uint8)
        else:
            # Simple overlay without transparency
            if emoji_h > 0 and emoji_w > 0:
                frame[emoji_y:emoji_y+emoji_h, emoji_x:emoji_x+emoji_w] = emoji_resized[:emoji_h, :emoji_w, :3]
    
    return frame

# Function to detect emotion using OpenCV features
def detect_emotion(roi_gray):
    """
    Detect emotion using simple OpenCV-based features
    This is a simplified approach that looks for specific facial features
    """
    # Initialize cascades for different facial features
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    # Detect smiles and eyes
    smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
    
    # Simple rule-based emotion classification
    if len(smiles) > 0:
        return 'happy'
    
    # Additional features could be integrated to detect other emotions
    # This is a simplified version that focuses on easily detectable expressions
    
    # Count pixels in the upper half vs lower half for a rough estimate of expression
    height, width = roi_gray.shape
    upper_half = roi_gray[0:height//2, :]
    lower_half = roi_gray[height//2:, :]
    
    upper_mean = np.mean(upper_half)
    lower_mean = np.mean(lower_half)
    
    # Apply some heuristic rules for emotion detection
    if len(eyes) >= 2:
        eye_ratio = eyes[0][2] * eyes[0][3] / (width * height)
        
        # Wide eyes might indicate surprise
        if eye_ratio > 0.05:
            return 'surprise'
        
        # Looking down might indicate sadness
        if upper_mean < lower_mean - 10:
            return 'sad'
        
        # High contrast in the lower face might indicate anger
        lower_std = np.std(lower_half)
        if lower_std > 50:
            return 'angry'
    
    return 'neutral'

# Main function for real-time facial expression recognition
def facial_expression_recognition():
    # Load face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Could not open webcam")
        return
    
    print("Facial Expression Recognition Started. Press 'q' to quit.")
    
    while True:
        # Read frame
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break
        
        # Create a copy of the frame for displaying emoji
        display_frame = frame.copy()
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        for (x, y, w, h) in faces:
            # Extract face ROI
            roi_gray = gray[y:y+h, x:x+w]
            
            # Detect emotion
            emotion = detect_emotion(roi_gray)
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Display emotion text
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Overlay corresponding emoji on the display frame
            if emotion in loaded_emojis:
                display_frame = overlay_emoji(display_frame, loaded_emojis[emotion], x, y, w, h)
        
        # Show both frames
        cv2.imshow('Expression Detection', frame)
        cv2.imshow('Emoji Display', display_frame)
        
        # Break loop on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    facial_expression_recognition()