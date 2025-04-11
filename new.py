import cv2
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the emotions we want to detect
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Paths to emoji images - update these paths to match your emoji dataset location
emoji_images = {
    'angry': 'path/to/emoji_dataset/angry.png',
    'disgust': 'path/to/emoji_dataset/disgust.png',
    'fear': 'path/to/emoji_dataset/fear.png',
    'happy': 'path/to/emoji_dataset/happy.png',
    'sad': 'path/to/emoji_dataset/sad.png',
    'surprise': 'path/to/emoji_dataset/surprise.png',
    'neutral': 'path/to/emoji_dataset/neutral.png'
}

# Load emoji images
loaded_emojis = {}
for emotion, path in emoji_images.items():
    if os.path.exists(path):
       loaded_emojis[emotion] = cv2.imread("C:\Users\harsh\OneDrive\Desktop\emoji\emoji\happy.png", -1)
  # -1 to preserve alpha channel if PNG
    else:
        print(f"Warning: Emoji image for {emotion} not found at {path}")

# Function to create and train the emotion detection model
def create_emotion_model():
    model = Sequential()
    
    # First Convolutional Layer
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Second Convolutional Layer
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Flatten and Dense Layers
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))  # 7 emotions
    
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
    
    return model

# Function to train the model with your dataset
def train_emotion_model(model, train_data_path, validation_data_path):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')
    
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_data_path,
        target_size=(48, 48),
        batch_size=64,
        color_mode='grayscale',
        class_mode='categorical')
    
    validation_generator = validation_datagen.flow_from_directory(
        validation_data_path,
        target_size=(48, 48),
        batch_size=64,
        color_mode='grayscale',
        class_mode='categorical')
    
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // 64,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // 64)
    
    return model

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
        if emoji_resized.shape[2] == 4:
            # Extract RGB and alpha channels
            emoji_rgb = emoji_resized[:emoji_h, :emoji_w, 0:3]
            emoji_alpha = emoji_resized[:emoji_h, :emoji_w, 3] / 255.0
            
            # Reshape alpha for broadcasting
            alpha = emoji_alpha.reshape(emoji_h, emoji_w, 1)
            
            # Extract the region of interest from the frame
            roi = frame[emoji_y:emoji_y+emoji_h, emoji_x:emoji_x+emoji_w]
            
            # Overlay emoji on frame using alpha blending
            frame[emoji_y:emoji_y+emoji_h, emoji_x:emoji_x+emoji_w] = (1-alpha) * roi + alpha * emoji_rgb
        else:
            # Simple overlay without transparency
            frame[emoji_y:emoji_y+emoji_h, emoji_x:emoji_x+emoji_w] = emoji_resized[:emoji_h, :emoji_w]
    
    return frame

# Main function for real-time facial expression recognition
def facial_expression_recognition():
    # Load face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Create and load or train the emotion detection model
    model = create_emotion_model()
    
    # Option 1: Train the model with your dataset
    # train_data_path = 'path/to/your/expression_dataset/train'
    # validation_data_path = 'path/to/your/expression_dataset/validation'
    # model = train_emotion_model(model, train_data_path, validation_data_path)
    # model.save('emotion_model.h5')
    
    # Option 2: Load pre-trained model if available
    # model.load_weights('emotion_model.h5')
    
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
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            # Extract face ROI
            roi_gray = gray[y:y+h, x:x+w]
            
            # Resize to 48x48 for emotion detection
            roi_gray = cv2.resize(roi_gray, (48, 48))
            
            # Normalize and reshape for the model
            roi = roi_gray.astype('float') / 255.0
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)
            
            # Make prediction
            # In a real implementation, you would uncomment this
            # prediction = model.predict(roi)[0]
            # emotion_idx = np.argmax(prediction)
            # emotion_text = emotions[emotion_idx]
            
            # For demo purposes, we'll use a simple placeholder that detects smiles
            # You should replace this with actual model prediction in production
            smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
            smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
            
            if len(smiles) > 0:
                emotion_text = 'happy'
            else:
                emotion_text = 'neutral'
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Display emotion text
            cv2.putText(frame, emotion_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Overlay corresponding emoji on the display frame
            if emotion_text in loaded_emojis:
                display_frame = overlay_emoji(display_frame, loaded_emojis[emotion_text], x, y, w, h)
        
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