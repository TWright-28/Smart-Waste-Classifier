import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import classification_report, accuracy_score

direct = "C:/Users/Tanner/Desktop/ENGR 518/Project"     #path to dataset directory
size = (128, 128)                                         #target size for resizing images
categories = ['organic', 'recyclable', 'trash']         #labels

data, labels = [], []
for id, category in enumerate(categories):                  #going through the directory 
    categoryPath = os.path.join(direct, category)           #path to each category
    if os.path.isdir(categoryPath):                         #checking path exists
        print(f"Loading images from {categoryPath}...")
        for filename in os.listdir(categoryPath):               #looping through each file in category
            img_path = os.path.join(categoryPath, filename)     #getting image path 
            try:
                img = Image.open(img_path).convert('L').resize(size)    #Grayscale and resize to standard 64x64 to image
                data.append(np.array(img) / 255.0)                      # Normalize pixel values 0-1 which is standard for relu and most cnn and add to data list
                labels.append(id)                                       #add the image id to the label array 
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
    else:
        print(f"Directory not found: {categoryPath}")

X = np.array(data)[..., np.newaxis]  # converting list to numpy array, data is 2d (64x64) also adding grayscale channel to end of array in data
y = np.array(labels)                 # turning labels into numpy array 
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)       #80/20 train/test split    

# Build CNN model
model = Sequential([
    Conv2D(32, (2, 2), activation='relu', input_shape=(128, 128, 1)),       #32 filters, each filter is a 2x2. Relu activation. input shape is 64, 64, 1
    MaxPooling2D(pool_size=(3, 3)),                                       #pooling, pool size is 3x3
    Dropout(0.25),                                                        #Dropout to randomly deactivate 25% neurons
    Conv2D(64, (3, 3), activation='relu'),                                # 64 FIlters, 2x2 filter. relu activation 
    MaxPooling2D(pool_size=(2, 2)),                                       # pool size of 2x2
    Dropout(0.25),                                                        #dropout another 25% of neurons
    Conv2D(128, (4, 4), activation='relu'),                               #128 filters, 4x4 size of filter, relu
    MaxPooling2D(pool_size=(2, 2)),                                       #pool size 2x2
    Dropout(0.25),                                                        #DROPout
    Flatten(),                                                            #flatten to prepare for 1d final layer
    Dense(128, activation='relu'),                                        #Dense 128 neuron fully connected layer
    Dropout(0.5),                                                         #dropout of 50 
    Dense(3, activation='softmax')                                        #Output layer using softmax to determine which label is best 
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  

history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
y_pred = np.argmax(model.predict(X_test), axis=1)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
