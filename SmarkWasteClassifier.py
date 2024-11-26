import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load and Preprocess Images
def preprocess_image(image_path, target_size=(64, 64)):
    try:
        # Open image, convert to grayscale, and resize
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = img.resize(target_size)  # Resize to target size
        img_array = np.array(img) / 255.0  # Normalize pixel values to 0-1
        return img_array
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def load_images(folder, label, target_size=(64, 64)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img_array = preprocess_image(img_path, target_size)
        if img_array is not None:
            images.append(img_array)
            labels.append(label)
    return images, labels

def load_dataset(base_path, target_size=(64, 64)):
    categories = ['organic', 'recyclable', 'trash']
    data = []
    labels = []
    for idx, category in enumerate(categories):
        category_path = os.path.join(base_path, category)
        if os.path.isdir(category_path):
            print(f"Loading images from {category_path}...")
            images, lbls = load_images(category_path, idx, target_size)
            data.extend(images)
            labels.extend(lbls)
        else:
            print(f"Directory not found: {category_path}")
    return np.array(data), np.array(labels)

# Step 2: Build CNN
def build_cnn(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')  # Output layer for 3 classes: organic, recyclable, trash
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Main Execution
if __name__ == "__main__":
    # Specify dataset path
    dataset_path = "C:/Users/Tanner/Desktop/ENGR 518/Project"  # Replace with your dataset path
    img_size = (64, 64)  # Target size for resizing images

    print("Loading dataset...")
    X, y = load_dataset(dataset_path, img_size)

    print("Adding channel dimension...")
    X = X[..., np.newaxis]  # Add channel dimension for grayscale images

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Building CNN...")
    model = build_cnn(input_shape=(img_size[0], img_size[1], 1))

    print("Training model...")
    history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))

    print("Evaluating model...")
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Plot Training History
    import matplotlib.pyplot as plt

    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')
    plt.show()

    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss')
    plt.show()

    print("Saving model...")
    model.save('organic_recyclable_trash_cnn.h5')
    print("Model saved as 'organic_recyclable_trash_cnn.h5'")
