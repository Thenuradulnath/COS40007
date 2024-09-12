import numpy as np
import os
import pandas as pd
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Define the new relative paths
train_dir = 'Corrosion'
test_dir = 'test_data'
output_dir = 'cnn_test'  # Directory to save the test outcome images

# Parameters
img_height, img_width = 28, 28
batch_size = 32
epochs = 10
num_classes = 2  # rust and no rust

# ImageDataGenerator for training and testing with color images
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    color_mode='rgb'  # Use 'rgb' for color images
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    color_mode='rgb',
    shuffle=False  # Disable shuffling for accurate predictions
)

# Build the model
model = Sequential([
    Input(shape=(img_height, img_width, 3)),  # Input shape for RGB images (3 channels)
    Conv2D(8, 3, activation='relu'),
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(10, activation='relu'),
    Dense(1, activation='sigmoid'),  # Output layer for binary classification
])

# Compile the model
model.compile(
    optimizer=Adam(),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(
    train_generator,
    epochs=epochs,
    validation_data=test_generator
)

# Save the model weights
model.save_weights('cnn_rust_classification.weights.h5')

# Evaluate the model
loss, accuracy = model.evaluate(test_generator)
print(f'Test accuracy: {accuracy * 100:.2f}%')

# Predict on the test set
predictions = model.predict(test_generator)
predicted_classes = np.round(predictions).astype(int)  # Round predictions to get 0 or 1

# True labels
true_classes = test_generator.classes

# Class labels
class_labels = list(test_generator.class_indices.keys())

# Create a table of true class and predicted class
results = pd.DataFrame({
    'Filename': test_generator.filenames,
    'True Class': [class_labels[int(i)] for i in true_classes],
    'Predicted Class': [class_labels[int(i)] for i in predicted_classes]
})

# Display the table
print(results)

# Calculate final accuracy
correct_predictions = np.sum(predicted_classes.flatten() == true_classes)
final_accuracy = correct_predictions / len(true_classes)
print(f'Final Overall Accuracy: {final_accuracy * 100:.2f}%')

# Save the results to a CSV file
results.to_csv('cnn_test_results.csv', index=False)

# Create a directory for the CNN test outcome images
os.makedirs(output_dir, exist_ok=True)

# Save images with predicted labels
for i, filename in enumerate(test_generator.filenames):
    img_path = os.path.join(test_dir, filename)
    imgBGR = cv2.imread(img_path)  # Read the image in BGR format
    imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)  # Convert to RGB format

    # Get the true and predicted class labels
    true_label = class_labels[true_classes[i]]
    predicted_label = class_labels[predicted_classes[i][0]]

    # Add predicted class to the image
    imgRGB = cv2.putText(
        imgRGB, f"True: {true_label} | Predicted: {predicted_label}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA
    )

    # Save the output image
    output_img_path = os.path.join(output_dir, f"test_image_{i+1}.jpg")
    cv2.imwrite(output_img_path, cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR))  # Save in BGR format

    print(f"Saved test outcome image {i+1} to {output_img_path}")