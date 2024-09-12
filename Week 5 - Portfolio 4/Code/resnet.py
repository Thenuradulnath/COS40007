import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications.resnet50 import preprocess_input

# Global Constants
NUM_CLASSES = 2  # rust and no rust
IMAGE_RESIZE = 224  # ResNet50 requires 224x224 input images
DENSE_LAYER_ACTIVATION = 'softmax'
OBJECTIVE_FUNCTION = 'categorical_crossentropy'
LOSS_METRICS = ['accuracy']
NUM_EPOCHS = 10
EARLY_STOP_PATIENCE = 3
BATCH_SIZE_TRAINING = 32
BATCH_SIZE_VALIDATION = 32

# Define data directories
train_dir = 'Corrosion'
test_dir = 'test_data'
output_dir = 'resent50_test'  # Directory to save ResNet50 test images

# Initialize ImageDataGenerator with preprocessing
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = data_generator.flow_from_directory(
    train_dir,
    target_size=(IMAGE_RESIZE, IMAGE_RESIZE),
    batch_size=BATCH_SIZE_TRAINING,
    class_mode='categorical',
    shuffle=True
)

validation_generator = data_generator.flow_from_directory(
    test_dir,
    target_size=(IMAGE_RESIZE, IMAGE_RESIZE),
    batch_size=BATCH_SIZE_VALIDATION,
    class_mode='categorical',
    shuffle=False  # Do not shuffle so we can correctly map true labels with predictions
)

# Adjust steps per epoch based on generator length
steps_per_epoch = train_generator.samples // BATCH_SIZE_TRAINING
validation_steps = validation_generator.samples // BATCH_SIZE_VALIDATION

# Define the input tensor
input_tensor = Input(shape=(IMAGE_RESIZE, IMAGE_RESIZE, 3))

# Load ResNet50 model without the top layer, using ImageNet weights
base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)

# Add a new GlobalAveragePooling2D layer after the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add a dense layer for classification
x = Dense(NUM_CLASSES, activation=DENSE_LAYER_ACTIVATION)(x)

# Define the final model
model = Model(inputs=base_model.input, outputs=x)

# Freeze the base model layers to prevent training them
for layer in base_model.layers:
    layer.trainable = False

# Print the model summary to verify structure
model.summary()

# Compile the model
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss=OBJECTIVE_FUNCTION, metrics=LOSS_METRICS)

# Callbacks for Early Stopping and Model Checkpointing
cb_early_stopper = EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE)
cb_checkpointer = ModelCheckpoint(filepath='best_model.keras', monitor='val_loss', save_best_only=True, mode='auto')

# Train the Model
model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=NUM_EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[cb_checkpointer, cb_early_stopper]
)

# Load the best model weights
model.load_weights('best_model.keras')

# Evaluate the Model on Test Data
test_loss, test_accuracy = model.evaluate(validation_generator)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')

# Predict on test set
validation_generator.reset()  # Resetting the generator for new predictions
predictions = model.predict(validation_generator, steps=validation_steps, verbose=1)
predicted_class_indices = np.argmax(predictions, axis=1)

# True class indices from the validation generator
true_class_indices = validation_generator.classes

# Class labels
class_labels = list(validation_generator.class_indices.keys())
predicted_labels = [class_labels[idx] for idx in predicted_class_indices]
true_labels = [class_labels[idx] for idx in true_class_indices]

# Create a DataFrame to store true and predicted classes
results_df = pd.DataFrame({
    'Filename': validation_generator.filenames,
    'True Class': true_labels,
    'Predicted Class': predicted_labels
})

# Print the table of true vs predicted classes
print(results_df)

# Calculate final accuracy
correct_predictions = np.sum(predicted_class_indices == true_class_indices)
final_accuracy = correct_predictions / len(true_class_indices)

print(f'Final Overall Accuracy: {final_accuracy * 100:.2f}%')

# Save the results to a CSV file for reference
results_df.to_csv('resnet50_test_results.csv', index=False)

# Create a directory for the ResNet50 test outcome images
os.makedirs(output_dir, exist_ok=True)

# Save images with predicted labels
for i, filename in enumerate(validation_generator.filenames):
    img_path = os.path.join(test_dir, filename)
    imgBGR = cv2.imread(img_path)  # Read image in BGR format
    imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)  # Convert to RGB format

    # Get the predicted class label
    predicted_class = predicted_labels[i]

    # Add predicted class to the image
    imgRGB = cv2.putText(
        imgRGB, f"Predicted: {predicted_class}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA
    )

    # Save the output image
    output_img_path = os.path.join(output_dir, f"test_image_{i+1}.jpg")
    cv2.imwrite(output_img_path, cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR))  # Save in BGR format

    print(f"Saved ResNet50 test outcome image {i+1} to {output_img_path}")

# Visualize some predictions
f, ax = plt.subplots(5, 5, figsize=(15, 15))

for i in range(18):  # Display the first 18 images
    img_path = os.path.join(test_dir, validation_generator.filenames[i])
    imgBGR = cv2.imread(img_path)  # Read image in BGR format
    imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)  # Convert to RGB format

    # Get the predicted class label
    predicted_class = predicted_labels[i]

    # Display the image and prediction
    ax[i // 5, i % 5].imshow(imgRGB)
    ax[i // 5, i % 5].axis('off')
    ax[i // 5, i % 5].set_title(f"Predicted: {predicted_class}")

plt.tight_layout()
plt.show()