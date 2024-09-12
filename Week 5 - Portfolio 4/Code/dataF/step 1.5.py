import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import load_model

# Define input shape based on what the ResNet50 expects
input_shape = (150, 150, 3)

# Load ResNet50 model with include_top=False to allow customization
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

# Define a new model structure
model_resnet = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Adjust output for binary classification
])

# Load the weights
model_resnet.load_weights('resnet50_model.h5')

# Test Data Generator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_dir = 'test_data'
img_height, img_width = 150, 150
batch_size = 32

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Evaluate the ResNet50 model
resnet_test_loss, resnet_test_acc = model_resnet.evaluate(test_generator)
print(f"Test Accuracy for ResNet50 Model: {resnet_test_acc}")
