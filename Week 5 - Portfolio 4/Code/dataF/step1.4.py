import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set the directories for the dataset
dataset_dir = 'Corrosion'
img_height, img_width = 150, 150
batch_size = 32

# Prepare the ImageDataGenerators
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Prepare the data generators for training and validation
train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='training',
    shuffle=True
)

validation_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
    shuffle=True
)

# Load the ResNet50 model pre-trained on ImageNet, excluding the top layers
resnet_model = tf.keras.applications.ResNet50(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze the pre-trained ResNet50 layers
resnet_model.trainable = False

# Add custom layers on top of ResNet50
model_resnet = tf.keras.Sequential([
    resnet_model,
    tf.keras.layers.GlobalAveragePooling2D(),  # Flatten the output
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model_resnet.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])

# Train the model
history_resnet = model_resnet.fit(train_generator, validation_data=validation_generator, epochs=10)

# Save the trained ResNet50 model
model_resnet.save('resnet50_model.h5')

print("ResNet50 model trained and saved.")
