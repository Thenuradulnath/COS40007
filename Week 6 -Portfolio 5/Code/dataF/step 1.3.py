import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the trained CNN model
model = tf.keras.models.load_model('simple_cnn_model.h5')

# Set the directories for the test set
test_dir = 'test_data'
img_height, img_width = 150, 150
batch_size = 32

# Prepare the test data generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Evaluate the CNN model on the test set
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy for Simple CNN Model: {test_acc}")
