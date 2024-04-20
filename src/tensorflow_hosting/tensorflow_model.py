import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('path_to_your_model')

# Convert the model to SavedModel format
tf.saved_model.save(model, 'path_to_saved_model')
