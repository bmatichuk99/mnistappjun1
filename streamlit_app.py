import streamlit as st
import tensorflow as tf
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps

# Load MNIST data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Create inverted images for training
x_train_inverted = 1 - x_train
x_test_inverted = 1 - x_test

# Combine original and inverted images
x_train_combined = np.concatenate((x_train, x_train_inverted), axis=0)
y_train_combined = np.concatenate((y_train, y_train), axis=0)
x_test_combined = np.concatenate((x_test, x_test_inverted), axis=0)
y_test_combined = np.concatenate((y_test, y_test), axis=0)

# Create a simple neural network model
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Initialize the model
model = create_model()

# Streamlit application
st.title('MNIST Neural Network Trainer and Tester')

# Train button
if st.button('Train Neural Network'):
    model.fit(x_train_combined, y_train_combined, epochs=5)
    st.write("Model trained successfully!")

# Create a canvas for drawing digits
st.write("Draw a digit (0-9) below:")
canvas_result = st_canvas(
    fill_color="white",  # Fixed fill color with some opacity
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Test button
if st.button('Test Neural Network'):
    if canvas_result.image_data is not None:
        # Convert the canvas to a PIL image
        image = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
        image = image.convert('L')  # Convert to grayscale
        image = ImageOps.invert(image)  # Invert the image
        image = image.resize((28, 28))  # Resize to 28x28

        # Display the processed image
        st.image(image, caption='Processed Image for Model Input', use_column_width=True)

        # Prepare the image for prediction
        image = np.array(image).reshape(1, 28, 28) / 255.0

        # Predict the digit
        prediction = np.argmax(model.predict(image), axis=-1)
        st.write(f'The model predicts: {prediction[0]}')
