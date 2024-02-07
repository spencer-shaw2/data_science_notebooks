import gradio as gr
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

labels = ['pizza', 'steak'] # 0 = pizza, 1 = steak
model = tf.keras.models.load_model('pizza_steak_model.keras')

def preprocess_and_predict(img):
    # Preprocess: rescale, resize, convert to tensor, and add batch axis
    img = img / 255.0
    img = tf.constant(img)
    img = tf.image.resize(img, size=(224, 224))
    img = tf.expand_dims(img, axis=0)
    # Predict the image class
    proba_prediction = model.predict(img, verbose=False)
    predicted_label = labels[int(tf.round(proba_prediction))]
    probability = np.abs(0.5 - proba_prediction[0, 0])*2
    formatted_probability = "{:.1%}".format(probability)
    return f"{predicted_label} ({formatted_probability} probability)"

with gr.Blocks() as demo:
    # Create the title heading
    with gr.Row():
        gr.HTML("<h1 style='text-align: center;'>Pizza/Steak Classification App</h1>")
    # Create a description below the heading
    with gr.Row():
        gr.HTML("<p style='text-align: center;'>This app features a very simple AI whose only job is to tell you if your image is representative of pizza or steak. That's it. Copy an image from Google and paste it into the interface (while the image is copied to your clipboard, click on the small clipboard icon just above the 'Clear' button). After hitting submit, the AI will tell you which class your image belongs to and how confident it is. Enjoy!</p>")
    # Create the rest of the interface
    with gr.Row():
        gr.Interface(fn=preprocess_and_predict,
                     inputs=gr.Image(),
                     outputs=gr.Label())
    
demo.launch()