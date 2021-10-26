import cv2
from tensorflow.keras.models import load_model
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import os

# Containers for the titles
header = st.container()
predictor = st.container()

# Save .h5 file in the same directory to load
directory = os.path.dirname(os.path.abspath(__file__))

# Load trained data
@st.cache(allow_output_mutation=True)
def load():
    return load_model(os.path.join(directory, 'model.h5'))
model = load()

with header:
    st.title('Hand writing recognizer')
    st.text('By Brian Han')

CANVAS_SIZE = 192

col1, col2 = st.columns(2)

# Canvas setting
with col1:
    canvas = st_canvas(
        fill_color='#000000',
        stroke_width=20,
        stroke_color='#FFFFFF',
        background_color='#000000',
        width=CANVAS_SIZE,
        height=CANVAS_SIZE,
        drawing_mode='freedraw',
        key='canvas'
    )

with predictor:
    st.header('Write your number in the left box')

# Process handwriting
if canvas.image_data is not None:
    img = canvas.image_data.astype(np.uint8)
    img = cv2.resize(img, dsize=(28, 28))
    preview_img = cv2.resize(img, dsize=(CANVAS_SIZE, CANVAS_SIZE), interpolation=cv2.INTER_NEAREST)

    col2.image(preview_img)

    x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x = x.reshape((-1, 28, 28, 1))
    y = model.predict(x).squeeze()

    # If there is no input, then display 0
    if (x == np.zeros(shape=(1,28,28, 1))).all():
        y = np.array([0,0,0,0,0,0,0,0,0,0])

    st.header('Predicted number below')
    st.write('## Result: %d' % np.argmax(y))
    st.bar_chart(y)
