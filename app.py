import streamlit as st
from PIL import Image
import tempfile
from predict import predict

st.set_page_config(page_title="Spot the Fake Photo")

st.title("Spot the Fake Photo")

st.write(
    "Upload or capture an image.\n\n"
    "0 = Real Photo\n"
    "1 = Photo of a Screen"
)

uploaded = st.camera_input("Take a photo")

if uploaded is None:
    uploaded = st.file_uploader(
        "Or upload an image",
        type=["jpg","jpeg","png"]
    )

if uploaded is not None:

    image = Image.open(uploaded)

    st.image(image, use_container_width=True)

    with tempfile.NamedTemporaryFile(
        suffix=".jpg",
        delete=False
    ) as tmp:

        image.save(tmp.name)

        score = predict(tmp.name)

    st.subheader("Prediction")

    st.progress(float(score))

    if score > 0.5:
        st.error(f"Photo of Screen ({score:.2f})")
    else:
        st.success(f"Real Photo ({1-score:.2f})")