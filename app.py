import streamlit as st
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="VisionIQ AI",
    page_icon="👁️",
    layout="centered"
)

st.title("👁️ VisionIQ AI")
st.subheader("AI-Powered Visual Intelligence Platform")

# -----------------------------
# Load Florence-2 Model
# -----------------------------
@st.cache_resource
def load_model():

    model_id = "microsoft/Florence-2-base"

    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True
    )

    return processor, model

# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Convert image to RGB
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Generate Caption"):

        with st.spinner("Loading Florence-2 Model..."):

            processor, model = load_model()

        with st.spinner("Analyzing Image..."):

            prompt = "<CAPTION>"

            inputs = processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            )

            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=50,
                num_beams=3
            )

            generated_text = processor.batch_decode(
                generated_ids,
                skip_special_tokens=False
            )[0]

            parsed_answer = processor.post_process_generation(
                generated_text,
                task=prompt,
                image_size=(image.width, image.height)
            )

            st.success("Caption Generated Successfully!")

            st.subheader("Generated Caption")

            if isinstance(parsed_answer, dict):
                st.write(parsed_answer.get("<CAPTION>", generated_text))
            else:
                st.write(parsed_answer)