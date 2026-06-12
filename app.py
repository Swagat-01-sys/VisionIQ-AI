import streamlit as st
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
import sys

print("=" * 50)
print("PYTHON:", sys.executable)
print("=" * 50)
st.set_page_config(page_title="VisionIQ AI")

st.title("👁️ VisionIQ AI")
st.subheader("AI-Powered Visual Intelligence Platform")

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

processor, model = load_model()

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Generate Caption"):

        with st.spinner("Analyzing image..."):

            prompt = "<CAPTION>"

            inputs = processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            )

            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=50
            )

            generated_text = processor.batch_decode(
                generated_ids,
                skip_special_tokens=False
            )[0]

            st.success("Caption Generated")

            st.write(generated_text)