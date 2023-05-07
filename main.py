import streamlit as st
import os
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch




def predict_step(image_paths):

    """ Generates output """
    print("==> Initializing...")
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    max_length = 16
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)
    

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

st.set_page_config(page_title="Image Caption")
st.write("Image Caption generator")


def load_image(image_file):
    img = Image.open(image_file)
    img = img.convert(mode="RGB")
    img.save("temp.jpeg")
    return img

st.write("### Upload image")
image_file = st.file_uploader("Upload", type=["png","jpg","jpeg"])

if image_file is not None:
    file_details = {"filename":image_file.name, "filetype":image_file.type,"filesize":image_file.size}
    # st.write(file_details)
    st.write("#### Image Uploaded")
    st.image(load_image(image_file),width=250)

a = st.button("Generate")

if a:
    text = predict_step(["temp.jpeg"])
    st.write(text[0])
