import streamlit as st
import torch
import segmentation_models_pytorch as smp
from torchvision import transforms
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download
from io import BytesIO

st.set_page_config(
    page_title="Low-Light Image Enhancement",
    page_icon="🌙",
    layout="wide"
)

st.title("🌙 Low-Light Image Enhancement")
st.write(
    "Upload a low-light image and enhance it using a **U-Net with ResNet34 encoder** trained on LOL v1 and LOL v2 datasets."
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=3,
        activation=None
    )
    
    model_path = hf_hub_download(
        repo_id="jeevikahunnurkar/low_light_img_enhancer",  
        filename="low_light_Unet.pth",  
    )
    
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    
    if isinstance(state_dict, dict):
        model.load_state_dict(state_dict)
    else:
        model = state_dict
        
    model.to(device)
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with st.spinner("Enhancing Image..."):
        with torch.no_grad():
            output = model(input_tensor)
            output = torch.clamp(output, 0, 1)

    output_image = (
        output.squeeze(0)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )

    output_image = (output_image * 255).astype(np.uint8)
    output_image = Image.fromarray(output_image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Enhanced Image")
        st.image(output_image, use_container_width=True)

    buffer = BytesIO()
    output_image.save(buffer, format="PNG")

    st.download_button(
        label="📥 Download Enhanced Image",
        data=buffer.getvalue(),
        file_name="enhanced_image.png",
        mime="image/png"
    )
