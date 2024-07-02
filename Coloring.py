import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from colour import sRGB_to_XYZ, XYZ_to_Lab, Lab_to_XYZ, XYZ_to_sRGB
import numpy as np
import io
import pickle
import cv2

# Tentukan device untuk PyTorch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# st.write(f"Using device: {device}")

# Fungsi dan kelas model yang diberikan
def lab_to_rgb(L, ab, device):
    L = 100 * L
    ab = (ab - 0.5) * 256
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).detach().cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img = Lab_to_XYZ(img)
        img = XYZ_to_sRGB(img)
        rgb_imgs.append(img)
    return torch.tensor(np.stack(rgb_imgs, axis=0)).permute(0, 3, 1, 2).to(device)

def load_model(model_file):
    with open(model_file, 'rb') as f:
        loaded_model = pickle.load(f)
    return loaded_model

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )
        self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_ = x.detach().clone()
        x_ = self.block(x_)
        residual = self.identity(x)
        out = x_ + residual
        return self.relu(out)

class EncoderBlock(nn.Module):
    def __init__(self, in_chans, out_chans, sampling_factor=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(sampling_factor),
            ConvBlock(in_chans, out_chans)
        )

    def forward(self, x):
        return self.block(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_chans, out_chans, sampling_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.block = ConvBlock(in_chans + out_chans, out_chans)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, dropout_rate=0.1):
        super().__init__()
        self.encoder = nn.ModuleList([
            ConvBlock(in_channels, 64),
            EncoderBlock(64, 128),
            EncoderBlock(128, 256),
            EncoderBlock(256, 512),
        ])
        self.decoder = nn.ModuleList([
            DecoderBlock(512, 256),
            DecoderBlock(256, 128),
            DecoderBlock(128, 64)
        ])
        self.dropout = nn.Dropout2d(dropout_rate)
        self.logits = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        encoded = []
        for enc in self.encoder:
            x = enc(x)
            x = self.dropout(x)
            encoded.append(x)
        enc_out = encoded.pop()
        for dec in self.decoder:
            enc_out = encoded.pop()
            x = dec(x, enc_out)
        return F.sigmoid(self.logits(x))

# def colorize_image(model, input_image, device='cuda'):
#     transform = transforms.Compose([
#         transforms.Resize((256, 128), antialias=True),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=0, std=0.5)
#     ])
#     model.eval()
#     input_tensor = transform(input_image).unsqueeze(0).to(device)
#     with torch.no_grad():
#         output = model(input_tensor)
#     L = input_tensor * 100
#     ab = (output - 0.5) * 256
#     Lab = torch.cat([L, ab], dim=1).squeeze().permute(1, 2, 0).cpu().numpy()
#     rgb_image = XYZ_to_sRGB(Lab_to_XYZ(Lab))
#     rgb_image = np.clip(rgb_image, 0, 1)
#     rgb_image = (rgb_image * 255).astype(np.uint8)
#     return Image.fromarray(rgb_image)


def colorize_image(model, input_image, device='cuda', noise_level=0.1):
    transform = transforms.Compose([
        transforms.Resize((256, 128), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=0, std=0.5)
    ])
    model.eval()
    input_tensor = transform(input_image).unsqueeze(0).to(device)
    
    # Menambahkan noise acak pada tensor input
    noise = torch.randn_like(input_tensor) * noise_level
    input_tensor_noisy = input_tensor + noise
    
    with torch.no_grad():
        output = model(input_tensor_noisy)
    
    L = input_tensor * 100
    ab = (output - 0.5) * 256
    Lab = torch.cat([L, ab], dim=1).squeeze().permute(1, 2, 0).cpu().numpy()
    rgb_image = XYZ_to_sRGB(Lab_to_XYZ(Lab))
    rgb_image = np.clip(rgb_image, 0, 1)
    rgb_image = (rgb_image * 255).astype(np.uint8)
    return Image.fromarray(rgb_image)

def cal_colorful(Lab_image): 
    Lab_image = Lab_to_XYZ(Lab_image)  # Convert LAB to XYZ
    Lab_image = XYZ_to_sRGB(Lab_image)  # Convert XYZ to sRGB
    
    rg = Lab_image[:, 0] - Lab_image[:, 1] # Calculate the red-green channel
    yb = 0.5 * (Lab_image[:, 0] + Lab_image[:, 1]) - Lab_image[:, 2] # Calculate the yellow-blue channel

    mean_rgyb = np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2) # Calculate the mean of the RG and YB channels
    std_rgyb = np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)   # Calculate the standard deviation of the RG and YB channels

    colorfullnes = (std_rgyb + 0.3 * mean_rgyb) # Calculate the colourfulness score
    
    return round((colorfullnes), 4)    # Return the mean colourfulness score

def calculate_colourfulness(image):

    # Split image into R, G, B channels

    # (B, G, R) = cv2.split(image.astype("float"))
    # # Compute rg = R - G and yb = 0.5 * (R + G) - B
    # rg = np.absolute(R - G)
    # yb = np.absolute(0.5 * (R + G) - B)
    # # Compute the mean and standard deviation of both `rg` and `yb`
    # std_rg, mean_rg = (np.std(rg), np.mean(rg))
    # std_yb, mean_yb = (np.std(yb), np.mean(yb))
    # # Combine the mean and standard deviations
    # colourfulness = np.sqrt(std_rg ** 2 + std_yb ** 2) + 0.3 * np.sqrt(mean_rg ** 2 + mean_yb ** 2)
    return cal_colorful(image)

# Load model globally
model = load_model('models/generator_model.pkl')

# Streamlit interface
st.title("Image Colorization with CGAN")

# Upload file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert('L')

    col1, col2 = st.columns(2)

    # Kolom pertama untuk gambar inputan
    with col1:
        st.image(input_image, caption='Uploaded Image', width=200)
        # st.image(input_image, caption='Uploaded Image', use_column_width=True)

    with col2:
        # st.write("Colorizing...")
        # Colorize image
        # colorized_image = colorize_image(model, input_image, device)
        colorized_image = colorize_image(model, input_image, device, noise_level=0.1)
    
        # Display colorized image
        # st.image(colorized_image, caption='Colorized Image', use_column_width=True)
        st.image(colorized_image, caption='Colorized Image', width=200)

        # Display Colourfulness score
        colorfulness_score = calculate_colourfulness(colorized_image)
        st.markdown(f"**Colourfulness Score:** {colorfulness_score:.2f}")
        
        # Provide a download link for the colorized image
        buf = io.BytesIO()
        colorized_image.save(buf, format="JPEG")
        byte_im = buf.getvalue()
        st.download_button(
            label="Download colorized image",
            data=byte_im,
            file_name="colorized_image.jpg",
            mime="image/jpeg"
        )