import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# --- Streamlit page config ---
st.set_page_config(page_title="Crack Segmentation App", layout="wide")

# --- Albumentations Transform (Same as Training) ---
albumentations_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# --- Preprocessing Function ---
def preprocess_image(image):
    # Auto-rotate horizontal images to vertical
    if image.width > image.height:
        image = image.rotate(90, expand=True)

    image = np.array(image.convert("RGB"))
    augmented = albumentations_transform(image=image)
    return augmented["image"].unsqueeze(0)


# --- Unnormalize for visualization ---
def unnormalize_image(image_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    image = image_tensor.cpu() * std + mean
    image = image.clamp(0, 1)
    return image.permute(1, 2, 0).numpy()

# --- Overlay prediction on image ---
def overlay_prediction_on_image(image, mask, alpha=0.5):
    image = image.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])[:, None, None]
    std = np.array([0.229, 0.224, 0.225])[:, None, None]
    image = (image * std + mean)
    image = np.clip(image, 0, 1) * 255
    image = image.astype(np.uint8).transpose(1, 2, 0)

    mask = (mask.squeeze() > 0.5).astype(np.uint8) * 255
    mask_colored = np.zeros_like(image)
    mask_colored[..., 0] = mask
    overlay = cv2.addWeighted(image, 1 - alpha, mask_colored, alpha, 0)
    return overlay

# --- Define Model Architecture (match Jupyter version) ---
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, n_coefficients):
        super(AttentionBlock, self).__init__()
        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return skip_connection * psi


class AttentionUNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(AttentionUNet, self).__init__()
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock(img_ch, 64)
        self.Conv2 = ConvBlock(64, 128)
        self.Conv3 = ConvBlock(128, 256)
        self.Conv4 = ConvBlock(256, 512)
        self.Conv5 = ConvBlock(512, 1024)

        self.Up5 = UpConv(1024, 512)
        self.Att5 = AttentionBlock(F_g=512, F_l=512, n_coefficients=256)
        self.UpConv5 = ConvBlock(1024, 512)

        self.Up4 = UpConv(512, 256)
        self.Att4 = AttentionBlock(F_g=256, F_l=256, n_coefficients=128)
        self.UpConv4 = ConvBlock(512, 256)

        self.Up3 = UpConv(256, 128)
        self.Att3 = AttentionBlock(F_g=128, F_l=128, n_coefficients=64)
        self.UpConv3 = ConvBlock(256, 128)

        self.Up2 = UpConv(128, 64)
        self.Att2 = AttentionBlock(F_g=64, F_l=64, n_coefficients=32)
        self.UpConv2 = ConvBlock(128, 64)

        self.Conv = nn.Conv2d(64, output_ch, kernel_size=1)

    def forward(self, x):
        e1 = self.Conv1(x)
        e2 = self.Conv2(self.MaxPool(e1))
        e3 = self.Conv3(self.MaxPool(e2))
        e4 = self.Conv4(self.MaxPool(e3))
        e5 = self.Conv5(self.MaxPool(e4))

        d5 = self.Up5(e5)
        x4 = self.Att5(d5, e4)
        d5 = self.UpConv5(torch.cat((x4, d5), dim=1))

        d4 = self.Up4(d5)
        x3 = self.Att4(d4, e3)
        d4 = self.UpConv4(torch.cat((x3, d4), dim=1))

        d3 = self.Up3(d4)
        x2 = self.Att3(d3, e2)
        d3 = self.UpConv3(torch.cat((x2, d3), dim=1))

        d2 = self.Up2(d3)
        x1 = self.Att2(d2, e1)
        d2 = self.UpConv2(torch.cat((x1, d2), dim=1))

        out = self.Conv(d2)
        return out

# --- Load model ---
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionUNet().to(device)
    state_dict = torch.load("2000_attention_unet_best.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device

# --- Streamlit app logic ---
def main():
    st.title("Crack Segmentation Viewer")
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            model, device = load_model()
            input_tensor = preprocess_image(image).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                pred = (torch.sigmoid(output) > 0.5).float()

            img = input_tensor[0]
            mask = pred[0]
            overlay = overlay_prediction_on_image(img, mask)

            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(unnormalize_image(img)); ax[0].set_title("Original"); ax[0].axis("off")
            ax[1].imshow(mask.squeeze().cpu().numpy(), cmap="gray"); ax[1].set_title("Predicted Mask"); ax[1].axis("off")
            ax[2].imshow(overlay); ax[2].set_title("Overlay"); ax[2].axis("off")
            st.pyplot(fig)

if __name__ == "__main__":
    main()
