def overlay_prediction_on_image(image, mask, alpha=0.5):
    import numpy as np
    import cv2
    import torch

    # Step 1: Move image to CPU and convert to numpy
    image = image.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()

    # Step 2: Unnormalize (assuming ImageNet mean/std)
    mean = np.array([0.485, 0.456, 0.406])[:, None, None]
    std = np.array([0.229, 0.224, 0.225])[:, None, None]
    image = (image * std + mean)  # now in [0, 1+]
    image = np.clip(image, 0, 1) * 255  # to [0, 255]
    image = image.astype(np.uint8).transpose(1, 2, 0)  # [H,W,3]

    # Step 3: Prepare binary mask
    mask = mask.squeeze()  # [H, W]
    mask = (mask > 0.5).astype(np.uint8) * 255

    # Step 4: Create a red mask overlay
    mask_colored = np.zeros_like(image)
    mask_colored[..., 0] = mask  # Red channel

    # Step 5: Overlay mask
    overlay = cv2.addWeighted(image, 1 - alpha, mask_colored, alpha, 0)
    return overlay
def unnormalize_image(image_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    image = image_tensor.cpu() * std + mean
    image = image.clamp(0, 1)
    return image.permute(1, 2, 0).numpy()
def predict_and_overlay(model, dataloader, num_samples=4):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    import cv2  # OpenCV used for blending

    with torch.no_grad():
        count = 0
        for batch in dataloader:
            inputs = batch['image'].to(device)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()

            for i in range(inputs.size(0)):
                if count >= num_samples:
                    return

                image = inputs[i]
                mask = preds[i]

                overlay = overlay_prediction_on_image(image, mask)

                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                ax[0].imshow(unnormalize_image(image))

                ax[0].set_title("Original Image")
                ax[0].axis("off")

                ax[1].imshow(mask.squeeze().cpu().numpy(), cmap='gray')
                ax[1].set_title("Predicted Mask")
                ax[1].axis("off")

                ax[2].imshow(overlay)
                ax[2].set_title("Overlay")
                ax[2].axis("off")

                plt.tight_layout()
                plt.show()

                count += 1
