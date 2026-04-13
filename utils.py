import os
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import ResUNetPlusPlus

def rle_to_mask(rle_string, height=256, width=1600):
    """
    Convert RLE (Run-Length Encoding) string to binary mask
    """
    if not isinstance(rle_string, str) or rle_string == '':
        return np.zeros((height, width), dtype=np.uint8)
    
    mask = np.zeros(height * width, dtype=np.uint8)
    rle_numbers = [int(num) for num in rle_string.split()]
    
    for i in range(0, len(rle_numbers), 2):
        start = rle_numbers[i] - 1
        length = rle_numbers[i + 1]
        mask[start:start + length] = 1
    
    mask = mask.reshape((width, height)).T
    return mask

def get_inference_transform():
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def load_model(checkpoint_path, device):
    model = ResUNetPlusPlus(in_channels=3, num_classes=4).to(device)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    model.eval()
    return model

def predict(image_path, model, device, threshold=0.5):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image_rgb.shape[:2]
    
    transform = get_inference_transform()
    transformed = transform(image=image_rgb)
    image_tensor = transformed['image'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        output = torch.sigmoid(output)
    
    predictions = output.squeeze(0).cpu().numpy()
    
    # Process results
    results = []
    colors = [
        (255, 0, 0),    # Red for Class 1
        (0, 255, 0),    # Green for Class 2
        (0, 0, 255),    # Blue for Class 3
        (255, 255, 0)   # Yellow for Class 4
    ]
    class_names = ['Class 1', 'Class 2', 'Class 3', 'Class 4']
    
    combined_overlay = image_rgb.copy()
    
    defect_info = []
    for i in range(4):
        mask = predictions[i]
        binary_mask = (mask > threshold).astype(np.uint8)
        
        # Resize mask to original size
        binary_mask_resized = cv2.resize(binary_mask, (original_size[1], original_size[0]), 
                                         interpolation=cv2.INTER_NEAREST)
        
        confidence = 0.0
        if binary_mask.sum() > 0:
            confidence = float(mask[binary_mask == 1].mean())
            
            # Add to overlay
            color_mask = np.zeros_like(image_rgb)
            for c in range(3):
                color_mask[:, :, c] = binary_mask_resized * colors[i][c]
            combined_overlay = cv2.addWeighted(combined_overlay, 1, color_mask.astype(np.uint8), 0.4, 0)
            
            defect_info.append({
                'class': class_names[i],
                'confidence': f"{confidence:.3f}",
                'detected': True
            })
        else:
            defect_info.append({
                'class': class_names[i],
                'confidence': "0.000",
                'detected': False
            })
            
    return combined_overlay, defect_info
