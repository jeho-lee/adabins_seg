import numpy as np
from PIL import Image
import torch
import os

"""
Cityscapes colormap for visualization.
"""
# Official Cityscapes color map (class index to RGB color)
color_map = np.array([
    [128, 64, 128],    # Class 0: Road
    [244, 35, 232],    # Class 1: Sidewalk
    [70, 70, 70],      # Class 2: Building
    [102, 102, 156],   # Class 3: Wall
    [190, 153, 153],   # Class 4: Fence
    [153, 153, 153],   # Class 5: Pole
    [250, 170, 30],    # Class 6: Traffic light
    [220, 220, 0],     # Class 7: Traffic sign
    [107, 142, 35],    # Class 8: Vegetation
    [152, 251, 152],   # Class 9: Terrain
    [70, 130, 180],    # Class 10: Sky
    [220, 20, 60],     # Class 11: Person
    [255, 0, 0],       # Class 12: Rider
    [0, 0, 142],       # Class 13: Car
    [0, 0, 70],        # Class 14: Truck
    [0, 60, 100],      # Class 15: Bus
    [0, 80, 100],      # Class 16: Train
    [0, 0, 230],       # Class 17: Motorcycle
    [119, 11, 32],     # Class 18: Bicycle
], dtype=np.uint8)

# Add an entry for the ignore class (Class 255)
ignore_color = np.array([0, 0, 0], dtype=np.uint8)  # Black color for ignore class

# Create an extended color map that includes the ignore class
extended_color_map = np.zeros((256, 3), dtype=np.uint8)
extended_color_map[:19] = color_map  # First 19 entries are for valid classes
extended_color_map[255] = ignore_color  # The 255th entry is for the ignore class

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg
    
def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calculate the confusion matrix by given label and pred
    """
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_gt = np.asarray(
    label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=int)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index, minlength=num_class**2)
    confusion_matrix = label_count.reshape(num_class, num_class)

    return confusion_matrix

def visualize_segmentation(images, labels, predictions, save_dir, epoch, ignore_label=255, num_vis=10):
    """
    Visualize and save segmentation results for the first num_vis images in the batch.

    Args:
        images (Tensor): Input images tensor of shape (N, 3, H, W)
        labels (Tensor): Ground truth labels tensor of shape (N, H, W)
        predictions (Tensor): Predicted labels tensor of shape (N, H, W)
        save_dir (str): Directory to save images
        epoch (int): Current epoch number (used in filenames)
        num_vis (int, optional): Number of images to visualize from the batch (default: 10)
    """
    # Denormalize images
    mean = [0.485, 0.456, 0.406]  # Adjust if your dataset uses different values
    std = [0.229, 0.224, 0.225]

    # Move tensors to CPU and denormalize
    images = images.cpu()
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    images = images * std + mean
    images = (images * 255).clamp(0, 255).byte().numpy()  # Shape: (N, 3, H, W)

    labels = labels.cpu().numpy()          # Shape: (N, H, W)
    predictions = predictions.cpu().numpy()  # Shape: (N, H, W)

    batch_size = images.shape[0]
    num_vis = min(batch_size, num_vis)  # Ensure we don't exceed the batch size
    for i in range(num_vis):
        image = images[i].transpose(1, 2, 0)  # Convert to (H, W, 3)
        label = labels[i]
        prediction = predictions[i]

        # Map ignore labels (255) to the extended color map
        label_mapped = extended_color_map[label]
        prediction_mapped = extended_color_map[prediction]

        # Convert arrays to PIL Images
        image_pil = Image.fromarray(image)
        label_color = Image.fromarray(label_mapped.astype(np.uint8))  # Label colors
        prediction_color = Image.fromarray(prediction_mapped.astype(np.uint8))  # Prediction colors

        # Optionally, overlay the segmentation on the original image
        overlay = Image.blend(image_pil.convert('RGBA'), prediction_color.convert('RGBA'), alpha=0.5)

        # Use a generic filename that includes epoch and sample index
        filename_prefix = f"epoch{epoch}_sample{i}"
        image_pil.save(os.path.join(save_dir, f"{filename_prefix}_image.png"))
        label_color.save(os.path.join(save_dir, f"{filename_prefix}_label.png"))
        prediction_color.save(os.path.join(save_dir, f"{filename_prefix}_prediction.png"))
        overlay.save(os.path.join(save_dir, f"{filename_prefix}_overlay.png"))
