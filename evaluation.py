# Numerical operations and array processing
import numpy as np

# Deep learning and neural networks
import torch

# Image quality assessment
from skimage.metrics import structural_similarity as ssim

# Color space conversions
from colour import sRGB_to_XYZ, XYZ_to_Lab, Lab_to_XYZ, XYZ_to_sRGB


def calculate_ssim(L, ab_gen, ab_gt):
    """
    Calculate the Structural Similarity Index (SSIM)
    """
    L = 100 * L # Scale the L component from 0-1 to 0-100
    ab_gen = (ab_gen - 0.5) * 256 # Adjust the a and b components to the correct range
    ab_gt = (ab_gt - 0.5) * 256 # Adjust the a and b components to the correct range

    Lab_gens = torch.cat([L, ab_gen], dim=1).permute(0, 2, 3, 1).detach().cpu().numpy() # Combine L, a, b, and rearrange the format for processing
    Lab_gts = torch.cat([L, ab_gt], dim=1).permute(0, 2, 3, 1).detach().cpu().numpy()   # Combine L, a, b, and rearrange the format for processing
    
    scores = [] # Initialize a list to store the SSIM scores
    for img_gen, img_gt in zip(Lab_gens, Lab_gts):  # Iterate over the generated and ground truth images
        img_gen = Lab_to_XYZ(img_gen)   # Convert LAB to XYZ
        img_gen = XYZ_to_sRGB(img_gen)  # Convert XYZ to sRGB
        
        img_gt = Lab_to_XYZ(img_gt)    # Convert LAB to XYZ
        img_gt = XYZ_to_sRGB(img_gt)    # Convert XYZ to sRGB
        
        scores.append(ssim(img_gen, img_gt, channel_axis=-1, data_range=255)) # Calculate the SSIM score
    
    return round(np.mean(scores), 4)    # Return the mean SSIM score

def calculate_colourfulness(L, ab_gen): 
    """
    Calculate the Colourfulness.
    References
    ----------
    D. Hasler and S.E.Suesstrunk, ``Measuring colorfulness in natural images," Human
    Vision andElectronicImagingVIII, Proceedings of the SPIE, 5007:87-95, 2003.
    ----------
    """
    L = 100 * L     # Scale the L component from 0-1 to 0-100
    ab_gen = (ab_gen - 0.5) * 256   # Adjust the a and b components to the correct range

    Lab_gens = torch.cat([L, ab_gen], dim=1).permute(0, 2, 3, 1).detach().cpu().numpy()     # Combine L, a, b, and rearrange the format for processing
    
    scores = []     # Initialize a list to store the colourfulness scores
    for img_gen in Lab_gens:   # Iterate over the generated images
        img_gen = Lab_to_XYZ(img_gen)  # Convert LAB to XYZ
        img_gen = XYZ_to_sRGB(img_gen)  # Convert XYZ to sRGB
        
        rg = img_gen[:, 0] - img_gen[:, 1] # Calculate the red-green channel
        yb = 0.5 * (img_gen[:, 0] + img_gen[:, 1]) - img_gen[:, 2] # Calculate the yellow-blue channel

        mean_rgyb = np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2) # Calculate the mean of the RG and YB channels
        std_rgyb = np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)   # Calculate the standard deviation of the RG and YB channels

        scores.append(std_rgyb + 0.3 * mean_rgyb) # Calculate the colourfulness score
    
    return round(np.mean(scores), 4)    # Return the mean colourfulness score