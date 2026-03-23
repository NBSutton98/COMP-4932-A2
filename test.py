import cv2
import numpy as np

# A list of the time steps 
time_steps = [1, 2, 3, 4, 5, 6, 7, 8]

for t in time_steps:
    # 1. Construct the paths to BOTH images for this time step
    path0 = f"images/W0.t{t}.jpg"
    path1 = f"images/W1.t{t}.jpg"
    
    # 2. Load the images into variables
    img0 = cv2.imread(path0)
    img1 = cv2.imread(path1)
    img0 = img0.astype(float)
    img1 = img1.astype(float)
    alpha = t / (max(time_steps) + 1)
    blending = (1-alpha) * img0 + alpha * img1
    # 4. Constrain values to the 0-255 range
    blending = np.clip(blending, 0, 255)

    # 5. Convert back to the image format (uint8)
    blending = blending.astype(np.uint8)

    # 6. Save the result!
    cv2.imwrite(f"blended_t{t}.jpg", blending)   
