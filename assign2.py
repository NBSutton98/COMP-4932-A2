import numpy as np
import math
import matplotlib.pyplot as plt

# --- Step 1: Decode the 64-vector from the RLC sequence ---
dcVal = -26
acVal = [(0, -3), (1, 3), (0, 2), (0, -6), (0, 2), (0, -4), (0, 1), (0, -3), (0, 1), 
         (0, 1), (0, 5), (0, 1), (0, 2), (0, 1), (0, 2), (5, -1), (0, 1), (0, 0)]

vector = [0] * 64
vector[0] = dcVal  # Set the first index to the DC value
pos = 1            # Start at index 1 for AC values

for run, val in acVal:
    if run == 0 and val == 0: 
        break
    pos += run        # Skip the zeros
    vector[pos] = val # Place the value
    pos += 1          # Move to the next available slot

# --- Step 2: Reconstruct the 8x8 block using zig-zag scan ---
zigzag_indices = [
    (0,0), (0,1), (1,0), (2,0), (1,1), (0,2), (0,3), (1,2),
    (2,1), (3,0), (4,0), (3,1), (2,2), (1,3), (0,4), (0,5),
    (1,4), (2,3), (3,2), (4,1), (5,0), (6,0), (5,1), (4,2),
    (3,3), (2,4), (1,5), (0,6), (0,7), (1,6), (2,5), (3,4),
    (4,3), (5,2), (6,1), (7,0), (7,1), (6,2), (5,3), (4,4),
    (3,5), (2,6), (1,7), (2,7), (3,6), (4,5), (5,4), (6,3),
    (7,2), (7,3), (6,4), (5,5), (4,6), (3,7), (4,7), (5,6),
    (6,5), (7,4), (7,5), (6,6), (5,7), (6,7), (7,6), (7,7)
]

dct_matrix = np.zeros((8, 8))
for i in range(64):
    row, col = zigzag_indices[i]
    dct_matrix[row][col] = vector[i]

# --- Step 3: Multiply by the luminance quantization table ---
Q = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

f_uv = dct_matrix * Q

# Step 4: Perform a 2D inverse discrete cosine transform 
reconstructed_pixels = np.zeros((8, 8))

def C(xi):
    if xi == 0:
        return math.sqrt(2) / 2
    else:
        return 1.0

for i in range(8):
    for j in range(8):
        sum_val = 0.0
        for u in range(8):
            for v in range(8):
                cos_u = math.cos(((2 * i + 1) * u * math.pi) / 16)
                cos_v = math.cos(((2 * j + 1) * v * math.pi) / 16)
                term = (C(u) * C(v) / 4.0) * cos_u * cos_v * f_uv[u, v]
                sum_val += term
        reconstructed_pixels[i, j] = sum_val

# Step 5: Add 128 to each entry to scale back to 0-255 ---
final_image = reconstructed_pixels + 128
final_image = np.round(final_image).astype(int)

# Console Output (Deliverables) 
checkpoint_matrix = np.round(reconstructed_pixels).astype(int)
print("--- Checkpoint Matrix (Step 4) ---")
for row in checkpoint_matrix:
    print(" ".join(f"{val:4d}" for val in row))

print("\n--- Final Reconstructed Pixels (Step 5) ---")
for row in final_image:
    print(" ".join(f"{val:4d}" for val in row))

# --- Bonus: Display and Visually Compare Images ---
# Hardcoded original image from the assignment PDF 
original_image = np.array([
    [52, 55, 61, 66, 70, 61, 64, 73],
    [63, 59, 55, 90, 109, 85, 69, 72],
    [62, 59, 68, 113, 144, 104, 66, 73],
    [63, 58, 71, 122, 154, 106, 70, 69],
    [67, 61, 68, 104, 126, 88, 68, 70],
    [79, 65, 60, 70, 77, 68, 58, 75],
    [85, 71, 64, 59, 55, 61, 65, 83],
    [87, 79, 69, 68, 65, 76, 78, 94] 
])

# Create a side-by-side plot using matplotlib
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot Original Image
im1 = axes[0].imshow(original_image, cmap='gray', vmin=0, vmax=255)
axes[0].set_title('Original 8x8 Block')
axes[0].axis('off')

# Plot Decoded Image
im2 = axes[1].imshow(final_image, cmap='gray', vmin=0, vmax=255)
axes[1].set_title('Decoded 8x8 Block (Reconstructed)')
axes[1].axis('off')

plt.tight_layout()
plt.show()