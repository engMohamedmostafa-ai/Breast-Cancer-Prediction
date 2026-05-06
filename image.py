import numpy as np
import matplotlib.pyplot as plt

three_gray_img = np.array([
 [0, 0, 1, 0 , 0, 1, 0],
 [0, 0, 1, 0, 0, 1, 0],
 [0, 0, 1, 0, 0, 1, 0],
 [0, 0,1, 100, 1, 0, 0],
 [0, 0,1, 0, 0, 1, 0],
 [0, 0, 1, 0, 0, 1, 0],
 [0, 0, 1, 0 , 1, 0, 0]], dtype=np.uint8)
# Display the image using matplotlib
plt.imshow(three_gray_img, cmap='gray')
#plt.imshow(three_gray_img) # default cmap='viridis'
plt.axis('off') # Hide axis
plt.title('Digit Three grayscale Image') # set the title
plt.colorbar()