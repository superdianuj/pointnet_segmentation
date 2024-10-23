import cv2 as cv
from matplotlib import pyplot as plt
import os
import re

imgs=os.listdir('points')
ground_imgs=[img for img in imgs if 'ground' in img]
pred_imgs=[img for img in imgs if 'pred' in img]
# Sort pred_points list by extracting the first integer from each filename
pred_imgs.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))

# Sort ground_points list by extracting the first integer from each filename
ground_imgs.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))


num_imgs=len(pred_imgs)


fig = plt.figure(figsize=(10, num_imgs * 5))  # Adjust the figure size

# Add bold column titles for the two groups
fig.text(0.25, 0.99, 'Ground Truth', fontsize=32, fontweight='bold', ha='center')
fig.text(0.75, 0.99, 'Predicted', fontsize=32, fontweight='bold', ha='center')

for i in range(num_imgs):
    ground_img = ground_imgs[i]
    pred_img = pred_imgs[i]

    epoch = re.findall(r'\d+', ground_img)[0]

    ground_image = cv.imread('points/' + ground_img)
    ground_image = cv.cvtColor(ground_image, cv.COLOR_BGR2RGB)  # Convert to RGB
    # center crop
    h, w, _ = ground_image.shape
    ground_image = ground_image[:, w//4:3*w//4]


    pred_image = cv.imread('points/' + pred_img)
    pred_image = cv.cvtColor(pred_image, cv.COLOR_BGR2RGB)  # Convert to RGB
    # center crop
    h, w, _ = pred_image.shape
    pred_image = pred_image[:, w//4:3*w//4]

    # Plot ground truth image
    plt.subplot(num_imgs, 2, 2 * i + 1)
    plt.imshow(ground_image)
    plt.title(f"Epoch : {epoch}", fontsize=24)
    plt.axis('off')

    # Plot predicted image
    plt.subplot(num_imgs, 2, 2 * i + 2)
    plt.imshow(pred_image)
    plt.title(f"Epoch : {epoch}", fontsize=24)
    plt.axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.99])  # Leave space for the column titles
plt.savefig('combined.png')



