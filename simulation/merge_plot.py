#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 13:45:26 2024

@author: mmip
"""
#%%
from PIL import Image
directory = "results/results_for_rmd/res10/"

image_paths = [directory+"score_shapley.png",
               directory+"GRAD.png",
               directory+"GRAD_features.png",
               directory+"IG.png",
               directory+"GRAD_LM.png",
               directory+"IG_LM.png"]


# Open images and store them in a list
images = [Image.open(img) for img in image_paths]

# Assume all images are of the same size, get the size of the first image
width, height = images[0].size

# Create a new blank image with the appropriate size (2 rows, 3 columns)
new_image = Image.new('RGB', (width * 3, height * 2))

# Paste the images into the new image at the appropriate position
for i, img in enumerate(images):
    row = i // 3
    col = i % 3
    new_image.paste(img, (col * width, row * height))

# Save the merged image
new_image.save(directory+'merged_image.png')

#%%
directory = "results/results_for_rmd/res10/"

image_paths = [directory+"score_shapley_zoomed.png",
               directory+"GRAD_zoomed.png",
               directory+"GRAD_features_zoomed.png",
               directory+"IG_zoomed.png",
               directory+"GRAD_LM_zoomed.png",
               directory+"IG_LM_zoomed.png"]

images = [Image.open(img) for img in image_paths]

# Assume all images are of the same size, get the size of the first image
width, height = images[0].size

# Create a new blank image with the appropriate size (2 rows, 3 columns)
new_image = Image.new('RGB', (width * 3, height * 2))

# Paste the images into the new image at the appropriate position
for i, img in enumerate(images):
    row = i // 3
    col = i % 3
    new_image.paste(img, (col * width, row * height))

# Save the merged image
new_image.save(directory+'merged_image_zoomed.png')