import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from segment_anything import sam_model_registry, SamPredictor
import tkinter as tk
from tkinter import filedialog

# Function to download the SAM checkpoint if it doesn't exist
def download_checkpoint(sam_checkpoint):
    if not os.path.exists(sam_checkpoint):
        import urllib.request
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        urllib.request.urlretrieve(url, sam_checkpoint)
        print(f"Downloaded checkpoint to {sam_checkpoint}")

# Function to find the largest rectangle inside a binary mask
def largest_rectangle_histogram(heights):
    stack = []
    max_area = 0
    left_bound = right_bound = top_bound = 0

    heights.append(0)  # Append a zero to flush out remaining stack at the end
    for i in range(len(heights)):
        while stack and heights[i] < heights[stack[-1]]:
            h = heights[stack.pop()]
            w = i if not stack else i - stack[-1] - 1
            area = h * w
            if area > max_area:
                max_area = area
                left_bound = stack[-1] + 1 if stack else 0
                right_bound = i - 1
                top_bound = h
        stack.append(i)

    heights.pop()  # Remove the appended zero
    return max_area, left_bound, right_bound, top_bound

def largest_rectangle_inside_boundary(mask):
    binary_mask = mask.astype(np.uint8)
    heights = [0] * binary_mask.shape[1]
    max_area = 0
    best_rect = (0, 0, 0, 0)

    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i, j] == 1:
                heights[j] += 1
            else:
                heights[j] = 0

        area, left, right, height = largest_rectangle_histogram(heights)
        if area > max_area:
            max_area = area
            best_rect = (left, i - height + 1, right - left + 1, height)

    return best_rect

# Function to select input and output folders
def select_folder(title):
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    folder_selected = filedialog.askdirectory(title=title)
    return folder_selected

# Function to show interactive image and allow the user to click points
def select_point(image_rgb, mask_predictor):
    point = None
    fig, ax = plt.subplots()
    ax.imshow(image_rgb)

    # Function to handle point clicks
    def onclick(event):
        nonlocal point
        if event.xdata is not None and event.ydata is not None:
            # Store the clicked point
            point = (int(event.xdata), int(event.ydata))
            # Perform mask prediction for the clicked point
            point_coords = np.array([point])
            point_labels = np.array([1])
            masks, scores, logits = mask_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False
            )
            # Update the image with the mask preview
            if len(masks) > 0:
                mask = masks[0]
                ax.clear()  # Clear the previous image and mask
                ax.imshow(image_rgb)
                ax.imshow(mask, cmap='jet', alpha=0.5)  # Preview mask overlay
                fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)

    # Create an OK button to confirm the selection
    ok_button = tk.Button(plt.gcf().canvas.manager.window, text='OK', command=lambda: plt.close(fig))
    ok_button.pack()

    # Show the interactive plot and wait for the user to close it
    plt.show()

    return point  # Return the clicked point

# Main script execution
if __name__ == "__main__":
    # Select input and output folders
    input_folder = select_folder("Select Input Folder")
    if not input_folder:
        print("No input folder selected. Exiting.")
        exit()

    output_folder = select_folder("Select Output Folder")
    if not output_folder:
        print("No output folder selected. Exiting.")
        exit()

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Download the SAM model checkpoint
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    download_checkpoint(sam_checkpoint)

    # Initialize the SAM model
    model_type = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_predictor = SamPredictor(sam)

    # Store selected points for all images
    selected_points = {}

    # Process images in the input folder
    input_extensions = ['*.JPG', '*.jpg', '*.JPEG', '*.jpeg', '*.PNG', '*.png']
    input_files = []
    
    # Loop through the extensions and gather files
    for ext in input_extensions:
        input_files.extend(glob.glob(os.path.join(input_folder, ext)))

    if not input_files:
        print("No images found in the input folder.")
    else:
        print("Processing the following images:")
        for image_path in input_files:
            print(image_path)

            # Read the image
            image_bgr = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            # Set the image for SAM prediction
            mask_predictor.set_image(image_rgb)

            # Let the user interactively select a point and preview the mask
            point = select_point(image_rgb, mask_predictor)
            selected_points[image_path] = point  # Store the selected point for the current image

        # After all images are selected, process the masks
        for image_path, point in selected_points.items():
            if point:
                # Read the image again (necessary for cropping)
                image_bgr = cv2.imread(image_path)
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

                # Set the image for SAM prediction again
                mask_predictor.set_image(image_rgb)

                # Predict mask for the selected point
                point_coords = np.array([point])
                point_labels = np.array([1])
                masks, scores, logits = mask_predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=False
                )

                print(f"Generated {len(masks)} masks for {image_path}.")

                # Check if any masks were generated
                if len(masks) > 0:
                    largest_rect = largest_rectangle_inside_boundary(masks[0])
                    print("Largest rectangle:", largest_rect)

                    x, y, w, h = largest_rect
                    cropped_image = image_rgb[y:y+h, x:x+w]

                    # Save cropped image
                    output_path = os.path.join(output_folder, f"cropped_{os.path.basename(image_path)}")
                    plt.imsave(output_path, cropped_image)
                    print(f"Cropped image saved: {output_path}")
                else:
                    print(f"No masks generated for {image_path}.")
