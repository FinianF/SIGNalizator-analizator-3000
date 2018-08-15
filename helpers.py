# Helper functions
import cv2
import os
import glob  # library for loading images from a directory


# This function loads in images and their labels and places them in a list
# The list contains all images and their associated labels
# For example, after data is loaded, im_list[0][:] will be the first image-label pair in the list
def load_dataset(image_dir):
    # Populate this empty image list
    im_list = []
    image_types = ["none", "pedistrain", "no_drive","stop","way_out","no_entry","road_works","parking","a_unevenness"]

    # Iterate through each color folder
    for im_type in image_types:

        # Iterate through each image file in each image_type folder
        # glob reads in any image with the extension "image_dir/im_type/*"
        for file in glob.glob(os.path.join(image_dir, im_type, "*")):

            # Read in the image
            # im = mpimg.imread(file)
            im = cv2.imread(file)

            # Check if the image exists/if it's been correctly read-in
            if not im is None:
                # Append the image, and it's type ("none", "pedistrain", "no_drive","stop","way-out","no_entry","road_works","parking","a_unevenness") to the image list
                im_list.append((im, im_type))

    return im_list


