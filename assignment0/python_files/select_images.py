import os
import matplotlib.pyplot as plt
from PIL import Image
import random

class ImageFolder:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.image_files = sorted([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
        self.labels = ['flower', 'dog', 'flower', 'dog', 'dog']

    def get_image_labels(self):
        return self.labels


    def display_select_images(self, indices_of_kept_images, eval_mode = False):
        all_image_paths = self.image_files
        paths_of_images_to_display = []

        ##############################################################################
        # TODO: use 'indiced_of_kept_images' and 'all_image_paths' to get a list of  #
        # 'paths_of_images_to_display', which will then be passed into the display   #
        # function                                                                   #
        ##############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        paths_of_images_to_display = [all_image_paths[i] for i in indices_of_kept_images]

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        if eval_mode:
            return paths_of_images_to_display

        self.display(paths_of_images_to_display)


    def display_all_images_random_order(self):
        """Display images in the folder using matplotlib, horizontally and without axes."""
        shuffled_list = self.image_files.copy()
        random.shuffle(shuffled_list)
        self.display(shuffled_list)


    def display(self, image_paths):
        num_images = len(image_paths)
        fig, axes = plt.subplots(1, num_images, figsize=(num_images * 5, 5))  # Adjust figsize as needed
        if num_images == 1:  # If there's only one image, axes won't be a list
            axes = [axes]
        for ax, image_file in zip(axes, image_paths):
            img_path = os.path.join(self.folder_path, image_file)
            img = Image.open(img_path)
            ax.imshow(img)
            ax.axis('off')  # This hides the axis

        plt.show()
