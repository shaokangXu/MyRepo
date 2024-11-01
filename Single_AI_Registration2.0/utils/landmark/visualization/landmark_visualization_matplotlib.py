import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from utils.io.common import create_directories_for_file_name
from utils.landmark.visualization.landmark_visualization_base import LandmarkVisualizationBase


class LandmarkVisualizationMatplotlib(LandmarkVisualizationBase):
    """
    Class for landmark groundtruth and prediction visualization. Also performs axis projection for 3D images. Uses matplotlib for visualization.
    """
    def __init__(self, figsize=(10, 5), dpi=100, annotation_fontsize=12, *args, **kwargs):
        """
        Initializer.
        :param figsize: The figsize for plt.subplots.
        :param dpi: The dpi of the plot.
        :param annotation_fontsize: The fontsize of the annotation.
        :param args: *args.
        :param kwargs: **kwargs.
        """
        super(LandmarkVisualizationMatplotlib, self).__init__(*args, **kwargs)
        self.figsize = figsize
        self.dpi = dpi
        self.annotation_fontsize = annotation_fontsize

    def prepare_image_canvas(self, image_np_list):
        """
        Prepares a canvas (e.g., np image or matplotlib axis) for each projection image.
        :param image_np_list: A list of np images, representing the image projections.
        :return: A list of image canvases. Will be given to merge_image_canvas.
        """
        fig, ax = plt.subplots(nrows=1, ncols=len(image_np_list), figsize=self.figsize, dpi=self.dpi, squeeze=False, frameon=False)
        ax = ax[0]  # squeeze only first dimension
        for i, image_np in enumerate(image_np_list):
            ax[i].imshow(np.stack([image_np] * 3, axis=-1))
            ax[i].axis('off')
        return ax

    def merge_image_canvas(self, image_canvas_list):
        """
        Merge image canvases to a single image.
        :param image_canvas_list: A list of image canvas objects (returned by prepare_image_canvas).
        :return: An image canvas.
        """
        for image_canvas in image_canvas_list:
            if self.flip_y_axis:
                image_canvas.invert_yaxis()
        return None

    def save(self, image_canvas_merged, filename):
        """
        Save the merged image canvas to the filename.
        :param image_canvas_merged: A merged image canvas (returned by merge_image_canvas).
        :param filename: The filename to save the image canvas to.
        """
        create_directories_for_file_name(filename)
        plt.tight_layout()
        plt.savefig(filename, dpi=self.dpi)
        plt.close()

    def visualize_landmark(self, image_canvas, landmark, color, annotation, annotation_color):
        """
        Visualize a single landmark.
        :param image_canvas: The image canvas object.
        :param landmark: The landmark.
        :param color: The landmark color.
        :param annotation: The annotation string.
        :param annotation_color: The annotation color.
        """
        coords = landmark.coords
        if self.spacing is not None:
            coords = [c / s for c, s in zip(coords, self.spacing)]
        p = mpatches.Circle((coords[0], coords[1]), self.radius, facecolor=[c / 255.0 for c in color], edgecolor=None)
        image_canvas.add_patch(p)
        if annotation is not None:
            image_canvas.text(coords[0] + 1.5 * self.radius, coords[1], f'{annotation}', fontsize=self.annotation_fontsize, horizontalalignment='left', verticalalignment='center', color=[c / 255.0 for c in annotation_color])

    def visualize_from_to_landmark_offset(self, image_canvas, landmark_from, landmark_to, color):
        """
        Visualize the offset between to landmarks.
        :param image_canvas: The image canvas object.
        :param landmark_from: The from landmark.
        :param landmark_to: The to landmark.
        :param color: The landmark color.
        """
        p = mpatches.Arrow(landmark_from.coords[0], landmark_from.coords[1], landmark_to.coords[0] - landmark_from.coords[0], landmark_to.coords[1] - landmark_from.coords[1], color=[c / 255.0 for c in color])
        image_canvas.add_patch(p)
