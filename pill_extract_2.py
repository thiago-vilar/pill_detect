import os
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import stag
from rembg import remove

class ExtractFeatures:
    def __init__(self, stag_id, med_type, image_path, neg_img_path):
        self.stag_id = stag_id
        self.med_type = med_type
        self.image_path = image_path
        self.neg_img_path = neg_img_path
        self.load_images() 
        self.corners = None
        self.ids = None
        self.homogenized_image = None
        self.scan_areas = {}
        self.pixel_size_mm = None
        self.data = []

    def load_images(self):
        """Load the primary and negative images from the specified paths."""
        self.image = cv2.imread(self.image_path)
        self.neg_image = cv2.imread(self.neg_img_path)
        if self.image is None or self.neg_image is None:
            raise ValueError("One or both images could not be loaded, please check the paths.")
        
    def detect_stag(self, ids, corners):
        """Detects a specific stag marker in the image and initializes measurements for pixel size."""
        config = {'libraryHD': 17, 'errorCorrection': -1}
        self.corners, self.ids, _ = stag.detectMarkers(self.image, **config)
        if self.ids is not None and self.stag_id in self.ids:
            index = np.where(self.ids == self.stag_id)[0][0]
            self.corners = self.corners[index].reshape(-1, 2)
            self.calculate_pixel_size_mm()
            return True
        print("Marker with ID", self.stag_id, "not found.")
        return False

    def calculate_pixel_size_mm(self):
        """Calculates the pixel size in millimeters using the stag marker as reference."""
        if self.corners is not None:
            width_px = np.max(self.corners[:, 0]) - np.min(self.corners[:, 0])
            self.pixel_size_mm = 20.0 / width_px 

    def homogenize_image_based_on_corners(self, image, corners):
        """Normalizes the image perspective based on the detected stag corners."""
        if self.corners is None:
            print("Corners not detected.")
            return None
        x, y, w, h = cv2.boundingRect(self.corners.astype(np.float32))
        aligned_corners = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ], dtype='float32')
        transform_matrix = cv2.getPerspectiveTransform(self.corners, aligned_corners)
        self.homogenized_image = cv2.warpPerspective(self.image, transform_matrix, (self.image.shape[1], self.image.shape[0]))
        return self.homogenized_image

    def display_scan_area_by_markers(self, med_type):
        """Displays the scan area on the homogenized image based on the stag location."""
        if self.homogenized_image is None:
            print("Homogenized image is not available.")
            return None
        corner = self.corners.reshape(-1, 2).astype(int)
        centroid_x = int(np.mean(corner[:, 0]))
        centroid_y = int(np.mean(corner[:, 1]))
        cv2.putText(self.homogenized_image, f'ID:{self.stag_id}', (centroid_x + 45, centroid_y -15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        if med_type == "Pill":  
            width = np.max(corner[:, 0]) - np.min(corner[:, 0])
            pixel_size_mm = width / 20
            crop_width = int(30 * pixel_size_mm)
            crop_height = int(60 * pixel_size_mm)
            crop_y_adjustment = int(10 * pixel_size_mm)
            x_min = max(centroid_x - crop_width, 0)
            x_max = min(centroid_x + crop_width, self.homogenized_image.shape[1])
            y_min = max(centroid_y - crop_height - crop_y_adjustment, 0)
            y_max = max(centroid_y - crop_y_adjustment, 0)
            cv2.rectangle(self.homogenized_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
            self.scan_areas[self.stag_id] = (x_min, x_max, y_min, y_max)
            return self.homogenized_image
        else:
            print(f"It's not a red Pill!")

    def crop_scan_area(self):
        """Crops the defined scan area from the homogenized image for further processing."""
        if self.stag_id not in self.scan_areas:
            print(f'ID {self.stag_id} not found.')
            return None
        x_min, x_max, y_min, y_max = self.scan_areas[self.stag_id]
        cropped_image = self.homogenized_image[y_min:y_max, x_min:x_max]
        return cropped_image
    
    def subtract_images(self, image1, image2):
        """Subtracts the second image from the first image."""
        if image1.shape != image2.shape:
            print("Images do not match in size and cannot be directly subtracted.")
            return None
        return cv2.subtract(image1, image2)

    def process_images(self):
        """images."""
        image_med, image_sub = self.detect_stag(self.image, self.neg_image)
        homog_1, homog_2 = self.homogenize_image_based_on_corners(image_med, image_sub)
        plt.imshow(cv2.cvtColor(homog_1, homog_2, cv2.COLOR_BGR2RGB))
        plt.title('Homogenized Image')
        plt.show()
        marked_image_1, marked_image_2 = self.display_scan_area_by_markers(self.med_type)
        if (marked_image_1 is not None) and (marked_image_2 is not None):
            plt.imshow(cv2.cvtColor(marked_image_1, marked_image_2, cv2.COLOR_BGR2RGB))
            plt.title('Marked Scan Areas')
            plt.show()
            cropped_1, cropped_2 = self.crop_scan_area(marked_image_1, marked_image_2)
            if (cropped_1 is not None) and (cropped_2 is not None) :
                    plt.imshow(cv2.cvtColor(cropped_1, cropped_2, cv2.COLOR_BGR2RGB))
                    plt.title('Cropped Scan Areas')
                    plt.show()



def main():
    stag_id = 2
    med_type = "Pill"
    image_path = ".\\thiago_fotos_MED\\img_2_002.jpg"
    neg_img_path = ".\\thiago_fotos_SUB\\img_2_002.jpg"
    processor = ExtractFeatures(stag_id, med_type, image_path, neg_img_path)
    processor.process_images()



if __name__ == "__main__":
    main()
