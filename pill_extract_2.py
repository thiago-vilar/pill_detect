import cv2
import numpy as np
from matplotlib import pyplot as plt
import stag

class ExtractFeatures:
    def __init__(self, image_path, neg_image_path, stag_id, med_type):
        """Initializes the class with paths to the images, an identifier for the stag marker, and the type of medicine."""
        self.image_path = image_path
        self.neg_image_path = neg_image_path
        self.stag_id = stag_id
        self.med_type = med_type
        self.load_images()
        self.corners = None
        self.scan_areas = {}

    def load_images(self):
        """Load the primary and negative images from the specified paths."""
        self.image = cv2.imread(self.image_path)
        self.neg_image = cv2.imread(self.neg_image_path)
        if self.image is None or self.neg_image is None:
            raise ValueError("One or both images could not be loaded. Please check the paths.")

    def detect_stag(self, image):
        """Detects a specific stag marker in the image and initializes measurements for pixel size."""
        config = {'libraryHD': 17, 'errorCorrection': 0}
        corners, ids, _ = stag.detectMarkers(image, **config)
        if ids is not None and self.stag_id in ids:
            index = np.where(ids == self.stag_id)[0][0]
            self.corners = corners[index].reshape(-1, 2)
            return self.corners
        print(f"Marker with ID {self.stag_id} not found in one of the images.")
        return None

    def homogenize_image_based_on_corners(self, image, corners):
        """Normalizes the image perspective based on the detected stag corners."""
        if corners is None:
            print("Corners not detected.")
            return None
        x, y, w, h = cv2.boundingRect(corners.astype(np.float32))
        aligned_corners = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ], dtype='float32')
        transform_matrix = cv2.getPerspectiveTransform(corners, aligned_corners)
        return cv2.warpPerspective(image, transform_matrix, (image.shape[1], image.shape[0]))

    def display_scan_area_by_markers(self, image):
        """Displays the scan area on the homogenized image based on the stag location."""
        if image is None or self.corners is None:
            print("Homogenized image or corners are not available.")
            return None
        corner = self.corners.astype(int)
        centroid_x = int(np.mean(corner[:, 0]))
        centroid_y = int(np.mean(corner[:, 1]))
        cv2.putText(image, f'ID:{self.stag_id}', (centroid_x + 45, centroid_y -15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

        width = np.max(corner[:, 0]) - np.min(corner[:, 0])
        pixel_size_mm = width / 20
        crop_width = int(30 * pixel_size_mm)
        crop_height = int(60 if self.med_type == "Pill" else 75 * pixel_size_mm)
        crop_y_adjustment = int(10 * pixel_size_mm)

        x_min = max(centroid_x - crop_width, 0)
        x_max = min(centroid_x + crop_width, image.shape[1])
        y_min = max(centroid_y - crop_height - crop_y_adjustment, 0)
        y_max = max(centroid_y - crop_y_adjustment, 0)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        
        self.scan_areas[self.stag_id] = (x_min, x_max, y_min, y_max)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Image with Scan Area')
        plt.show()
        return (x_min, x_max, y_min, y_max)

    def crop_scan_area(self, image, crop_coords):
        """Crops the defined scan area from the homogenized image for further processing."""
        x_min, x_max, y_min, y_max = crop_coords
        cropped_image = image[y_min:y_max, x_min:x_max]
        return cropped_image

    def process_images(self):
        """Process both images, detect markers, homogenize, display, and subtract the images."""
        corners_image = self.detect_stag(self.image)
        corners_neg_image = self.detect_stag(self.neg_image)

        if corners_image is not None and corners_neg_image is not None:
            homogenized_image = self.homogenize_image_based_on_corners(self.image, corners_image)
            homogenized_neg_image = self.homogenize_image_based_on_corners(self.neg_image, corners_neg_image)

            crop_coords_image = self.display_scan_area_by_markers(homogenized_image)
            crop_coords_neg_image = self.display_scan_area_by_markers(homogenized_neg_image)

            cropped_image = self.crop_scan_area(homogenized_image, crop_coords_image)
            cropped_neg_image = self.crop_scan_area(homogenized_neg_image, crop_coords_neg_image)

            subtracted_image = cv2.subtract(cropped_image, cropped_neg_image)
            self.display_image(subtracted_image, 'Subtracted Image')
        else:
            print("Markers not found in one or both images.")

    def display_image(self, image, title):
        """Displays the image."""
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.show()

def main():
    stag_id = 2
    med_type = "Pill"
    image_path = ".\\thiago_fotos_MED\\img_2_002.jpg"
    neg_image_path = ".\\thiago_fotos_SUB\\img_2_002.jpg"
    
    processor = ExtractFeatures(image_path, neg_image_path, stag_id, med_type)
    processor.process_images()

if __name__ == "__main__":
    main()
