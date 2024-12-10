import os
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import stag
from rembg import remove

class ExtractFeatures:
    def __init__(self, image_path, neg_image_path, stag_id, med_type):
        self.image_path = image_path
        self.neg_image_path = neg_image_path
        self.stag_id = stag_id
        self.med_type = med_type
        self.pixel_size_mm = None
        self.load_images()
        self.corners = None
        self.scan_areas = {}
        self.preload_rembg()

    def preload_rembg(self):
        dummy_image = np.full((100, 100, 3), 128, dtype=np.uint8)
        try:
            is_success, buffer = cv2.imencode(".png", dummy_image)
            if is_success:
                remove(buffer.tobytes())
            print("rembg model preloaded successfully.")
        except Exception as e:
            print(f"Failed to preload rembg model: {e}")

    def load_images(self):
        self.image = cv2.imread(self.image_path)
        self.neg_image = cv2.imread(self.neg_image_path)
        if self.image is None or self.neg_image is None:
            raise ValueError("One or both images could not be loaded. Please check the paths.")

    def detect_stag(self, image):
        config = {'libraryHD': 17, 'errorCorrection': 0}
        corners, ids, _ = stag.detectMarkers(image, **config)
        if ids is not None and self.stag_id in ids:
            index = np.where(ids == self.stag_id)[0][0]
            self.corners = corners[index].reshape(-1, 2)
            self.calculate_pixel_size_mm()
            return self.corners
        print(f"Marker with ID {self.stag_id} not found in one of the images.")
        return None

    def calculate_pixel_size_mm(self):
        if self.corners is not None:
            width_px = np.max(self.corners[:, 0]) - np.min(self.corners[:, 0])
            self.pixel_size_mm = 20.0 / width_px

    def homogenize_image_based_on_corners(self, image, corners):
        if corners is None:
            print("Corners not detected.")
            return None
        x, y, w, h = cv2.boundingRect(corners.astype(np.float32))
        aligned_corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype='float32')
        transform_matrix = cv2.getPerspectiveTransform(corners, aligned_corners)
        return cv2.warpPerspective(image, transform_matrix, (image.shape[1], image.shape[0]))

    def display_scan_area_by_markers(self, image):
        if self.pixel_size_mm is None or image is None or self.corners is None:
            return None
        corners_int = self.corners.astype(int)
        centroid_x = int(np.mean(corners_int[:, 0]))
        centroid_y = int(np.mean(corners_int[:, 1]))
        cv2.putText(image, f'ID:{self.stag_id}', (centroid_x + 45, centroid_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        marker_width_px = np.max(corners_int[:, 0]) - np.min(corners_int[:, 0])
        pixel_size_mm = marker_width_px / 20.0
        crop_width = int(30 * pixel_size_mm)
        crop_height = int(60 * pixel_size_mm if self.med_type == "Pill" else 75 * pixel_size_mm)
        crop_y_adjustment = int(10 * pixel_size_mm)
        x_min = max(centroid_x - crop_width, 0)
        x_max = min(centroid_x + crop_width, image.shape[1])
        y_min = max(centroid_y - crop_height - crop_y_adjustment, 0)
        y_max = min(centroid_y - crop_y_adjustment, image.shape[0])
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        self.scan_areas[self.stag_id] = (x_min, x_max, y_min, y_max)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Image with Scan Area')
        plt.show()
        return (x_min, x_max, y_min, y_max)

    def crop_scan_area(self, image, crop_coords):
        x_min, x_max, y_min, y_max = crop_coords
        return image[y_min:y_max, x_min:x_max]

    def filter_pdi(self, image):
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
        filter = np.array([[0, 1, 0], [1, -4.5, 1], [0, 1, 0]], dtype=float)
        ddepth = cv2.CV_16S
        img_filtered = cv2.filter2D(img_blur, ddepth, filter)
        return cv2.convertScaleAbs(img_filtered)

    def remove_background(self, image_np_array):
        is_success, buffer = cv2.imencode(".png", image_np_array)
        if not is_success:
            raise ValueError("Failed to encode image for background removal.")
        output_image = remove(buffer.tobytes())
        return cv2.imdecode(np.frombuffer(output_image, np.uint8), cv2.IMREAD_UNCHANGED)

    def calculate_histograms(self, img_med):
        histograms = {}
        if img_med is not None and img_med.ndim == 3:
            img_rgb = cv2.cvtColor(img_med, cv2.COLOR_BGR2RGB)
            colors = ('r', 'g', 'b')
            for i, color in enumerate(colors):
                hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
                histograms[color] = hist.flatten()
            img_gray = cv2.cvtColor(img_med, cv2.COLOR_RGB2GRAY)
            hist_gray = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
            histograms['gray'] = hist_gray.flatten()
        return histograms

    def display_histogram(self, histograms, title):
        if histograms:
            plt.figure(figsize=(10, 5))
            plt.title(title)
            colors = {'r': 'red', 'g': 'green', 'b': 'blue', 'gray': 'gray'}
            for channel, hist in histograms.items():
                plt.plot(hist, color=colors[channel], label=f'{channel.upper()} channel')
            plt.xlim([0, 256])
            plt.xlabel('Pixel value')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            print("No histogram data to display.")

    def create_mask(self, img):
        if img.shape[2] == 4:
            img = img[:, :, :3]
        lower_bound = np.array([30, 30, 30], dtype=np.uint8)
        upper_bound = np.array([255, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(img, lower_bound, upper_bound)
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGBA)

    def find_and_draw_contours(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if largest_contour.size > 0:
                mask_with_contours = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA)
                cv2.drawContours(mask_with_contours, [largest_contour], -1, (0, 0, 255, 255), 2)
                return mask_with_contours, largest_contour
        return None, None

    def compute_chain_code(self, contour):
        start_point = contour[0][0]
        current_point = start_point
        chain_code = []
        moves = {(-1, 0): 3, (-1, 1): 2, (0, 1): 1, (1, 1): 0, (1, 0): 7, (1, -1): 6, (0, -1): 5, (-1, -1): 4}
        for i in range(1, len(contour)):
            next_point = contour[i][0]
            dx = next_point[0] - current_point[0]
            dy = next_point[1] - current_point[1]
            move = (dx // abs(dx) if dx != 0 else 0, dy // abs(dy) if dy != 0 else 0)
            if move in moves:
                chain_code.append(moves[move])
            current_point = next_point
        # Close the loop to ensure complete coverage
        dx = start_point[0] - current_point[0]
        dy = start_point[1] - current_point[1]
        move = (dx // abs(dx) if dx != 0 else 0, dy // abs(dy) if dy != 0 else 0)
        if move in moves:
            chain_code.append(moves[move])
        return chain_code, len(chain_code)

    def draw_chain_code(self, img_med, contour, chain_code):
        start_point = tuple(contour[0][0])
        current_point = start_point
        moves = {0: (1, 1), 1: (0, 1), 2: (-1, 1), 3: (-1, 0), 4: (-1, -1), 5: (0, -1), 6: (1, -1), 7: (1, 0)}
        for code in chain_code:
            dx, dy = moves[code]
            next_point = (current_point[0] + dx, current_point[1] + dy)
            cv2.line(img_med, current_point, next_point, (255, 255, 255), 1)
            current_point = next_point
        return img_med

    def medicine_measures(self, cropped_img, largest_contour):
        if largest_contour is None or len(largest_contour) == 0 or self.pixel_size_mm is None:
            return None, None
        measures = []
        measured_img = cropped_img.copy()
        x, y, w, h = cv2.boundingRect(largest_contour)
        width_mm = w * self.pixel_size_mm
        height_mm = h * self.pixel_size_mm
        measures.append((width_mm, height_mm))
        cv2.rectangle(measured_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(measured_img, f"{width_mm:.1f}mm x {height_mm:.1f}mm", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        return measures, measured_img

    def collect_data(self, img_med, mask, largest_contour, chain_code, measures, histograms):
        images_directory = "extracted_features/images"
        os.makedirs(images_directory, exist_ok=True)
        med_img_path = os.path.join(images_directory, f"{self.stag_id}_img_med.png")
        cv2.imwrite(med_img_path, img_med)
        mask_img_path = os.path.join(images_directory, f"{self.stag_id}_mask.png")
        mask_rgba = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGBA)
        mask_rgba[:, :, 3] = mask
        cv2.imwrite(mask_img_path, mask_rgba)
        contour_img_path = os.path.join(images_directory, f"{self.stag_id}_contour.png")
        contour_img = img_med.copy()
        cv2.drawContours(contour_img, [largest_contour], -1, (0, 255, 0), 3)
        cv2.imwrite(contour_img_path, contour_img)
        data_entry = {
            "Medicine Type": self.med_type,
            "Chain Code Length": len(chain_code),
            "Measurements": measures,
            **{f"{color}_histogram": [np.mean(hist), np.std(hist), np.min(hist), np.max(hist)] for color, hist in histograms.items()}
        }
        self.save_data_to_csv(data_entry)

    def save_data_to_csv(self, data, filename='extracted_features/medicines_features.csv'):
        df = pd.DataFrame([data])
        if os.path.exists(filename):
            df.to_csv(filename, mode='a', header=False, index=False)
        else:
            df.to_csv(filename, mode='w', index=False)
        print(f"Data appended to {filename}.")

    def process_images(self):
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
            filter_pdi = self.filter_pdi(subtracted_image)
            self.display_image(filter_pdi, 'Laplacian Gray Balance')
            remove_background = self.remove_background(filter_pdi)
            self.display_image(remove_background, 'Removed Background')
            erode_results = self.erode(remove_background)
            self.display_image(erode_results, 'Erosion for corrections')
            dilatation_results = self.dilatation(erode_results)
            self.display_image(dilatation_results, 'Dilatation to origin')
            remove_background_2 = self.remove_background(subtracted_image)
            histogram = self.calculate_histograms(remove_background_2)
            self.display_histogram(histogram, "Histogram by Subtracted and Removed")
            mask = self.create_mask(remove_background)
            self.display_image(mask, 'Mask')
            contoured_image, largest_contour = self.find_and_draw_contours(mask)
            if contoured_image is not None and largest_contour is not None and largest_contour.size > 0:
                self.display_image(contoured_image, 'Largest Contour by Mask')
                chain_code, _ = self.compute_chain_code(largest_contour)
                if chain_code is not None:
                    chain_drawn_image = self.draw_chain_code(remove_background, largest_contour, chain_code)
                    self.display_image(chain_drawn_image, 'Chain Code Drawn')
                measures, measured_medicine = self.medicine_measures(cropped_image, [largest_contour])
                if measured_medicine is not None:
                    self.display_image(measured_medicine, 'Measured Medicine')
                    data_entry = self.collect_data(measured_medicine, mask, largest_contour, chain_code, measures, histogram)
                    self.save_data_to_csv(data_entry)
        else:
            print("Markers not found.")

    def display_image(self, image, title):
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.show()

def main():
    stag_id = 8
    med_type = "Pill"
    image_path = ".\\thiago_fotos_MED\\img_8_008.jpg"
    neg_image_path = ".\\thiago_fotos_SUB\\img_8_006.jpg"
    processor = ExtractFeatures(image_path, neg_image_path, stag_id, med_type)
    processor.process_images()

if __name__ == "__main__":
    main()
