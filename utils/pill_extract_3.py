import os
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import stag
from rembg import remove

class ExtractFeatures:
    def __init__(self, image_path, neg_image_path, stag_id, med_type):
        """Initializes the class with paths to the images, an identifier for the stag marker, and the type of medicine."""
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
        import numpy as np
        dummy_image = np.full((100, 100, 3), 128, dtype=np.uint8)
        try:
            is_success, buffer = cv2.imencode(".png", dummy_image)
            if is_success:
                remove(buffer.tobytes())
            print("rembg model preloaded successfully.")
        except Exception as e:
            print(f"Failed to preload rembg model: {e}")


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
            self.calculate_pixel_size_mm()
            return self.corners
        print(f"Marker with ID {self.stag_id} not found in one of the images.")
        return None
    
    def calculate_pixel_size_mm(self):
        if self.corners is not None:
            width_px = np.max(self.corners[:, 0]) - np.min(self.corners[:, 0])
            self.pixel_size_mm = 20.0 / width_px
        else:
            self.pixel_size_mm = None  # Ensure it is set to None if corners are not detected
            print("Failed to detect corners to calculate pixel size.")



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

        if self.pixel_size_mm is None:
            print("Pixel size not set, cannot calculate scan area.")
            return None
        if image is None:
            print("Homogenized image is not available.")
            return None
        if self.corners is None:
            print("Corners are not detected.")
            return None
        corners_int = self.corners.astype(int)
        centroid_x = int(np.mean(corners_int[:, 0]))
        centroid_y = int(np.mean(corners_int[:, 1]))
        cv2.putText(image, f'ID:{self.stag_id}', (centroid_x + 45, centroid_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        marker_width_px = np.max(corners_int[:, 0]) - np.min(corners_int[:, 0])
        pixel_size_mm = marker_width_px / 20.0  
        crop_width = int(30 * pixel_size_mm)
        crop_height = int(60 * pixel_size_mm) if self.med_type == "Pill" else int(75 * pixel_size_mm)
        crop_y_adjustment = int(10 * pixel_size_mm)

        # Calcula os limites de recorte
        x_min = max(centroid_x - crop_width, 0)
        x_max = min(centroid_x + crop_width, image.shape[1])
        y_min = max(centroid_y - crop_height - crop_y_adjustment, 0)
        y_max = min(centroid_y - crop_y_adjustment, image.shape[0])

        # Desenha o retângulo na imagem
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

        # Armazena as coordenadas de recorte
        self.scan_areas[self.stag_id] = (x_min, x_max, y_min, y_max)

        # Exibe a imagem
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Image with Scan Area')
        plt.show()

        # Retorna as coordenadas de recorte
        return (x_min, x_max, y_min, y_max)


    def crop_scan_area(self, image, crop_coords):
        """Crops the defined scan area from the homogenized image for further processing."""
        x_min, x_max, y_min, y_max = crop_coords
        cropped_image = image[y_min:y_max, x_min:x_max]
        return cropped_image
    
    def filter_pdi(self, image):
        """Applies Gaussian blur followed by a Sobel filter to enhance horizontal edges of a reflective object."""
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
        filter = np.array([
            [0,  1, 0],
            [1, -4.5, 1],
            [0,  1, 0]
        ])
        ddepth = cv2.CV_16S
        img_filtered = cv2.filter2D(img_blur, ddepth, filter)
        abs_img_filtered = cv2.convertScaleAbs(img_filtered)
        return abs_img_filtered
    
    def remove_background(self, image_np_array):
        """Removes the background from the image using the rembg library."""
        is_success, buffer = cv2.imencode(".png", image_np_array)
        if not is_success:
            raise ValueError("Failed to encode image for background removal.")
        output_image = remove(buffer.tobytes())
        img_med = cv2.imdecode(np.frombuffer(output_image, np.uint8), cv2.IMREAD_UNCHANGED)
        if img_med is None:
            raise ValueError("Failed to decode processed image.")
        return img_med
    

    def erode(self, image):
        filter_erode = np.array([[0,1,0],
                                 [1,1,1],
                                 [0,1,0]],dtype=np.uint8)
        # filter_erode = np.ones((5,3), dtype=np.uint8)
        erode_dst = cv2.erode(image, filter_erode,iterations=3)
        return erode_dst
    
    def dilatation(self, image):
        filter_dilate =  np.array([ [0,1,0],
                                    [1,1,1],
                                    [0,1,0]],dtype=np.uint8)
        dilatation_dst = cv2.dilate(image, filter_dilate, iterations=3)
        return dilatation_dst

    def calculate_histograms(self, img_med):
        """Calculates and returns RGB and Gray histograms for the processed image."""
        histograms = {}
        if img_med is not None and img_med.ndim == 3:
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img_med, cv2.COLOR_BGR2RGB)
            
            # Calcular histogramas para cada canal de cor RGB
            colors = ('r', 'g', 'b')
            for i, color in enumerate(colors):
                hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
                histograms[color] = hist.flatten()
            
            # Converter para escala de cinza e calcular o histograma
            img_gray = cv2.cvtColor(img_med, cv2.COLOR_RGB2GRAY)
            hist_gray = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
            histograms['gray'] = hist_gray.flatten()

        return histograms
    
    def display_histogram(self, histograms, title):
        """Displays RGB and Gray histograms on a single plot."""
        if histograms:
            plt.figure(figsize=(10, 5))
            plt.title(title)
            colors = {'r': 'red', 'g': 'green', 'b': 'blue', 'gray': 'gray'}  
            
            # Plotar cada histograma
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
        """Creates a binary mask for the foreground object in the image and saves it with transparency."""
        if img.shape[2] == 4:
            img = img[:, :, :3]  # Remove alpha channel
        lower_bound = np.array([10, 10, 10])
        upper_bound = np.array([256, 256, 256])
        mask = cv2.inRange(img, lower_bound, upper_bound)
        # Convert binary mask to 4-channel 
        mask_rgba = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGBA)
        mask_rgba[:, :, 3] = mask 
        # # Save the mask as a .pkl file
        # directory = 'features/mask'
        # if not os.path.exists(directory):
        #     os.makedirs(directory)
        # file_number = 0
        # while os.path.exists(f'{directory}/mask_{file_number}.pkl'):
        #     file_number += 1
        # file_path = f'{directory}/mask_{file_number}.pkl'
        # with open(file_path, 'wb') as file:
        #     pickle.dump(mask, file)
        # print(f'Mask saved as {file_path} with transparency in {directory}')
        return mask
    
    def find_and_draw_contours(self, mask):
        """Finds and draws the largest contour around the foreground object."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if largest_contour.size > 0:
                mask_with_contours = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA)
                mask_with_contours[:, :, 3] = mask
                cv2.drawContours(mask_with_contours, [largest_contour], -1, (0, 0, 255, 255), 2)
                return mask_with_contours, largest_contour
        else:
            return None
        
    def compute_chain_code(self, contour):
        """Calculates the chain code for the contour which can be used for shape analysis."""
        start_point = contour[0][0]
        current_point = start_point
        chain_code = []
        moves = {
            (-1, 0) : 3,
            (-1, 1) : 2,
            (0, 1)  : 1,
            (1, 1)  : 0,
            (1, 0)  : 7,
            (1, -1) : 6,
            (0, -1) : 5,
            (-1, -1): 4
        }
        for i in range(1, len(contour)):
            next_point = contour[i][0]
            dx = next_point[0] - current_point[0]
            dy = next_point[1] - current_point[1]
            if dx != 0:
                dx = dx // abs(dx)
            if dy != 0:
                dy = dy // abs(dy)
            move = (dx, dy)
            if move in moves:
                chain_code.append(moves[move])
            current_point = next_point
        # Close the loop
        dx = start_point[0] - current_point[0]
        dy = start_point[1] - current_point[1]
        if dx != 0:
            dx = dx // abs(dx)
        if dy != 0:
            dy = dy // abs(dy)
        move = (dx, dy)
        if move in moves:
            chain_code.append(moves[move])
        return chain_code, len(chain_code)

    def draw_chain_code(self, img_med, contour, chain_code):
        """Draws the chain code on the image to visually represent contour direction changes."""
        start_point = tuple(contour[0][0])
        current_point = start_point
        moves = {
            0: (1, 1),    # bottom-right
            1: (0, 1),    # right
            2: (-1, 1),   # top-right
            3: (-1, 0),   # top
            4: (-1, -1),  # top-left
            5: (0, -1),   # left
            6: (1, -1),   # bottom-left
            7: (1, 0)     # bottom
        }
        for code in chain_code:
            dx, dy = moves[code]
            next_point = (current_point[0] + dx, current_point[1] + dy)
            cv2.line(img_med, current_point, next_point, (255, 255, 255), 1)
            current_point = next_point
        return img_med

    def medicine_measures(self, cropped_img, largest_contour):
        if largest_contour is None or len(largest_contour) == 0:
            print("No contours found.")
            return None
        if self.pixel_size_mm is None:
            print("Pixel size not set.")
            return None  
        
        measures = []
        measured_img = cropped_img.copy()
        for point in largest_contour:
            x, y, w, h = cv2.boundingRect(point)  
            width_mm = w * self.pixel_size_mm  
            height_mm = h * self.pixel_size_mm 
            measures.append((width_mm, height_mm))
            cv2.rectangle(measured_img, (x, y), (x + w, y + h), (0, 255, 0), 1) 
            cv2.putText(measured_img, f"{width_mm:.1f}mm x {height_mm:.1f}mm", 
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1) 
            print(f"Bounding box at ({x}, {y}), width: {width_mm:.1f}mm, height: {height_mm:.1f}mm")

        return measures, measured_img  

    def calculate_media_and_desv_histograms(self, img_med):
        # Calcula histogramas RGB e cinza
        histograms = {'r': [], 'g': [], 'b': [], 'gray': []}
        if img_med is not None and img_med.ndim == 3:
            img_rgb = cv2.cvtColor(img_med, cv2.COLOR_BGR2RGB)
            for i, color in enumerate(['r', 'g', 'b']):
                hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
                histograms[color] = hist.flatten().tolist()
            img_gray = cv2.cvtColor(img_med, cv2.COLOR_RGB2GRAY)
            hist_gray = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
            histograms['gray'] = hist_gray.flatten().tolist()
        return histograms

    def collect_data(self, img_med, mask, largest_contour, chain_code, measures, histograms):
        if largest_contour is not None:
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = w / h
            area = cv2.contourArea(largest_contour)
            rect_area = w * h
            extent = area / rect_area
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area
        else:
            aspect_ratio, area, extent, solidity = 0, 0, 0, 0
        
        data_entry = {
            "Medicine Type": self.med_type,
            "Contour Area": area,
            "Contour Perimeter": cv2.arcLength(largest_contour, True) if largest_contour is not None else 0,
            "Chain Code Length": len(chain_code),
            "Chain Code": chain_code,
            "RGB Histograms": histograms['r'] + histograms['g'] + histograms['b'],
            "Gray Histogram": histograms['gray'],
            "Measures": measures,
            "Aspect Ratio": aspect_ratio,
            "Extent": extent,
            "Solidity": solidity
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
           
            histogram = self.calculate_histograms(subtracted_image)
            self.display_histogram(histogram, "Histogram by Subtracted Image")

            filter_pdi = self.filter_pdi(subtracted_image)
            self.display_image(filter_pdi, 'Laplacian Gray Balance')

            remove_background = self.remove_background(filter_pdi)
            self.display_image(remove_background, 'Removed Background')       
            
            # erode_results = self.erode(remove_background)
            # self.display_image(erode_results, 'Erosion for corrections')
            
            # dilatation_results = self.dilatation(erode_results)
            # self.display_image(dilatation_results, 'Dilatation to origin')
            
            mask = self.create_mask(remove_background)
            self.display_image(mask, 'Mask ')
            
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
    stag_id = 8, 2
    med_type = "Pill"
    image_path = ".\\thiago_fotos_MED\\img_8_008.jpg"
    neg_image_path = "thiago_fotos_SUB\img_8_008.jpg"
    processor = ExtractFeatures(image_path, neg_image_path, stag_id, med_type)
    processor.process_images()

if __name__ == "__main__":
    main()
