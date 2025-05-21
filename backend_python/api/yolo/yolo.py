import base64
import io
import cv2
from ultralytics import YOLO
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Agg')  # Use a non-interactive backend
import numpy as np
from pathlib import Path
import easyocr
from keras._tf_keras.keras.preprocessing import image
from keras._tf_keras.keras.preprocessing.image import array_to_img, img_to_array
from keras._tf_keras.keras.models import load_model
import keras_ocr

from backend_python.settings import BASE_DIR

pipeline = keras_ocr.pipeline.Pipeline()
class YOLODetector:
    def __init__(self, attributes_model_path=f"{BASE_DIR}/api/yolo/weights/best.pt", card_model_path=f"{BASE_DIR}/api/yolo/weights/segment_model.pt"):
        # Load the pretrained YOLO model
        self.attributes_model = YOLO(attributes_model_path)
        self.card_model = YOLO(card_model_path)

    def run_detection(self, image_path, confidence_threshold=0.42):
        # Load the image using OpenCV
        image = self.run_card_detection(image_path)

        # Convert the image to Gray then RGB format for YOLOv8, why grays? cause the model trained with grayscale id card image
        image_grayscaled = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)

        results = self.attributes_model.predict(image_grayscaled, conf=confidence_threshold, stream=False)[0]

        # Process results
        base64_image = None
        foto_image = None
        detected_objects = []
        for det in results.boxes:  # Loop through each detection
            label_index = int(det.cls)
            label = self.attributes_model.names[label_index]
            confidence = float(det.conf)
            bbox = det.xyxy.cpu().numpy()[0]
            x_min, y_min, x_max, y_max = bbox[:4]

            detected_objects.append({
                'label': label,
                'confidence': confidence,
                'x_min': x_min,
                'y_min': y_min,
                'x_max': x_max,
                'y_max': y_max,
            })

        # Filter out duplicate labels and keep the one with the highest confidence
        labels_of_interest = [
            'foto', 'nik', 'nama', 'ttl', 'jeniskelamin', 'alamat', 
            'rtrw', 'keldesa', 'kecamatan', 'agama', 
            'statuskawin', 'pekerjaan', 'kewarganegaraan'
        ]
        
        max_confidences = {label: None for label in labels_of_interest}

        for obj in detected_objects:
            label = obj['label']
            if label in max_confidences:
                if max_confidences[label] is None or obj['confidence'] > max_confidences[label]['confidence']:
                    max_confidences[label] = obj        

        extracted_data = {}
        list_of_images = []
        for label, obj in max_confidences.items():
            if label == 'foto':
                continue
            if obj:
                x_min = int(obj['x_min'])
                y_min = int(obj['y_min'])
                x_max = int(obj['x_max'])
                y_max = int(obj['y_max'])

                # Crop the image to the bounding box of the label
                cropped_image = image[y_min:y_max, x_min:x_max]
                cropped_image_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(cropped_image_gray, (5, 5), 0)
                # T, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV)
                binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                # final = cv2.bitwise_and(cropped_image, cropped_image, mask=thresh)
                # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                # enhanced = clahe.apply(cropped_image)
                # blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
                # image = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
                # letterboxed = letterbox_image(image, (640, 640))
                # normalized = letterboxed.astype('float32') / 255.0
                # denoised = cv2.GaussianBlur(image, (3, 3), 0)
                # thresh = cv2.threshold(cropped_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                # dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
                # dist = cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
                # dist = (dist * 255).astype("uint8")
                # dist = cv2.GaussianBlur(dist, (5, 5), 0)
                # dist = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                _, buffer = cv2.imencode('.png', binary)
                cropped_image = base64.b64encode(buffer).decode('utf-8')
                list_of_images.append(cropped_image)
                io_buf = io.BytesIO(buffer)

                # Perform OCR using keras_ocr
                try:
                    # prediction = easyocr.Reader(['id', 'en'], gpu=False)
                    # prediction_groups = prediction.readtext(binary, detail=0, paragraph=True)
                    # if prediction_groups and len(prediction_groups) > 0:
                    #     extracted_data[label] = " ".join(prediction_groups[0])
                    prediction_groups = pipeline.recognize([keras_ocr.tools.read(io_buf)])
                    if prediction_groups and len(prediction_groups[0]) > 0:
                        extracted_data[label] = " ".join([text for text, _ in prediction_groups[0]])
                except Exception as e:
                    print(f"Error processing label {label}: {e}")

        print('Extracted Data:', extracted_data)

        if max_confidences['foto']:
            x_min = int(max_confidences['foto']['x_min'])
            y_min = int(max_confidences['foto']['y_min'])
            x_max = int(max_confidences['foto']['x_max'])
            y_max = int(max_confidences['foto']['y_max'])

            # Crop the image to the bounding box of 'foto'
            # foto_image = image[y_min:y_max, x_min:x_max]
            foto_image = image[y_min:y_max, x_min:x_max]
            # foto_image = cv2.cvtColor(foto_image, cv2.COLOR_BGR2GRAY)
            # blurred = cv2.GaussianBlur(foto_image, (5, 5), 0)
            # T, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)
            # final = cv2.bitwise_and(foto_image, foto_image, mask=thresh)
            # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            # enhanced = clahe.apply(foto_image)
            # blurred = cv2.GaussianBlur(foto_image, (3, 3), 0)
            # binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            # image = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
            # letterboxed = letterbox_image(image, (640, 640))
            # normalized = letterboxed.astype('float32') / 255.0
            # denoised = cv2.GaussianBlur(image, (3, 3), 0)
            # blur = cv2.GaussianBlur(foto_image, (5, 5), 0)
            # thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            # dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
            # dist = cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
            # dist = (dist * 255).astype("uint8")
            # dist = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]


            # Convert cropped image to base64
            _, buffer = cv2.imencode('.png', foto_image)
            foto_image = base64.b64encode(buffer).decode('utf-8')
        else:
            foto_image = None
            # If no 'foto' label is detected, set foto_image to None

        # Plot and convert the image to base64
        # For testing purposes
        plt.imshow(results.plot())
        plt.axis('off')
        plt.tight_layout()
        pic_IObytes = io.BytesIO()
        plt.savefig(pic_IObytes, format='png')
        pic_IObytes.seek(0)
        base64_image = base64.b64encode(pic_IObytes.read()).decode('utf-8')
        plt.close()
        # Close the plot to free memory

        return [extracted_data, foto_image, base64_image, list_of_images]

    def run_card_detection(self, image_path):
        """
        Run card detection on the provided image.
        :param image_path: The path to the image file.
        :return: A list of detected objects with their labels, confidence scores, and bounding boxes.
        """
        # Perform detection
        # Load the image using OpenCV
        image = cv2.imread(image_path)

        results = self.card_model.predict(image, conf=0.42, stream=False)[0]
        # print('result', results)
        
        img = np.copy(results.orig_img)
        
        aligned_crop = None  # Initialize aligned_crop to None
        for r in results:
            img = np.copy(r.orig_img)
            
            # Display original image
            # plt.figure(figsize=(10, 10))
            # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # plt.title("Original Image")
            # plt.axis('off')
            # plt.show()
            
            # Iterate each detected object
            for ci, c in enumerate(r):
                label = c.names[c.boxes.cls.tolist().pop()]
                contour = c.masks.xy.pop().astype(np.int32)  # Get object contour
                
                # --- Step 1: Get the best-fit rotated rectangle ---
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.intp(box)  # Corners of the rotated rectangle
                
                # --- Step 2: Compute rotation angle ---
                angle = rect[-1] - 90
                if angle < -45:  # Adjust angle to make it between -45 and 45
                    angle = 90 + angle
                
                # --- Step 3: Rotate the image to align the object ---
                (h, w) = img.shape[:2]
                center = rect[0]  # Center of the rotated rectangle
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)
                
                # --- Step 4: Crop the rotated object ---
                b_mask = np.zeros(img.shape[:2], np.uint8)
                cv2.drawContours(b_mask, [contour], -1, 255, -1)
                rotated_mask = cv2.warpAffine(b_mask, M, (w, h))
                
                # Find new bounding box after rotation
                rotated_contours, _ = cv2.findContours(rotated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if rotated_contours:
                    x, y, w, h = cv2.boundingRect(rotated_contours[0])
                    aligned_crop = rotated[y:y+h, x:x+w]
                    
                    # Display aligned object
                    # plt.imshow(cv2.cvtColor(aligned_crop, cv2.COLOR_BGR2RGB))
                    # plt.axis('off')
                    # plt.tight_layout()
                    # pic_IObytes = io.BytesIO()
                    # plt.savefig(pic_IObytes, format='png')
                    # pic_IObytes.seek(0)
                    # aligned_crop = base64.b64encode(pic_IObytes.read()).decode('utf-8')
                    # plt.close()


                    # aligned_crop = cv2.resize(aligned_crop, (640, 640))  # Resize to a fixed size for consistency
        return aligned_crop

    def crop_image_and_ocr(self, image, x_min, y_min, x_max, y_max):
        """
        Crop the image to the specified bounding box.
        :param image: The input image.
        :param x_min: Minimum x-coordinate of the bounding box.
        :param y_min: Minimum y-coordinate of the bounding box.
        :param x_max: Maximum x-coordinate of the bounding box.
        :param y_max: Maximum y-coordinate of the bounding box.
        :return: Cropped image.
        """
        return image[y_min:y_max, x_min:x_max]
        