import cv2
import numpy as np
from datetime import datetime
import os
import sys
from log_file import phase_1
import logging

# Configure logger
logger = phase_1("")
logging.basicConfig(level=logging.INFO)


class ImageProcessor:
    def __init__(self, image_path=None, logger=None):
        try:
            self.logger = logger or logging.getLogger()
            self.image = None
            if image_path:
                self.logger.info(f"Loading image: {image_path}")
                self.image = cv2.imread(image_path)
                if self.image is None:
                    self.logger.error(f"Failed to load image: {image_path}")
                    raise ValueError("Image not found")
                self.logger.info(f"Image loaded: {self.image.shape[1]}x{self.image.shape[0]}")
                self.original = self.image.copy()
            else:
                self.logger.warning("No image path provided")
        except Exception as e:
            self.logger.exception(f"Initialization failed: {str(e)}")
            raise

    def _ensure_image(self):
        if self.image is None:
            raise ValueError("No image loaded")

    def save_image(self, filename, results_dir):
        try:
            self._ensure_image()
            os.makedirs(results_dir, exist_ok=True)
            path = os.path.join(results_dir, filename)
            success = cv2.imwrite(path, self.image)
            if success:
                self.logger.info(f"Image saved: {path}")
                return path
            self.logger.error(f"Failed to save: {path}")
            return None
        except Exception as e:
            self.logger.exception(f"Save failed: {str(e)}")
            return None

    def reset_image(self):
        if self.original is not None:
            self.image = self.original.copy()
            self.logger.info("Image reset to original")
        return self

    def draw_line(self, start, end, color=(0, 255, 0), thickness=2):
        try:
            self._ensure_image()
            cv2.line(self.image, start, end, color, thickness)
        except Exception as e:
            self.logger.error(f"Draw line failed: {str(e)}")
        return self

    def draw_rectangle(self, top_left, bottom_right, color=(0, 0, 255), thickness=2):
        try:
            self._ensure_image()
            cv2.rectangle(self.image, top_left, bottom_right, color, thickness)
        except Exception as e:
            self.logger.error(f"Draw rectangle failed: {str(e)}")
        return self

    def draw_arrowed_line(self, start, end, color=(255, 0, 0), thickness=2):
        try:
            self._ensure_image()
            cv2.arrowedLine(self.image, start, end, color, thickness)
        except Exception as e:
            self.logger.error(f"Draw arrowed line failed: {str(e)}")
        return self

    def draw_circle(self, center, radius, color=(0, 255, 255), thickness=2):
        try:
            self._ensure_image()
            cv2.circle(self.image, center, radius, color, thickness)
        except Exception as e:
            self.logger.error(f"Draw circle failed: {str(e)}")
        return self

    def add_text(self, text, position, font_scale=1, color=(255, 255, 255), thickness=2):
        try:
            self._ensure_image()
            cv2.putText(
                self.image, text, position,
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                color, thickness, cv2.LINE_AA
            )
        except Exception as e:
            self.logger.error(f"Add text failed: {str(e)}")
        return self

    def add_datetime(self, position=(10, 30), color=(200, 200, 200)):
        try:
            self._ensure_image()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return self.add_text(timestamp, position, 0.8, color, 1)
        except Exception as e:
            self.logger.error(f"Add datetime failed: {str(e)}")
        return self

    def detect_faces(self, classifier_path):
        try:
            self._ensure_image()
            face_cascade = cv2.CascadeClassifier(classifier_path)
            if face_cascade.empty():
                raise ValueError("Failed to load Haar cascade")

            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            self.logger.info(f"Detected {len(faces)} faces")
        except Exception as e:
            self.logger.exception(f"Face detection failed: {str(e)}")
        return self

    def detect_edges(self, low=50, high=150, blur=True):
        try:
            self._ensure_image()
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            if blur:
                gray = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(gray, low, high)
            self.image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            self.logger.exception(f"Edge detection failed: {str(e)}")
        return self


class VideoFaceDetector:
    def __init__(self, video_path, classifier_path, logger, results_dir, output_name="face_detection.avi"):
        self.video_path = video_path
        self.classifier_path = classifier_path
        self.logger = logger
        self.results_dir = results_dir
        self.output_path = os.path.join(results_dir, output_name)

    def process_video(self):
        try:
            face_cascade = cv2.CascadeClassifier(self.classifier_path)
            if face_cascade.empty():
                raise ValueError("Failed to load Haar cascade")

            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise IOError(f"Failed to open video: {self.video_path}")

            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            os.makedirs(self.results_dir, exist_ok=True)
            out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Detect faces
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                out.write(frame)
                frame_count += 1

            cap.release()
            out.release()
            self.logger.info(f"Processed {frame_count} frames. Output: {self.output_path}")
            return True
        except Exception as e:
            self.logger.exception(f"Video processing failed: {str(e)}")
            return False


if __name__ == "__main__":
    # ========== CONFIGURE PATHS HERE ==========
    PROJECT_BASE = r"C:\Users\geeth\PycharmProjects\Opencv_"
    UPLOADS_DIR = os.path.join(PROJECT_BASE, "static", "uploads")
    RESULTS_DIR = os.path.join(PROJECT_BASE, "static", "results")

    IMAGE1_PATH = os.path.join(UPLOADS_DIR, "pexels-silvi-7363572.jpg")
    IMAGE2_PATH = os.path.join(UPLOADS_DIR, "mandril_color.tif")
    VIDEO_PATH = os.path.join(UPLOADS_DIR, "test_video.mp4")
    HAAR_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")

    # Create directories if they don't exist
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    logger.info("===== STARTING PROCESSING WORKFLOW =====")
    logger.info(f"Project base: {PROJECT_BASE}")
    logger.info(f"Image 1 path: {IMAGE1_PATH}")
    logger.info(f"Image 2 path: {IMAGE2_PATH}")
    logger.info(f"Video path: {VIDEO_PATH}")
    logger.info(f"Haar cascade path: {HAAR_PATH}")
    logger.info(f"Results directory: {RESULTS_DIR}")

    try:
        # Initialize image processor
        processor = ImageProcessor(IMAGE1_PATH, logger)

        # Save original image
        processor.save_image("0_original.jpg", RESULTS_DIR)

        # Line drawing
        processor.reset_image() \
            .draw_line((50, 50), (300, 300)) \
            .save_image("1_line.jpg", RESULTS_DIR)

        # Rectangle drawing
        processor.reset_image() \
            .draw_rectangle((100, 100), (250, 250)) \
            .save_image("2_rectangle.jpg", RESULTS_DIR)

        # Arrowed line drawing
        processor.reset_image() \
            .draw_arrowed_line((300, 100), (450, 250)) \
            .save_image("3_arrow.jpg", RESULTS_DIR)

        # Circle drawing
        processor.reset_image() \
            .draw_circle((400, 150), 60) \
            .save_image("4_circle.jpg", RESULTS_DIR)

        # Text addition
        processor.reset_image() \
            .add_text("Hello OpenCV!", (50, 50), 1.5, (0, 255, 0), 3) \
            .save_image("5_text.jpg", RESULTS_DIR)

        # Timestamp addition
        processor.reset_image() \
            .add_datetime() \
            .save_image("6_timestamp.jpg", RESULTS_DIR)

        # Face detection
        processor.reset_image() \
            .detect_faces(HAAR_PATH) \
            .save_image("7_faces.jpg", RESULTS_DIR)

        # Edge detection
        processor.reset_image() \
            .detect_edges() \
            .save_image("8_edges.jpg", RESULTS_DIR)

        # Blending example (using second image) - FIXED SIZE MISMATCH
        processor.reset_image()
        img2 = cv2.imread(IMAGE2_PATH)
        if img2 is not None:
            # Resize the second image to match the first image's dimensions
            img2_resized = cv2.resize(img2, (processor.image.shape[1], processor.image.shape[0]))
            processor.image = cv2.addWeighted(processor.image, 0.7, img2_resized, 0.3, 0)
            processor.save_image("9_blended.jpg", RESULTS_DIR)
        else:
            logger.error(f"Could not load second image for blending: {IMAGE2_PATH}")

        logger.info("All image operations completed successfully")

        # Process video if available
        if os.path.exists(VIDEO_PATH):
            logger.info(f"Processing video: {VIDEO_PATH}")
            video_processor = VideoFaceDetector(
                video_path=VIDEO_PATH,
                classifier_path=HAAR_PATH,
                logger=logger,
                results_dir=RESULTS_DIR,
                output_name="10_video_faces.avi"
            )
            if video_processor.process_video():
                logger.info("Video processing completed successfully")
            else:
                logger.error("Video processing failed")
        else:
            logger.error(f"Video not found: {VIDEO_PATH}")

    except Exception as e:
        logger.exception(f"Fatal error: {str(e)}")

    logger.info("===== PROCESSING COMPLETED =====")