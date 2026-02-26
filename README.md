# **Image_Video_Preprocessing_WebApplication_using_OpenCV**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![HTML5](https://img.shields.io/badge/HTML-HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS-CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-ES6-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)

A full-stack Image and Video Processing Web Application built using Flask and OpenCV.
The system enables real-time image manipulation, computer vision operations, and video processing through an interactive browser-based interface.

<img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/8f7f2d8c-9582-4e92-827d-7a7a5780bbfa" />

---

## **Project Overview**

This project implements a complete computer vision processing pipeline that:

- Performs image preprocessing and transformations
- Supports real-time face detection (image & video)
- Provides interactive drawing and annotation tools
- Enables ROI extraction and edge detection
- Processes video frames with OpenCV
- Follows secure file handling and modular backend design

The application demonstrates production-ready CV workflow from upload → processing → preview → download.

---

## **Objective**

To design and deploy a scalable web-based computer vision system that:

- Accepts image and video uploads
- Performs multiple OpenCV operations dynamically
- Allows real-time parameter customization
- Displays processed results instantly
- Maintains clean backend architecture with logging

---

## **Core Features**

### 1️⃣ Image Blending & Composition

- Equal blending using `cv2.addWeighted()`
- Adjustable weighted blending
- Automatic resizing before blending
- 
<img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/84461548-4c94-43ef-bf1d-d01f8d7c27ba" />

### 2️⃣ Image Reshaping

- Custom width & height resizing using `cv2.resize()`
- Aspect ratio handling
 
<img width="400" height="150" alt="image" src="https://github.com/user-attachments/assets/ff8d6422-a7c0-49fb-91df-d7492a255848" />

### 3️⃣ Region of Interest (ROI)

- Coordinate-based ROI extraction
- Interactive canvas-based ROI selection
- Pixel-level slicing operations

<img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/4cb3ae21-d92d-40de-8dd1-a89ae5f4eec5" />
<img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/db256ee4-0f9a-4069-8aff-04f8e644ed71" />

### 4️⃣ Geometric Drawing Operations

- Line → `cv2.line()`
- Rectangle → `cv2.rectangle()`
- Circle → `cv2.circle()`
- Arrow → `cv2.arrowedLine()`
- Customizable color, thickness, and coordinates
  
<img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/58ac237e-25cd-4ee4-ab8e-5f9c6aa9b681" />
<img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/435daf0c-7915-4c73-8d76-3db14968794a" />
<img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/1751daef-a0a6-4a76-ac19-4e360d99129e" />
<img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/bf1bddeb-550a-4567-a514-240d6cfcdcda" />


### 5️⃣ Text & Date-Time Overlay

- Text addition using `cv2.putText()`
- Dynamic timestamp insertion
- Custom font size, color, and positioning

<img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/6cd75924-98a9-4ff3-bdae-53645845f55e" />
<img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/7b251ee8-e556-49a1-897f-b395f564d4a7" />

### 6️⃣ Face Detection (Image & Video)

- Haar Cascade Classifier
- `detectMultiScale()` implementation
- Real-time video face detection
- Bounding box customization

<img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/9ab6b927-4fea-4e66-ba18-fa910b94deef" />

### 7️⃣ Edge Detection

- Canny Edge Detection → `cv2.Canny()`
- Sobel Operator
- Laplacian Operator
- Optional Gaussian Blur preprocessing

<img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/5f4299b6-0b93-4b53-85af-d60aa87b5067" />

### 8️⃣ Video Processing Engine

- Frame-by-frame processing
- VideoWriter integration
- FPS maintenance
- Automatic resource cleanup

<img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/43649cab-7122-48e3-857c-76dae6c33175" />

<img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/c7b9d0e6-d197-47ce-a352-c6f53677da23" />

---

## **Technical Stack**

### Backend

- Python
- Flask
- OpenCV (cv2)
- NumPy
- Werkzeug

### Frontend

- HTML5
- CSS3 (Responsive Design)
- JavaScript (ES6+)
- Canvas API
- OpenCV.js

---

## **Architecture Highlights**

- REST-based endpoint design
- Modular image processor class structure
- Structured logging system
- Secure file upload handling
- 200MB upload limit protection
- Exception handling for robustness

---

## **Project Structure**
```
Image_Video_Preprocessing_WebApplication_using_OpenCV/
│
├── app.py
├── image_processor.py
├── log_file.py
├── README.md
│
├── Face_detection/
│
├── static/
│
└── templates/
    └── index.html
```

---

## **Processing Workflow**

Upload → Select Technique → Adjust Parameters → Process → Preview → Download

Supports single-step and sequential processing operations.

---

## **Performance & Optimization**

- Efficient NumPy array handling
- Proper memory cleanup
- Frame-level video processing
- Structured logging for debugging
- Scalable architecture for multi-threading

---

## **Security Features**

- Secure filename handling
- File type validation
- Upload size restriction
- Controlled directory access
- Error handling with user feedback

---

## **Run Locally**
```
pip install -r requirements.txt
python app.py
```

Open in browser:
```
http://localhost:5000
```

---

## **Future Improvements**

- Cloud deployment (AWS / Render / Heroku)
- Deep learning–based face detection
- Batch image processing
- GPU acceleration support

---

## **Author**

**Hema Malini Gangumalla**
Aspiring Data Scientist

📧 [hemamalinig07@gmail.com](mailto:hemamalinig07@gmail.com)

---

## **License**

MIT License
