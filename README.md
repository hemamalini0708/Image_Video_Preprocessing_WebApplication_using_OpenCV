# Image_Video_Preprocessing_WebApplication_using_OpenCV

<img width="1200" height="514" alt="image" src="https://github.com/user-attachments/assets/8f7f2d8c-9582-4e92-827d-7a7a5780bbfa" />


# Project Overview
  A comprehensive web-based image and video processing application built with Flask, OpenCV, and modern web technologies. The tool provides browser-based editing and analysis capabilities for both images and videos with a user-friendly interface.

# Technical Stack
## Backend Technologies
  Python Flask - Web framework for API endpoints
  
  OpenCV (cv2) - Core image/video processing library
  
  NumPy - Numerical operations for image data manipulation
  
  Werkzeug - File upload handling and security
  
  Tempfile - Temporary file management for video processing

# Frontend Technologies
  HTML5 - Semantic structure with modern elements
  
  CSS3 - Advanced styling with gradients, animations, and responsive design
  
  JavaScript (ES6+) - Client-side interactivity and OpenCV.js integration
  
  Canvas API - Client-side image manipulation and drawing
  
  Font Awesome - Icon library for UI elements

# Core Image Processing Techniques Implemented
## 1. Image Blending & Composition
<img width="645" height="743" alt="image" src="https://github.com/user-attachments/assets/84461548-4c94-43ef-bf1d-d01f8d7c27ba" />


  Equal Weight Blending: cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
  
  Weighted Blending: Adjustable alpha blending with user-controlled weights
  
  Automatic Resizing: Images are resized to match dimensions before blending

## 2. Geometric Transformations
<img width="968" height="741" alt="image" src="https://github.com/user-attachments/assets/ff8d6422-a7c0-49fb-91df-d7492a255848" />


  Image Resizing: cv2.resize() with user-defined width and height
  
  Aspect Ratio Handling: Maintains proportions during reshaping operations

## 3. Region of Interest (ROI) Operations
<img width="776" height="663" alt="image" src="https://github.com/user-attachments/assets/4cb3ae21-d92d-40de-8dd1-a89ae5f4eec5" />
<img width="697" height="600" alt="image" src="https://github.com/user-attachments/assets/db256ee4-0f9a-4069-8aff-04f8e644ed71" />

  Rectangular ROI Extraction: img_cv[y:y+height, x:x+width]
  
  Interactive Selection: Canvas-based ROI selection with mouse interactions
  
  Coordinate-based Extraction: Precise pixel-level region extraction

## 4. Computer Vision Drawing Operations
<img width="926" height="618" alt="image" src="https://github.com/user-attachments/assets/58ac237e-25cd-4ee4-ab8e-5f9c6aa9b681" />
<img width="830" height="587" alt="image" src="https://github.com/user-attachments/assets/435daf0c-7915-4c73-8d76-3db14968794a" />
<img width="829" height="618" alt="image" src="https://github.com/user-attachments/assets/1751daef-a0a6-4a76-ac19-4e360d99129e" />
<img width="835" height="648" alt="image" src="https://github.com/user-attachments/assets/bf1bddeb-550a-4567-a514-240d6cfcdcda" />
  Primitive Shapes:
  
  Lines: cv2.line()
  
  Rectangles: cv2.rectangle()
  
  Circles: cv2.circle()
  
  Arrows: cv2.arrowedLine()
  
  Customizable Properties: Color, thickness, position parameters

## 5. Text & Annotation Features
<img width="670" height="733" alt="image" src="https://github.com/user-attachments/assets/6cd75924-98a9-4ff3-bdae-53645845f55e" />
<img width="961" height="680" alt="image" src="https://github.com/user-attachments/assets/7b251ee8-e556-49a1-897f-b395f564d4a7" />

  Text Overlay: cv2.putText() with customizable fonts and positioning
  
  DateTime Stamping: Automatic timestamp addition with multiple format options
  
  Font Control: Size, color, thickness, and position customization

## 6. Face Detection System
<img width="901" height="530" alt="image" src="https://github.com/user-attachments/assets/9ab6b927-4fea-4e66-ba18-fa910b94deef" />

  Haar Cascade Classifiers: cv2.CascadeClassifier()
  
  Multi-scale Detection: detectMultiScale() with scale factor and minimum neighbors
  
  Real-time Video Processing: Frame-by-frame face detection in videos
  
  Bounding Box Visualization: Customizable rectangle color and thickness

## 7. Feature Detection Algorithms
  Edge Detection:
<img width="578" height="719" alt="image" src="https://github.com/user-attachments/assets/5f4299b6-0b93-4b53-85af-d60aa87b5067" />

  Canny Edge Detector: cv2.Canny()
  
  Sobel Operator: cv2.Sobel() for gradient-based edge detection
  
  Laplacian Operator: cv2.Laplacian() for second-derivative edge detection
  
  Corner Detection:
  
  Harris Corner Detection: cv2.cornerHarris()
  
  Adaptive thresholding for corner identification

## 8. Video Processing Capabilities
<img width="1086" height="584" alt="image" src="https://github.com/user-attachments/assets/43649cab-7122-48e3-857c-76dae6c33175" />

<img width="1031" height="453" alt="image" src="https://github.com/user-attachments/assets/c7b9d0e6-d197-47ce-a352-c6f53677da23" />

  Frame Extraction: Capture and process individual video frames
  
  Temporal Analysis: Frame counting and timestamp overlay
  
  Video Compression: MP4 encoding with controlled quality settings
  
  Real-time Processing: Continuous frame processing with OpenCV VideoWriter

# Advanced Technical Features
## Image Processing Pipeline
  Input Validation: Secure file upload with type and size restrictions
  
  Memory Management: Efficient NumPy array handling for large images
  
  Color Space Conversions: BGR/RGB conversions and grayscale processing
  
  Pre-processing: Gaussian blur for noise reduction in edge detection


# Video Processing Engine
  Frame Rate Maintenance: Consistent FPS throughout processing
  
  Resource Cleanup: Proper release of video capture and writer objects
  
  Temporary File Management: Secure handling of large video files
  
  Progress Tracking: Frame-by-frame processing with completion metrics

# Client-Server Architecture
  RESTful API Design: Clean endpoint structure for each processing operation
  
  Binary Data Handling: Efficient image/video transmission via byte streams
  
  Error Handling: Comprehensive exception handling and user feedback
  
  CORS Management: Proper cross-origin resource sharing configuration

## Performance Optimizations
  Memory Efficiency
  Stream Processing: Chunk-based file handling for large videos
  
  Garbage Collection: Automatic cleanup of OpenCV objects
  
  Buffer Management: IO buffer optimization for image

## Processing Optimizations
  Multi-threading Ready: Architecture supports parallel processing
  
  GPU Acceleration: OpenCV configurations for hardware acceleration
  
  Cache Management: Efficient temporary file and result caching

## Security Features
  File Upload Security: Secure filename handling and type validation
  
  Size Limitations: 200MB maximum file size protection
  
  Input Sanitization: Protection against path traversal attacks
  
  Resource Limits: Processing timeout and memory limit enforcement

## User Interface Features
  Interactive Controls
  Real-time Previews: Instant visual feedback for all operations
  
  Parameter Adjustment: Live sliders and input controls
  
  Drag & Drop ROI: Mouse-based region selection
  
  Visual Feedback: Animated transitions and status indicators

  Responsive Design
  Mobile Compatibility: Adaptive layout for various screen sizes
  
  Touch Support: Gesture-based controls for mobile devices
  
  Progressive Enhancement: Graceful degradation for older browsers

# Deployment & Configuration
  Environment Requirements
  Python 3.7+ with OpenCV 4.5+
  
  Flask 2.0+ for web server functionality
  
  Modern Browser with Canvas and File API support
  
  Adequate Memory for large file processing
  
  Configuration Options
  Upload and output directory customization
  
  Maximum file size adjustments
  
  Processing timeout configurations
  
  Quality settings for video encoding

# Workflow Integration
  The application supports complete processing workflows:
  
  Upload → Process → Preview → Download
  
  Batch Operations: Multiple images with consistent parameters
  
  Pipeline Processing: Sequential operations on single files
  
  Comparative Analysis: Side-by-side result comparisons

This comprehensive toolbox demonstrates professional-grade computer vision implementation with production-ready 
features for educational, research, and practical image/video processing applications.

# References 
  The following resources were used throughout the design, development, and testing of 
  the Flask Image & Video Processing Application: 
  1. Flask Documentation – Official Flask documentation for web framework 
  features, routing, and server-side handling. 
  https://flask.palletsprojects.com/ 
  2. OpenCV Documentation – For image and video processing techniques including 
  face detection, edge detection, image blending, and video frame handling. 
  https://docs.opencv.org/ 
  3. Python Official Documentation – For Python programming references, file 
  handling, and integration with Flask and OpenCV. 
  https://docs.python.org/3/ 
  4. Stack Overflow – For practical coding solutions, debugging techniques, and 
  integration tips for Python, Flask, and OpenCV. 
  https://stackoverflow.com/ 
  5. W3Schools – For front-end development guidance including HTML, CSS, 
  JavaScript, and responsive web design techniques. 
  https://www.w3schools.com/ 
  6. TutorialsPoint – Reference for Python, Flask, and OpenCV tutorials, examples, 
  and best practices. 
  https://www.tutorialspoint.com/ 
  7. GitHub Repositories – Public repositories for example projects related to web
  based image and video processing using Flask and OpenCV. 
  https://github.com/ 
  94 
   
  8. Haar cascades - are a technique used in computer vision to detect objects in 
  images or video, most famously faces 
  https://github.com/opencv/opencv/tree/master/data/haarcascades 
  9. Font Awesome:  Used for integrating icons into the application’s UI. 
  https://fontawesome.com 
  10. Google Fonts – Used for consistent typography styling across the application. 
  https://fonts.google.com/  
  11. Research Papers & Articles – Online papers and articles for advanced 
  techniques in computer vision, face detection, and video analytics. 
