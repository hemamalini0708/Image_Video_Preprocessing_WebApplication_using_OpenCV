# Image_Video_Preprocessing_WebApplication_using_OpenCV

<img width="625" height="365" alt="image" src="https://github.com/user-attachments/assets/c99f8b84-3bfb-4ab8-b16c-225e1195b2cb" />

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
<img width="627" height="338" alt="image" src="https://github.com/user-attachments/assets/636b7394-1057-48f5-864c-37f90b55296e" />

  Equal Weight Blending: cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
  
  Weighted Blending: Adjustable alpha blending with user-controlled weights
  
  Automatic Resizing: Images are resized to match dimensions before blending

## 2. Geometric Transformations
<img width="623" height="482" alt="image" src="https://github.com/user-attachments/assets/e63df1b5-fc5a-4200-a002-5abdf45015c4" />

  Image Resizing: cv2.resize() with user-defined width and height
  
  Aspect Ratio Handling: Maintains proportions during reshaping operations

## 3. Region of Interest (ROI) Operations
<img width="601" height="297" alt="image" src="https://github.com/user-attachments/assets/7feae9f4-ba89-46b0-b2c1-bf4bb8fd80fc" />

  Rectangular ROI Extraction: img_cv[y:y+height, x:x+width]
  
  Interactive Selection: Canvas-based ROI selection with mouse interactions
  
  Coordinate-based Extraction: Precise pixel-level region extraction

## 4. Computer Vision Drawing Operations
<img width="939" height="665" alt="image" src="https://github.com/user-attachments/assets/985c3bf4-4911-416b-8d4f-911e38849d3c" />

  Primitive Shapes:
  
  Lines: cv2.line()
  
  Rectangles: cv2.rectangle()
  
  Circles: cv2.circle()
  
  Arrows: cv2.arrowedLine()
  
  Customizable Properties: Color, thickness, position parameters

## 5. Text & Annotation Features
<img width="555" height="502" alt="image" src="https://github.com/user-attachments/assets/3ea95311-a5bb-4533-a3bc-8842ec6e404e" />

  Text Overlay: cv2.putText() with customizable fonts and positioning
  
  DateTime Stamping: Automatic timestamp addition with multiple format options
  
  Font Control: Size, color, thickness, and position customization

## 6. Face Detection System
<img width="581" height="314" alt="image" src="https://github.com/user-attachments/assets/e3b446a5-e1d7-4bd5-876b-006093019804" />

  Haar Cascade Classifiers: cv2.CascadeClassifier()
  
  Multi-scale Detection: detectMultiScale() with scale factor and minimum neighbors
  
  Real-time Video Processing: Frame-by-frame face detection in videos
  
  Bounding Box Visualization: Customizable rectangle color and thickness

## 7. Feature Detection Algorithms
  Edge Detection:
  <img width="776" height="462" alt="image" src="https://github.com/user-attachments/assets/978e99e2-d766-43ee-a95c-71c6cf355b74" />

  Canny Edge Detector: cv2.Canny()
  
  Sobel Operator: cv2.Sobel() for gradient-based edge detection
  
  Laplacian Operator: cv2.Laplacian() for second-derivative edge detection
  
  Corner Detection:
  
  Harris Corner Detection: cv2.cornerHarris()
  
  Adaptive thresholding for corner identification

## 8. Video Processing Capabilities
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
<img width="666" height="410" alt="image" src="https://github.com/user-attachments/assets/d49e674e-4033-4637-85b9-b45740a746d9" />

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
