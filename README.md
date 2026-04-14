Real-Time Face Identification System using ESP32-CAM & Computer Vision

📌 Project Overview
This project is a high-performance Wireless Face Identification System that integrates IoT hardware with Deep Learning-based Computer Vision. It leverages the ESP32-CAM module to stream live video data over a Wi-Fi network to a Python-based processing engine. The system is designed to detect, recognize, and log identified individuals in real-time with high precision.

🛠️ Technical Architecture
The system operates on a client-server model:
1) The Edge Node (Hardware): An ESP32-CAM captures high-resolution frames and hosts an asynchronous web server to stream MJPEG data over the local network (LAN).
2) The Processing Engine (Software): A Python backend fetches the stream and utilizes the Dlib-based Face Recognition library to generate 128-dimensional facial embeddings for real-time matching.

🚀 Key Features
1) Wireless Vision: Full Wi-Fi-based video streaming, eliminating the need for physical data cables.
2) High-Accuracy Recognition: Uses state-of-the-art Deep Learning models for facial encoding and distance matching (Euclidean distance).
3) Automated Data Logging: Generates a real-time digital log of all identified subjects with precise timestamps.
4) Scalable Database: Support for dynamic dataset updates—simply add a new image to the directory without needing to retrain the model.
5) Low Latency Processing: Optimized for real-time identification with minimal lag between the camera sensor and the recognition output.

💻 Tech Stack
1) Hardware: ESP32-CAM Module, FTDI Programmer.
2) Firmware: C++ (Arduino Framework) for ESP32 Wi-Fi & Camera configuration.
3) Backend: Python 3.x.
5) Libraries: * OpenCV: For image pre-processing and frame manipulation.
i) Face_Recognition: For generating and comparing facial encodings.
ii) NumPy: For optimized mathematical operations on image arrays.
iii) Requests: For handling the MJPEG stream over HTTP.

📂 System Workflow
1) Initialization: The system loads authorized face encodings from a local directory.
2) Stream Acquisition: The Python script connects to the ESP32-CAM's IP address.
3) Face Detection: Each frame is scanned for facial landmarks.
4) Identification: Detected faces are compared against the database using a threshold-based matching algorithm.
5) Logging: Upon a successful match, the identity and timestamp are recorded automatically.

🔧 Installation & Setup
1. Hardware Flash: Upload the camera firmware to the ESP32-CAM using the Arduino IDE.
2. Execution: Run the main script with the ESP32-CAM's local IP address.
     python main_recognition.py
