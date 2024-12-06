**Drowsiness Detection System** 🚗💤
  This project implements a real-time drowsiness detection system using the Eye Aspect Ratio (EAR) method. The system monitors a user's eye movements and triggers an audio alert when drowsiness is detected, enhancing road safety.

🚀 **Features**
**Real-time Detection**: Continuously monitors eye movements using a webcam to calculate the EAR.
**Alert System**: Beeps when drowsiness is detected for immediate attention.
**Scalable Design**: Can be expanded to include additional features like yawning detection and head pose analysis.

🛠️ **Technologies Used**
**Python**: Programming language for implementation.
**OpenCV**: For video processing and image analysis.
**Dlib**: For facial landmark detection.
**SciPy**: For calculating distances required for EAR.

📂 **Repository Structure**
Drowsiness-Detection-System/
│
├── Code/
│   ├── drowsiness_detection.py                 # Main Python script
│   ├── requirements.txt                        # Dependencies list
│   ├── shape_predictor_68_face_landmarks.dat   # Pre-trained model
├──Report
│   ├── Project Report.pdf                      # Project Report
└──   README.md                                 # Documentation

⚙️ **How to Use**
1. **Prerequisites**
  *Install Python 3.8+.
  *Download the pre-trained Dlib shape predictor model:
  *shape_predictor_68_face_landmarks.dat  
  *Extract and place it in the Code/ directory.
2. **Installation**
Clone the repository:
    git clone https://github.com/NaveenCodeCraft/Drowsiness-Detection-System.git
    cd Drowsiness-Detection-System/Code
Install dependencies:
    pip install -r requirements.txt
3. **Running the Program**
    Make sure the shape_predictor_68_face_landmarks.dat file is in the same directory as the script.
    Run the script:
      python drowsiness_detection.py
   
📊 **How It Works**
Facial Landmark Detection:
The system uses Dlib’s 68 facial landmark predictor to locate key points around the eyes.
  ***EAR Calculation:**
    The Eye Aspect Ratio (EAR) is computed using specific eye landmarks to detect the openness of the eyes:

          EAR= horizontal distance / vertical distance​
     If the EAR falls below a threshold (0.3) for a certain number of frames (48), the system triggers an alert.
  ***Alert Mechanism**:
    A beep sound alerts the user when drowsiness is detected.
  
🤝 **Contributions**
  Contributions are welcome! Feel free to fork the repository, make improvements, and submit a pull request.

🌟 **Acknowledgments**
  Dlib for providing the facial landmark predictor.
  OpenCV for the image processing framework.
  Inspiration from the PyImageSearch tutorials.
