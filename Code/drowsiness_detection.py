from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib # type: ignore
import cv2
import winsound

# Set beep alert frequency and duration for drowsiness detection
frequency = 2500  # Frequency in Hz
duration = 1000   # Duration in ms

def eyeAspectRatio(eye):
    # Calculate the distances between the vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Calculate the distance between the horizontal eye landmarks
    C = dist.euclidean(eye[0], eye[3])
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

# Initialize counters and thresholds
count = 0
earThresh = 0.3  # EAR threshold for drowsiness
earFrames = 48   # Number of consecutive frames below the threshold for alert
shapePredictor = "shape_predictor_68_face_landmarks.dat"

# Access the webcam
cam = cv2.VideoCapture(0)

# Initialize dlib's face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor(shapePredictor)
except Exception as e:
    print(f"Error loading shape predictor: {e}")
    exit()

# Get the indexes for the left and right eyes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

while True:
    # Capture frame-by-frame from webcam
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Corrected COLOR_BGR2RGB to COLOR_BGR2GRAY

    # Detect faces in the grayscale frame
    rects = detector(gray, 0)

    for rect in rects:
        # Determine the facial landmarks for the face region
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)  # Convert shape to NumPy array

        # Extract the left and right eye coordinates
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # Calculate EAR for both eyes
        leftEar = eyeAspectRatio(leftEye)
        rightEar = eyeAspectRatio(rightEye)
        ear = (leftEar + rightEar) / 2.0

        # Draw contours around the eyes
        leftEyeHULL = cv2.convexHull(leftEye)
        rightEyeHULL = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHULL], -1, (0, 0, 255), 1)
        cv2.drawContours(frame, [rightEyeHULL], -1, (0, 0, 255), 1)

        # Check if EAR is below the threshold
        if ear < earThresh:
            count += 1
            # Trigger alert if drowsiness is detected
            if count >= earFrames:
                cv2.putText(frame, "DROWSINESS DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                winsound.Beep(frequency, duration)
        else:
            count = 0

    # Display the frame
    cv2.imshow("Drowsiness Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release the webcam and close windows
cam.release()
cv2.destroyAllWindows()

