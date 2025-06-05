import cv2
import mediapipe as mp

def main():
    # Initialize MediaPipe Pose model
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Open a video capture
    cap = cv2.VideoCapture(0)  # You can also use a video file by specifying the file path

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Pose
        results = pose.process(rgb_frame)

        # If pose landmarks are detected, you can extract and use them
        if results.pose_landmarks:
            # Access landmarks for the left and right arms
            left_arm_landmarks = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value:mp_pose.PoseLandmark.LEFT_WRIST.value + 1]
            right_arm_landmarks = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value:mp_pose.PoseLandmark.RIGHT_WRIST.value + 1]

            # Draw the landmarks on the frame
            draw_landmarks(frame, left_arm_landmarks)
            draw_landmarks(frame, right_arm_landmarks)

        # Display the frame
        cv2.imshow('Arm Pose Detection', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

def draw_landmarks(frame, landmarks):
    h, w, _ = frame.shape
    for landmark in landmarks:
        cx, cy = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)  # Draw a circle at each landmark

if __name__ == "__main__":
    main()
