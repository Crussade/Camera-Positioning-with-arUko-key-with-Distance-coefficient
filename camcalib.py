import cv2
import numpy as np
import cv2.aruco as aruco
import json
import os

# Parameters for the ArUco marker
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
aruco_params = aruco.DetectorParameters_create()

# Adjustable parameters
angle_threshold = 5  # Threshold for the floating bar to turn green
arrow_threshold = 10  # Threshold for arrows to disappear
scale_factor = 1.0  # Scale factor for GUI elements
size_factor = 1.0  # Factor for size calculations for extrinsic data

# Function to draw a big cross, interactive arrows, a floating bar, and indicator bars for guidance
def draw_guidance(frame, marker_corners, aruco_size):
    h, w = frame.shape[:2]
    center = (int(w // 2 * scale_factor), int(h // 2 * scale_factor))
    cross_length = int(100 * scale_factor)
    arrow_length = int(50 * scale_factor)
    bar_length = int(200 * scale_factor)
    bar_spacing = int(60 * scale_factor)
    bar_x_pos = int((w - 120) * scale_factor)  # Move bars to the left by increasing this offset
    bar_y_center = int(h // 2 * scale_factor)

    # Draw the big cross
    cv2.line(frame, (center[0] - cross_length, center[1]), (center[0] + cross_length, center[1]), (255, 255, 255), 2)
    cv2.line(frame, (center[0], center[1] - cross_length), (center[0], center[1] + cross_length), (255, 255, 255), 2)

    if marker_corners is not None:
        marker_center = marker_corners.mean(axis=0).astype(int)

        # Draw an arrow towards the marker center if offset is above the threshold
        if abs(marker_center[0] - center[0]) > arrow_threshold or abs(marker_center[1] - center[1]) > arrow_threshold:
            cv2.arrowedLine(frame, center, tuple(marker_center), (0, 255, 0), 2, tipLength=0.5)

        # Draw the marker center
        cv2.circle(frame, tuple(marker_center), 5, (255, 0, 0), -1)

        # Calculate offsets
        offset_x = marker_center[0] - center[0]
        offset_y = marker_center[1] - center[1]

        # Interactive arrows for guidance, disappear if offset is below threshold
        if abs(offset_x) > arrow_threshold:
            if offset_x > 0:
                # Move camera left
                cv2.arrowedLine(frame, (center[0] + cross_length, center[1]), (center[0] + cross_length + arrow_length, center[1]), (0, 0, 255), 2, tipLength=0.5)  # Red arrow
            elif offset_x < 0:
                # Move camera right
                cv2.arrowedLine(frame, (center[0] - cross_length, center[1]), (center[0] - cross_length - arrow_length, center[1]), (0, 0, 255), 2, tipLength=0.5)  # Red arrow

        if abs(offset_y) > arrow_threshold:
            if offset_y > 0:
                # Move camera up
                cv2.arrowedLine(frame, (center[0], center[1] + cross_length), (center[0], center[1] + cross_length + arrow_length), (0, 0, 255), 2, tipLength=0.5)  # Red arrow
            elif offset_y < 0:
                # Move camera down
                cv2.arrowedLine(frame, (center[0], center[1] - cross_length), (center[0], center[1] - cross_length - arrow_length), (0, 0, 255), 2, tipLength=0.5)  # Red arrow

        # Calculate the rotation angle (roll) from the marker corners
        vector_x = marker_corners[1] - marker_corners[0]
        angle_x = np.degrees(np.arctan2(vector_x[1], vector_x[0]))

        # Determine the color of the floating bar
        bar_color = (0, 255, 0) if abs(angle_x) < angle_threshold else (0, 0, 255)  # Green if level, red otherwise

        # Draw the floating bar
        def draw_rotated_bar(center, angle, length, color, thickness=4):
            angle_rad = np.radians(angle)
            half_length = length // 2
            start_point = (int(center[0] - half_length * np.cos(angle_rad)), int(center[1] - half_length * np.sin(angle_rad)))
            end_point = (int(center[0] + half_length * np.cos(angle_rad)), int(center[1] + half_length * np.sin(angle_rad)))
            cv2.line(frame, start_point, end_point, color, thickness)

        draw_rotated_bar(center, angle_x, cross_length * 2, bar_color)

        # Draw indicator bars for roll, x, and y on the right side of the screen
        bar_y_start = bar_y_center - bar_length // 2
        bar_y_end = bar_y_center + bar_length // 2

        # Indicator for X offset
        x_bar_color = (0, 255, 0) if abs(offset_x) < arrow_threshold else (0, 0, 255)
        cv2.line(frame, (bar_x_pos - bar_spacing, bar_y_start), 
                        (bar_x_pos - bar_spacing, bar_y_end), x_bar_color, 2)
        cv2.line(frame, (bar_x_pos - bar_spacing - 10, bar_y_center), 
                        (bar_x_pos - bar_spacing + 10, bar_y_center), (255, 255, 255), 1)  # Midpoint marker
        offset_x_pos = int(bar_y_center + (offset_x / w) * bar_length)
        cv2.circle(frame, (bar_x_pos - bar_spacing, offset_x_pos), 5, x_bar_color, -1)
        cv2.putText(frame, "X", (bar_x_pos - bar_spacing - 20, bar_y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, x_bar_color, 2)

        # Indicator for Y offset
        y_bar_color = (0, 255, 0) if abs(offset_y) < arrow_threshold else (0, 0, 255)
        cv2.line(frame, (bar_x_pos, bar_y_start), 
                        (bar_x_pos, bar_y_end), y_bar_color, 2)
        cv2.line(frame, (bar_x_pos - 10, bar_y_center), 
                        (bar_x_pos + 10, bar_y_center), (255, 255, 255), 1)  # Midpoint marker
        offset_y_pos = int(bar_y_center + (offset_y / h) * bar_length)
        cv2.circle(frame, (bar_x_pos, offset_y_pos), 5, y_bar_color, -1)
        cv2.putText(frame, "Y", (bar_x_pos - 10, bar_y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, y_bar_color, 2)

        # Indicator for Roll (angle_x)
        roll_bar_color = (0, 255, 0) if abs(angle_x) < angle_threshold else (0, 0, 255)
        cv2.line(frame, (bar_x_pos + bar_spacing, bar_y_start), 
                        (bar_x_pos + bar_spacing, bar_y_end), roll_bar_color, 2)
        cv2.line(frame, (bar_x_pos + bar_spacing - 10, bar_y_center), 
                        (bar_x_pos + bar_spacing + 10, bar_y_center), (255, 255, 255), 1)  # Midpoint marker
        roll_pos = int(bar_y_center + (angle_x / 90) * (bar_length // 2))  # Adjust calculation for correct mapping
        cv2.circle(frame, (bar_x_pos + bar_spacing, roll_pos), 5, roll_bar_color, -1)
        cv2.putText(frame, "Roll", (bar_x_pos + bar_spacing - 20, bar_y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, roll_bar_color, 2)

        # Calculate and display distance
        if len(marker_corners) == 4:
            top_left, top_right, bottom_right, bottom_left = marker_corners
            width = np.linalg.norm(top_right - top_left)
            height = np.linalg.norm(top_left - bottom_left)
            # Assume square markers for simplicity; otherwise, use width and height appropriately
            marker_size_in_pixels = (width + height) / 2
            distance = (aruco_size * size_factor) / marker_size_in_pixels
            cv2.putText(frame, f"Distance: {distance:.2f} units", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "Marker #6 not detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# Function to detect ArUco marker #6
def detect_aruco_marker(frame, aruco_dict, aruco_params, target_marker_id=6):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    
    if ids is not None:
        print(f"Detected marker IDs: {ids.flatten()}")
        for i, id in enumerate(ids.flatten()):
            if id == target_marker_id:
                aruco.drawDetectedMarkers(frame, corners, ids)
                return corners[i][0]
    return None

# Function to save calibration data
def save_calibration_data(camera_index, calibration_data):
    directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(directory, f'camera_{camera_index}_calibration.json')
    with open(file_path, 'w') as f:
        json.dump(calibration_data, f)
    print(f"Calibration data for camera {camera_index} saved to {file_path}")

# Initialize video capture for the camera
def calibrate_camera():
    camera_index = 0
    aruco_size = float(input("Enter the size of the ArUco marker in units: "))  # Prompt for ArUco marker size
    
    while True:
        print(f"Starting calibration for camera {camera_index}")
        cap = cv2.VideoCapture(camera_index)  # Change the index based on your camera setup
        if not cap.isOpened():
            print(f"Camera {camera_index} not found. Exiting...")
            break
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect ArUco marker #6
            marker_corners = detect_aruco_marker(frame, aruco_dict, aruco_params, target_marker_id=6)
            
            # Draw guidance on the frame
            draw_guidance(frame, marker_corners, aruco_size)
            
            # Display the frame
            cv2.imshow('Camera', frame)
            
            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release the capture and destroy all windows
        cap.release()
        cv2.destroyAllWindows()

        # Save calibration data (example data, replace with actual data)
        calibration_data = {
            'camera_index': camera_index,
            'scale_factor': scale_factor,
            'size_factor': size_factor,
            'angle_threshold': angle_threshold,
            'arrow_threshold': arrow_threshold,
            'aruco_size': aruco_size,
            'extrinsic_data': 'example_data'  # Replace with actual extrinsic data
        }
        save_calibration_data(camera_index, calibration_data)

        # Ask user if they want to continue with the next camera
        continue_response = input("Do you want to calibrate the next camera? (yes/no): ").strip().lower()
        if continue_response != 'yes':
            break

        camera_index += 1

if __name__ == "__main__":
    calibrate_camera()
