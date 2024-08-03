# Camera Calibration with ArUco Markers

https://www.youtube.com/watch?v=Xo1Q3AOIi9k


This project provides a script for calibrating multiple cameras using ArUco markers. The script includes visual aids to assist with camera alignment and saves calibration data for each camera.

## Features

- Visual aids (cross, arrows, floating bar) to guide camera alignment
- Calculates and displays the distance to the ArUco marker
- Saves calibration data to JSON files
- Adjustable thresholds for visual aids
- Scale factor for adapting the GUI to different resolutions

## Requirements

- Python 3.x
- OpenCV
- NumPy

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/camera-calibration.git
    cd camera-calibration
    ```

2. Install the required packages:
    ```sh
    pip install opencv-python-headless numpy
    ```

## Usage

1. Run the script:
    ```sh
    python calibrate_camera.py
    ```

2. Enter the size of the ArUco marker when prompted.

3. Follow the visual aids to align the camera with the marker.

4. Press `q` to save the calibration data and proceed to the next camera.

5. Calibration data is saved in the script's directory as `camera_<camera_index>_calibration.json`.

## Adjustable Parameters

- `angle_threshold`: Threshold for the floating bar to turn green (default: 5 degrees).
- `arrow_threshold`: Threshold for arrows to disappear (default: 10 pixels).
- `scale_factor`: Scale factor for GUI elements (default: 1.0).
- `size_factor`: Factor for size calculations for extrinsic data (default: 1.0).

These parameters can be adjusted within the script to fit your needs.

## Example Calibration Data

Example calibration data saved in `camera_<camera_index>_calibration.json`:
```json
{
    "camera_index": 0,
    "scale_factor": 1.0,
    "size_factor": 1.0,
    "angle_threshold": 5,
    "arrow_threshold": 10,
    "aruco_size": 50.0,
    "extrinsic_data": "example_data"
}
