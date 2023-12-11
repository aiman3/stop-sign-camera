# stop-sign-camera

A stop sign enforcement camera running on IoT devices.

## Dataset

The license plate detection model is trained on [Car License Plate Detection](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection) dataset on Kaggle

The vehicle detection model is trained on [Traffic Vehicles Object Detection](https://www.kaggle.com/datasets/saumyapatel/traffic-vehicles-object-detection/data) dataset on Kaggle

## Requirements

On your PC:

```txt
ultralytics==8.0.221
torch==2.1.1
torchvision==0.16.1
tqdm==4.65.0
easyocr==1.7.1
basicsr==1.4.2
opencv_contrib_python==4.8.1.78
opencv_python==4.8.1.78
opencv_python_headless==4.8.1.78
ffmpeg==1.4
```

On your Raspberry Pi:

```txt
gpiozero==2.0
picamera2==0.3.16
ffmpeg==1.4
opencv_python==4.8.1.78
```

## Setup

0. Create a environment with Anaconda (recommended)

1. Install requirements

    On your PC:

    ```bash
    pip install -r requirements_pc.txt
    ```

    On your RPi:

    ```bash
    pip install -r requirements_rpi.txt
    ```

2. Run the following command to start a TCP stream on your RPi:

    ```bash
    python stop-sign-camera/python/device/RPi_4/stream.py
    ```

3. Rename ```config_exmaple.ini``` to ```config.ini``` and modify the TCP stream URL or the path to video file in the it. In this case, you can put

    ```ini
    [video]
    source=tcp://raspberrypi.local:8554/
    output=path/to/output
    ```

    or

    ```ini
    [video]
    source=./path/to/your/video
    output=path/to/output
    ```

4. Configure the detection area (optional) and the two trigger lines before starting

   ```ini
    ; Cropping area for the detection for faster inference
    ; Leave blank if you want to infer the full frame
    [frame]
    x1=
    y1=
    x2=
    y2=

    ; the Y-axis for horizontal trigger/stop lines
    [lines]
    trigger_line=
    stop_line=
    ```

5. Run main.py

    ```bash
    python stop-sign-camera/python/main.py
    ```
