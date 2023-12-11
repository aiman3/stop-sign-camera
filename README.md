# stop-sign-camera

A WIP stop sign enforcement camera running on IoT devices.

## Dataset

The license plate detection model is trained on [Car License Plate Detection](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection) dataset on Kaggle

The vehicle detection model is trained on [Traffic Vehicles Object Detection](https://www.kaggle.com/datasets/saumyapatel/traffic-vehicles-object-detection/data) dataset on Kaggle

## Requirements

To be complete

## Setup

1. Run the following command to start a TCP stream on your RPi:

    ```bash
    python stop-sign-camera/python/device/RPi_4/stream.py
    ```

2. Rename ```config_exmaple.ini``` to ```config.ini``` and modify the TCP stream URL in the it. In this case, you can put

    ```ini
    [rtsp]
    url = tcp://raspberrypi.local:8554/
    ```

3. Run main.py

    ```bash
    python stop-sign-camera/python/main.py
    ```