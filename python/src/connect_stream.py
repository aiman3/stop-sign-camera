# Usage: run "python ./python/device/RPi_4/stream.py" on RPi first, then run this script:
#   python connect_stream.py

import cv2

cap = cv2.VideoCapture("tcp://raspberrypi.local:10001")
while (True):
    ret, frame = cap.read()
    cv2.imshow('frame', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
