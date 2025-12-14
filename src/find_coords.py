import mss
import cv2
import numpy as np

with mss.mss() as sct:
    monitor = sct.monitors[1]  # full screen
    img = np.array(sct.grab(monitor))
    cv2.imshow("Full Screen", img)
    cv2.setMouseCallback(
        "Full Screen",
        lambda event, x, y, flags, param:
            print(f"x={x}, y={y}") if event == cv2.EVENT_LBUTTONDOWN else None
    )
    cv2.waitKey(0)
