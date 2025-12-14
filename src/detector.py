import time
import re
import csv
from datetime import datetime
import threading

import numpy as np
import cv2
import mss
import pytesseract
import tkinter as tk

from pynput.keyboard import Controller
from pynput.mouse import Controller as MouseController
from pynput.mouse import Button


from src.config import (
    AD_REGION,
    EXPECTED_TEXT,
    TESSERACT_CONFIG,
    LOG_FILE
)




# ------------------ DEBUG FLAG ------------------ #
DEBUG = True # ← set to False to disable debug windows
# ------------------------------------------------ #


from src.config import POPUP_WIDTH, POPUP_HEIGHT, POPUP_OFFSET_RIGHT, POPUP_OFFSET_BOTTOM, POPUP_TEMPLATE_PATH, POPUP_MATCH_THRESHOLD

popup_template = cv2.imread(POPUP_TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
if popup_template is None:
    raise RuntimeError("Popup template image not found")


CAPTURE_INTERVAL_SECONDS = 0.75
MISS_COUNT_THRESHOLD = 5

mouse = MouseController()
keyboard = Controller()

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


SCREEN_WIDTH = 2560
SCREEN_HEIGHT = 1440

OVERLAY_WIDTH = SCREEN_WIDTH - 600
OVERLAY_HEIGHT = SCREEN_HEIGHT - 600
OVERLAY_COLOR = "black"
OVERLAY_ALPHA = 0.7

# Compute absolute coordinates for popup search region
POPUP_LEFT = SCREEN_WIDTH - POPUP_OFFSET_RIGHT - POPUP_WIDTH
POPUP_TOP = SCREEN_HEIGHT - POPUP_OFFSET_BOTTOM - POPUP_HEIGHT

POPUP_REGION = (
    POPUP_LEFT,
    POPUP_TOP,
    POPUP_WIDTH,
    POPUP_HEIGHT
)


# ---------------- DEBUG WINDOWS SETUP ---------------- #
if DEBUG:
    # Create named windows once to avoid spam-opening new windows
    cv2.namedWindow("DEBUG: Raw ROI", cv2.WINDOW_NORMAL)
    cv2.namedWindow("DEBUG: Popup ROI", cv2.WINDOW_NORMAL)


class PersistentOverlay:
    def __init__(self, width=OVERLAY_WIDTH, height=OVERLAY_HEIGHT, 
                 color=OVERLAY_COLOR, alpha=OVERLAY_ALPHA):
        self.width = width
        self.height = height
        self.color = color
        self.alpha = alpha
        self.root = None
        self.thread = None

    def _create_window(self):
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        x = (SCREEN_WIDTH - self.width) // 2
        y = (SCREEN_HEIGHT - self.height) // 2
        self.root.geometry(f"{self.width}x{self.height}+{x}+{y}")
        self.root.configure(bg=self.color)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", self.alpha)
        self.root.withdraw()
        self.root.mainloop()

    def start(self):
        self.thread = threading.Thread(target=self._create_window, daemon=True)
        self.thread.start()
        while self.root is None:
            time.sleep(0.05)

    def show(self):
        if self.root:
            self.root.deiconify()

    def hide(self):
        if self.root:
            self.root.withdraw()


def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    return gray, thresh   # ← MODIFIED (return both)


def detect_ad_text(img):
    return pytesseract.image_to_string(img, config=TESSERACT_CONFIG).strip()


def is_ad_detected(ocr_text):
    cleaned = re.sub(r'[^a-z]', '', ocr_text.lower())
    return EXPECTED_TEXT.lower() in cleaned


def log_detection(timestamp, detected_text):
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, detected_text])


def detect_popup(frame_gray):
    fh, fw = frame_gray.shape
    th, tw = popup_template.shape

    # SAFETY CHECK: template must fit inside search region
    if th > fh or tw > fw:
        if DEBUG:
            print(
                f"[DEBUG] Popup template ({tw}x{th}) "
                f"is larger than search region ({fw}x{fh}) — skipping match"
            )
        return False, None, 0.0

    result = cv2.matchTemplate(frame_gray, popup_template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val >= POPUP_MATCH_THRESHOLD:
        center_x = max_loc[0] + tw // 2
        center_y = max_loc[1] + th // 2
        return True, (center_x, center_y), max_val

    return False, None, max_val


def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))


def main():
    print("Starting YouTube Ad Detector (Press Ctrl+C to stop)")
    print("DEBUG MODE:", DEBUG)

    ad_active = False
    miss_count = 0
    last_popup_click_time = 0.0
    POPUP_CLICK_COOLDOWN = 5.0  # seconds


    overlay = PersistentOverlay(
        OVERLAY_WIDTH,
        OVERLAY_HEIGHT,
        color=OVERLAY_COLOR,
        alpha=OVERLAY_ALPHA
    )
    overlay.start()

    with mss.mss() as sct:
        ad_monitor = {
            "left": AD_REGION[0],
            "top": AD_REGION[1],
            "width": AD_REGION[2],
            "height": AD_REGION[3],
        }

        popup_monitor = {
            "left": POPUP_REGION[0],
            "top": POPUP_REGION[1],
            "width": POPUP_REGION[2],
            "height": POPUP_REGION[3],
        }

        try:
            while True:
                # ---------------- AD OCR REGION ----------------
                screenshot = sct.grab(ad_monitor)
                img = np.array(screenshot)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                gray, processed = preprocess_image(img_bgr)
                detected_text = detect_ad_text(processed)
                ad_detected = is_ad_detected(detected_text)

                # -------- DEBUG: RAW AD REGION --------
                if DEBUG:
                    debug_ad = img_bgr.copy()
                    cv2.rectangle(
                        debug_ad,
                        (0, 0),
                        (ad_monitor["width"] - 1, ad_monitor["height"] - 1),
                        (0, 255, 0),
                        1
                    )
                    cv2.putText(
                        debug_ad,
                        f"OCR: {detected_text}",
                        (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )
                    cv2.imshow("DEBUG: Raw ROI", debug_ad)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # ---------------- AD START ----------------
                if ad_detected:
                    miss_count = 0

                    if not ad_active:
                        timestamp = datetime.now().isoformat(timespec="seconds")
                        print(f"[{timestamp}] AD START → '{detected_text}'")
                        log_detection(timestamp, detected_text)

                        keyboard.press('m')
                        keyboard.release('m')

                        overlay.show()
                        ad_active = True

                    # ---------------- POPUP REGION ----------------
                    popup_img = np.array(sct.grab(popup_monitor))
                    popup_bgr = cv2.cvtColor(popup_img, cv2.COLOR_BGRA2BGR)
                    popup_gray = cv2.cvtColor(popup_bgr, cv2.COLOR_BGR2GRAY)

                    found, center, score = detect_popup(popup_gray)

                    # -------- DEBUG: RAW POPUP REGION --------
                    if DEBUG:
                        debug_popup = popup_bgr.copy()

                        # Draw ROI boundary
                        cv2.rectangle(
                            debug_popup,
                            (0, 0),
                            (popup_monitor["width"] - 1, popup_monitor["height"] - 1),
                            (255, 0, 255),  # pink
                            2
                        )

                        # Only draw center if popup was found
                        if found and center is not None:
                            cv2.circle(
                                debug_popup,
                                (int(center[0]), int(center[1])),
                                6,
                                (0, 0, 255),  # red dot
                                -1
                            )

                        cv2.putText(
                            debug_popup,
                            f"Match: {score:.2f}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2
                        )

                        cv2.imshow("DEBUG: Popup ROI", debug_popup)

                    if found:
                        now = time.time()

                        if now - last_popup_click_time >= POPUP_CLICK_COOLDOWN:
                            raw_x = popup_monitor["left"] + center[0]
                            raw_y = popup_monitor["top"] + center[1]

                            click_x = clamp(
                                raw_x,
                                popup_monitor["left"] + 5,
                                popup_monitor["left"] + popup_monitor["width"] - 5
                            )

                            click_y = clamp(
                                raw_y,
                                popup_monitor["top"] + 5,
                                popup_monitor["top"] + popup_monitor["height"] - 5
                            )

                            if DEBUG:
                                print(
                                    f"[DEBUG] Popup click at ({click_x}, {click_y}) "
                                    f"within region x[{popup_monitor['left']}, "
                                    f"{popup_monitor['left'] + popup_monitor['width']}] "
                                    f"y[{popup_monitor['top']}, "
                                    f"{popup_monitor['top'] + popup_monitor['height']}]"
                                )

                            mouse.position = (click_x, click_y)
                            time.sleep(0.15)
                            mouse.click(Button.left, 1)

                            last_popup_click_time = now

                        else:
                            if DEBUG:
                                print(
                                    f"[DEBUG] Popup detected but cooldown active "
                                    f"({now - last_popup_click_time:.1f}s elapsed)"
                                )

                # ---------------- AD END (MISS COUNTER) ----------------
                else:
                    if ad_active:
                        miss_count += 1
                        if miss_count >= MISS_COUNT_THRESHOLD:
                            timestamp = datetime.now().isoformat(timespec="seconds")
                            print(f"[{timestamp}] AD END")
                            log_detection(timestamp, "Ad ended")

                            keyboard.press('m')
                            keyboard.release('m')

                            overlay.hide()
                            ad_active = False
                            miss_count = 0

                time.sleep(CAPTURE_INTERVAL_SECONDS)

        finally:
            if DEBUG:
                cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
