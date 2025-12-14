# config.py

# Screen resolution (your setup)
SCREEN_WIDTH = 2560
SCREEN_HEIGHT = 1440

# Region where the "Ad" indicator appears (YOU WILL TUNE THIS)
# Format: (left, top, width, height)
AD_REGION = (
    55,   # x
    1230,   # y
    280,    # width
    100     # height
)


# Popup search region dimensions
POPUP_WIDTH = 300
POPUP_HEIGHT = 120

# Offset from screen edges
POPUP_OFFSET_RIGHT = 20
POPUP_OFFSET_BOTTOM = 120


POPUP_TEMPLATE_PATH = "assets/popup_template.png"

POPUP_MATCH_THRESHOLD = 0.85


# Expected text during ads
EXPECTED_TEXT = "Ad"   # or whatever 2 letters you’re detecting

# OCR settings
TESSERACT_CONFIG = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# Logging
LOG_FILE = "logs/ad_log.csv"
CAPTURE_INTERVAL_SECONDS = 1
