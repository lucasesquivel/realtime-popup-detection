# AD Detector

**AD Detector** is a Python-based automated popup detection and persistent overlay system.  
It identifies popups on the screen in real-time and displays a configurable semi-transparent overlay to prevent distractions or interruptions. Designed with modularity and configurability in mind, this project demonstrates strong Python engineering, UI handling, and automation skills.

---

## Features

- **Real-time Popup Detection**: Utilizes template matching to identify popups efficiently.
- **Persistent Overlay**: Configurable color and transparency overlay that adapts to screen dimensions.
- **Debug Mode**: Optional debug windows for ROI inspection, ensuring easy troubleshooting and fine-tuning.
- **Highly Configurable**: All parameters (overlay size, color, transparency, debug mode) are centralized in a `config.py` for easy adjustments.
- **Clean Logging**: Activity and detection events are logged in a structured manner for monitoring.

---

## Project Structure

adDetector/

│

├─ adDetector/ # Core source code

│ ├─ init.py

│ ├─ config.py # Centralized configuration

│ ├─ detector.py # Main detection logic

│ └─ find_coords.py # Helper functions for ROI and coordinates

│

├─ assets/ # Popup templates and other resources

│ └─ popup_template.png

│

├─ logs/ # Optional runtime logs

│

├─ tests/ # Optional unit tests

│

├─ .gitignore

├─ requirements.txt # Python dependencies

├─ README.md

└─ LICENSE


---

## Installation

1. Clone the repository:

<!-- ```bash
git clone https://github.com/yourusername/adDetector.git
cd adDetector -->


2. Create a virtual environment:
python -m venv venv
source venv/bin/activate  # Linux / macOS
venv\Scripts\activate     # Windows


3.  Install dependencies:
pip install -r requirements.txt


## Usage

Run the main detector:
python -m adDetector.detector
## Optional: Enable debug mode in config.py to see ROI windows for real-time detection visualization.


## Configuration

All overlay and detection parameters are centralized in config.py:
# Example:
 - OVERLAY_WIDTH = SCREEN_WIDTH - 600
 - OVERLAY_HEIGHT = SCREEN_HEIGHT - 600
 - OVERLAY_COLOR = "black"
 - OVERLAY_ALPHA = 0.7
 - DEBUG = True

This allows you to quickly adjust the overlay appearance, logging, and debug behavior without modifying core logic.


## Technologies & Skills Demonstrated
 - Python 3.x & OpenCV for image processing
 - Modular, maintainable project architecture
 - Real-time screen capture and overlay rendering
 - Logging and debugging best practices
 - Config-driven design for flexibility and scalability


## Screenshots: 
(work in progress)


## License
This project is open-source under the MIT License. See LICENSE for details.


## Optional Enhancements / Future Work
 - Multi-monitor support for full-screen applications
 - Dynamic template updates via user input
 - Machine learning-based popup classification for smarter detection
