# Sentinel

**Sentinel** is a desktop security monitoring application built with Python, PySide6, and OpenCV. It uses your camera to recognize registered people, detect unknown individuals, and automatically record events as they happen. The application allows you to monitor your camera feed, manage recognized people, and review past activity from a single program.

---

## Features

- Real-time face recognition for registered people
- Detection and tracking of unknown individuals
- Automatic recording when people enter and exit the camera's view
- People management:
  - Register new people
  - Rename people
  - Remove people
  - View past recorded appearances
- View event history with recorded thumbnails, timestamps, detected people, and video playback
- Filter events by person, year, month, or day
- Switch between connected cameras
- Toggle face recognition and event recording
- Store face data, recordings, and event metadata locally
- Background recognition processing to keep the application responsive

Sentinel continuously processes the camera feed, identifies detected faces, and tracks people across multiple frames before creating an event. This helps prevent temporary detection errors from creating unnecessary recordings.

---

## Installation & How to Use

### Requirements

- Python 3.10+
- Compatible webcam or camera
- PySide6
- OpenCV
- NumPy

### Installation

Download all files and folders within this repository and install the required Python libraries.

## How to Use

1. Launch Sentinel by running `main.pv` on your device, with at least one compatible camera connected to it.
2. Open **Settings** and select the camera you want to use.
3. Open **People** and select **+ Add Person** to register someone.
4. Enter their name and follow the on-screen instructions to capture their face.
5. Return to the **Dashboard** to begin monitoring. Sentinel will recognize registered people and detect unknown individuals as they appear.
6. Open **Events** to review recorded activity.
7. Use the filters to find events by person or date, or select **Play** to view a recording.

---

## Privacy

Sentinel is designed to operate locally. Registered face data, recordings, thumbnails, and event metadata are stored on the computer running the application.

Because face data and recorded video may contain sensitive personal information, access to the computer and Sentinel's storage directories should be appropriately protected.

If you use Sentinel to monitor other people, ensure that you have the appropriate consent and comply with applicable privacy regulations.

Face recognition may produce false positives and false negatives, so Sentinel should not be treated as an infallible identification system.

---

## License

[MIT License](https://opensource.org/license/MIT)

---
