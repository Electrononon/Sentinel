import sys
import cv2
import numpy as np
import os
import platform
import subprocess
import json
import calendar

from datetime import datetime
from collections import defaultdict

from PySide6.QtCore import Qt, QTimer, QThread, Signal, QPoint
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtMultimedia import QMediaDevices, QCamera, QMediaCaptureSession, QVideoSink, QCameraFormat
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QStackedLayout,
    QScrollArea,
    QMenu,
    QMessageBox,
    QInputDialog,
    QComboBox,
    QCheckBox,
)

from Recognizer import Recognizer
from FrameRenderer import FrameRenderer
from FaceRegistration import FaceRegistration
from EventRecorder import EventRecorder


# AI THREAD
class AIWorker(QThread):
    result_ready = Signal(list)

    def __init__(self, recognizer):
        super().__init__()
        self.recognizer = recognizer
        self.frame = None
        self.running = True
        self.busy = False

    def run(self):
        while self.running:

            if self.frame is None:
                self.msleep(5)
                continue

            self.busy = True

            frame = self.frame
            self.frame = None

            # Downscale frame to improve detection speed
            small = cv2.resize(frame, (320, 240))

            sx = frame.shape[1] / small.shape[1]
            sy = frame.shape[0] / small.shape[0]

            detections = self.recognizer.detect(small, sx, sy)

            self.busy = False

            self.result_ready.emit(detections)

    def submit(self, frame):
        if self.busy:
            return

        self.frame = frame

    def stop(self):
        self.running = False


# MAIN APP
class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Sentinel")
        self.resize(1200, 700)

        # UI ROOT
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)


        # SIDEBAR
        sidebar = QVBoxLayout()

        dashboard_btn = QPushButton("Dashboard")

        people_btn = QPushButton("People")

        events_btn = QPushButton("Events")

        settings_btn = QPushButton("Settings")

        quit_btn = QPushButton("Quit")
        quit_btn.setStyleSheet("""
            QPushButton {
                background-color: #bf4545;
                color: white;
                border-radius: 8px;
                padding: 8px;
            }

            QPushButton:hover {
                background-color: #e85f5f;
            }

            QPushButton:pressed {
                background-color: #f27979;
            }
        """)

        sidebar.addWidget(dashboard_btn)
        sidebar.addWidget(people_btn)
        sidebar.addWidget(events_btn)
        sidebar.addWidget(settings_btn)
        sidebar.addStretch()
        sidebar.addWidget(quit_btn)

        sidebar_widget = QWidget()
        sidebar_widget.setLayout(sidebar)
        sidebar_widget.setFixedWidth(200)


        # CAMERA AREA
        camera_layout = QVBoxLayout()

        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("background:black;")
        self.camera_label.setMinimumSize(640, 480)

        self.status_label = QLabel("System running • Camera active")
        self.status_label.setAlignment(Qt.AlignCenter)

        camera_layout.addWidget(self.camera_label)
        camera_layout.addWidget(self.status_label)

        camera_widget = QWidget()
        camera_widget.setLayout(camera_layout)


        # EVENTS SIDEBAR
        events_layout = QVBoxLayout()

        events_layout.addWidget(QLabel("Recent Events"))

        self.events_container = QWidget()
        container_layout = QStackedLayout(self.events_container)

        # List
        self.events_list = QListWidget()

        # Empty state
        self.empty_label = QLabel("No events recorded yet.")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setStyleSheet("""
            color: gray;
            font-size: 18px;
        """)

        container_layout.addWidget(self.events_list)
        container_layout.addWidget(self.empty_label)
        container_layout.setCurrentWidget(self.empty_label)

        events_layout.addWidget(self.events_container)

        events_widget = QWidget()
        events_widget.setLayout(events_layout)
        events_widget.setFixedWidth(250)


        # CENTER STACK
        self.pages = QStackedLayout()

        # Dashboard page
        dashboard_widget = QWidget()
        dashboard_layout = QHBoxLayout()
        dashboard_widget.setLayout(dashboard_layout)

        dashboard_layout.addWidget(camera_widget)
        dashboard_layout.addWidget(events_widget)

        # People page
        self.people_widget = QWidget()
        people_layout = QVBoxLayout()
        self.people_widget.setLayout(people_layout)

        title = QLabel("SAVED PEOPLE")
        title.setStyleSheet("""
            font-size: 22px;
            font-weight: bold;
        """)

        people_layout.addWidget(title)

        self.add_person_btn = QPushButton(
            "+ Add Person"
        )

        people_layout.addWidget(self.add_person_btn)

        self.people_container = QWidget()

        self.people_layout = QVBoxLayout()

        self.people_container.setLayout(
            self.people_layout
        )

        scroll = QScrollArea()

        scroll.setWidgetResizable(
            True
        )

        scroll.setWidget(
            self.people_container
        )

        people_layout.addWidget(scroll)

        # ADD PERSON PAGE
        self.add_face_widget = QWidget()

        add_layout = QVBoxLayout()
        self.add_face_widget.setLayout(add_layout)

        title = QLabel("Register New Person")
        title.setStyleSheet("""
            font-size:24px;
            font-weight:bold;
        """)
        add_layout.addWidget(title)

        # Camera preview
        self.registration_camera = QLabel()
        self.registration_camera.setMinimumSize(640, 480)
        self.registration_camera.setAlignment(Qt.AlignCenter)
        self.registration_camera.setStyleSheet("background:black;")
        add_layout.addWidget(self.registration_camera)

        # Instruction label
        self.instruction_label = QLabel("Look straight")
        self.instruction_label.setAlignment(Qt.AlignCenter)
        add_layout.addWidget(self.instruction_label)

        # Capture button
        self.capture_btn = QPushButton("Capture")
        self.capture_btn.clicked.connect(self.capture_face)
        add_layout.addWidget(self.capture_btn)

        # Cancel button
        self.cancel_btn = QPushButton("Cancel")
        add_layout.addWidget(self.cancel_btn)


        # EVENTS PAGE
        self.events_page = QWidget()

        events_page_layout = QVBoxLayout()
        self.events_page.setLayout(events_page_layout)

        title = QLabel("Recorded Events")
        title.setStyleSheet("""
            font-size:22px;
            font-weight:bold;
        """)

        events_page_layout.addWidget(title)

        # EVENT FILTERS

        filters_layout = QHBoxLayout()

        self.person_filter = QComboBox()
        self.person_filter.addItem("All People")

        self.year_filter = QComboBox()
        self.year_filter.addItem("Any Year")

        self.month_filter = QComboBox()
        self.month_filter.addItem("Any Month")

        self.day_filter = QComboBox()
        self.day_filter.addItem("Any Day")

        filters_layout.addWidget(self.person_filter)
        filters_layout.addWidget(self.year_filter)
        filters_layout.addWidget(self.month_filter)
        filters_layout.addWidget(self.day_filter)

        self.year_filter.currentIndexChanged.connect(
            self.update_day_filter
        )

        self.month_filter.currentIndexChanged.connect(
            self.update_day_filter
        )

        self.load_event_filters()

        self.person_filter.currentIndexChanged.connect(
            self.show_recordings
        )

        self.year_filter.currentIndexChanged.connect(
            self.show_recordings
        )

        self.month_filter.currentIndexChanged.connect(
            self.show_recordings
        )

        self.day_filter.currentIndexChanged.connect(
            self.show_recordings
        )

        events_page_layout.addLayout(filters_layout)

        self.recordings_container = QWidget()
        self.recordings_layout = QVBoxLayout()
        self.recordings_container.setLayout(self.recordings_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.recordings_container)

        events_page_layout.addWidget(scroll)

        # SETTINGS PAGE

        self.settings_widget = QWidget()

        settings_layout = QVBoxLayout()
        self.settings_widget.setLayout(settings_layout)

        title = QLabel("Settings")

        title.setStyleSheet("""
            font-size:24px;
            font-weight:bold;
        """)

        settings_layout.addWidget(title)

        self.unsaved_banner = QLabel(
            "⚠ You have unsaved changes."
        )

        self.unsaved_banner.setStyleSheet("""
            QLabel {
                background: #FFF3CD;
                color: #856404;
                border: 1px solid #FFEEBA;
                border-radius: 8px;
                padding: 8px;
                font-size: 14px;
            }
        """)

        self.unsaved_banner.hide()

        settings_layout.addWidget(
            self.unsaved_banner
        )

        # CAMERA SETTINGS
        camera_title = QLabel("Camera")

        camera_title.setStyleSheet(
            "font-size:18px;font-weight:bold;"
        )

        settings_layout.addWidget(camera_title)

        camera_label = QLabel(
            "Camera Source"
        )

        settings_layout.addWidget(
            camera_label
        )

        self.camera_selector = QComboBox()

        self.camera_devices = QMediaDevices.videoInputs()

        for camera in self.camera_devices:
            self.camera_selector.addItem(
                camera.description(),
                camera
            )

        settings_layout.addWidget(
            self.camera_selector
        )
        settings_layout.addWidget(
            self.camera_selector
        )

        # AI FACE DETECTION SETTINGS
        ai_title = QLabel(
            "Recognition"
        )

        ai_title.setStyleSheet(
            "font-size:18px;font-weight:bold;"
        )

        settings_layout.addWidget(
            ai_title
        )

        self.face_detection_toggle = QCheckBox(
            "Enable Face Recognition"
        )

        self.face_detection_toggle.setChecked(
            True
        )

        settings_layout.addWidget(
            self.face_detection_toggle
        )

        # STORAGE SETTINGS

        storage_title = QLabel(
            "Storage"
        )

        storage_title.setStyleSheet(
            "font-size:18px;font-weight:bold;"
        )

        settings_layout.addWidget(
            storage_title
        )

        self.save_events_toggle = QCheckBox(
            "Save Detection Events"
        )

        self.save_events_toggle.setChecked(
            True
        )

        settings_layout.addWidget(
            self.save_events_toggle
        )

        self.apply_settings_btn = QPushButton(
            "Apply"
        )

        self.apply_settings_btn.setEnabled(False)

        settings_layout.addWidget(
            self.apply_settings_btn
        )

        self.settings_status = QLabel("")
        self.settings_status.setStyleSheet("""
            color:#4CAF50;
        """)

        settings_layout.addWidget(
            self.settings_status
        )

        settings_layout.addStretch()

        # Add pages
        self.pages.addWidget(dashboard_widget)
        self.pages.addWidget(self.people_widget)
        self.pages.addWidget(self.add_face_widget)
        self.pages.addWidget(self.events_page)
        self.pages.addWidget(self.settings_widget)

        # Add sidebar + pages
        main_layout.addWidget(sidebar_widget)
        main_layout.addLayout(self.pages)

        # Buttons
        dashboard_btn.clicked.connect(self.show_dashboard)
        people_btn.clicked.connect(self.show_people)
        events_btn.clicked.connect(self.show_recordings)
        settings_btn.clicked.connect(self.show_settings)
        quit_btn.clicked.connect(QApplication.quit)

        self.apply_settings_btn.clicked.connect(
            self.apply_settings
        )

        self.camera_selector.currentIndexChanged.connect(
            self.mark_unsaved
        )

        self.face_detection_toggle.toggled.connect(
            self.mark_unsaved
        )

        self.save_events_toggle.toggled.connect(
            self.mark_unsaved
        )

        self.add_person_btn.clicked.connect(self.show_add_face)

        # CAMERA
        self.camera_devices = QMediaDevices.videoInputs()

        self.current_camera = None

        self.camera = None

        self.capture_session = QMediaCaptureSession()

        self.video_sink = QVideoSink()

        self.capture_session.setVideoSink(
            self.video_sink
        )

        self.video_sink.videoFrameChanged.connect(
            self.process_camera_frame
        )

        if self.camera_devices:
            self.start_camera(
                self.camera_devices[0]
            )

        # DATABASE
        self.database = self.load_database()

        # AI + RENDERER
        self.recognizer = Recognizer(self.database)
        self.renderer = FrameRenderer()
        self.registration = FaceRegistration()
        self.event_recorder = EventRecorder()

        self.detections = []

        self.face_recognition_enabled = True
        self.unsaved_changes = False

        # TRACKING STATE
        self.current_faces = set()
        self.seen_frames = defaultdict(int)
        self.missing_frames = defaultdict(int)

        self.detect_threshold = 5  # The detected person must appear in at least 5 frames
        self.missing_threshold = 15  # The detected person must be gone for at least 15 frames

        # WORKER THREAD
        self.worker = AIWorker(self.recognizer)
        self.worker.result_ready.connect(self.update_detections)
        self.worker.start()

        # FRAME LOOP
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(60)

    # DATABASE
    def load_database(self):
        database = {}

        if not os.path.exists("database"):
            return database

        for file in os.listdir("database"):
            if file.endswith(".npy"):
                name = file[:-4]
                database[name] = np.load(os.path.join("database", file))

        return database

    def mark_unsaved(self):

        self.unsaved_changes = True

        self.unsaved_banner.show()

        self.apply_settings_btn.setEnabled(True)

    def clear_unsaved(self):

        self.unsaved_changes = False

        self.unsaved_banner.hide()

        self.apply_settings_btn.setEnabled(False)

    # EMPTY STATE CONTROL
    def show_events(self):
        self.empty_label.hide()
        self.events_list.show()
        self.events_container.layout().setCurrentWidget(self.events_list)

    # DETECTIONS HANDLER
    def update_detections(self, detections):
        self.detections = detections

        now = datetime.now().strftime("%b %d: %H:%M:%S")

        detected_names = set()
        unknown_count = 0

        for det in detections:
            name = det.get("name", "Unknown")

            if name == "Unknown":
                unknown_count += 1

            detected_names.add(name)

            if self.event_recorder.recording:
                self.event_recorder.unknown_count = unknown_count

            self.seen_frames[name] += 1
            self.missing_frames[name] = 0

            if name in self.current_faces:
                self.event_recorder.keep_alive()

        # Handle person's entry into frame
        for name in detected_names:
            if name not in self.current_faces:
                if self.seen_frames[name] >= self.detect_threshold:
                    self.current_faces.add(name)

                    if not self.event_recorder.recording:
                        self.event_recorder.start(
                            name,
                            self.current_frame,
                            unknown_count
                        )
                    else:
                        self.event_recorder.add_person(name)

                    if self.empty_label.isVisible():
                        self.show_events()

                    self.events_list.addItem(f"{now} - {name} was detected")
                    self.events_list.scrollToBottom()

        # Handle person's exit from frame
        for name in list(self.current_faces):
            if name not in detected_names:
                self.missing_frames[name] += 1

                if self.missing_frames[name] >= self.missing_threshold:
                    self.current_faces.remove(name)

                    self.event_recorder.stop()

                    self.events_list.addItem(f"{now} - {name} left frame")
                    self.events_list.scrollToBottom()

                    self.seen_frames[name] = 0
                    self.missing_frames[name] = 0

    # FRAME LOOP
    def update_frame(self):

        if not hasattr(self, "current_frame"):
            return

        frame = self.current_frame.copy()

        self.event_recorder.update(frame)

        # DASHBOARD MODE
        if self.pages.currentWidget() != self.add_face_widget:

            # Run recognition
            if self.face_recognition_enabled:

                if not self.worker.busy:
                    self.worker.submit(frame)

                display = self.renderer.draw(
                    frame.copy(),
                    self.detections
                )

            else:

                display = frame

            target = self.camera_label

        # ADD PERSON MODE
        else:

            # Raw camera only
            display = frame

            target = self.registration_camera

        # CONVERT FRAME TO QT IMAGE

        rgb = cv2.cvtColor(
            display,
            cv2.COLOR_BGR2RGB
        )

        h, w, ch = rgb.shape

        qt_img = QImage(
            rgb.data,
            w,
            h,
            ch * w,
            QImage.Format_RGB888
        )

        target.setPixmap(
            QPixmap.fromImage(qt_img).scaled(
                target.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

    # CLEANUP
    def closeEvent(self, event):

        msg = QMessageBox(self)

        msg.setWindowTitle(
            "Exit Sentinel"
        )

        msg.setText(
            "Are you sure you want to quit?"
        )

        msg.setInformativeText(
            "Any active recording will be stopped."
        )

        msg.setIcon(
            QMessageBox.Question
        )

        quit_btn = msg.addButton(
            "Quit",
            QMessageBox.AcceptRole
        )

        cancel_btn = msg.addButton(
            "Cancel",
            QMessageBox.RejectRole
        )

        msg.exec()

        if msg.clickedButton() != quit_btn:
            event.ignore()
            return

        self.timer.stop()

        # stop AI thread
        self.worker.stop()
        self.worker.wait()

        # finish any active recording
        self.event_recorder.shutdown()

        # release camera
        if self.camera:
            self.camera.stop()

        event.accept()

    def show_people(self):

        if self.pages.currentWidget() == self.settings_widget:
            if not self.confirm_unsaved_changes():
                return

        while self.people_layout.count():

            item = self.people_layout.takeAt(0)

            widget = item.widget()

            if widget:
                widget.deleteLater()

        for name in sorted(self.database.keys()):
            last_seen = self.get_last_seen(name)

            card = PersonCard(
                name,
                last_seen
            )

            card.remove_requested.connect(
                self.remove_person
            )

            card.view_appearances_requested.connect(
                self.show_person_appearances
            )

            card.rename_requested.connect(
                self.rename_person
            )

            self.people_layout.addWidget(
                card
            )

        self.people_layout.addStretch()

        self.pages.setCurrentWidget(
            self.people_widget
        )

    def remove_person(self, name):
        msg = QMessageBox(self)

        msg.setWindowTitle(
            "Delete Person"
        )

        msg.setText(
            f"Delete '{name}'?"
        )

        msg.setInformativeText(
            "This action cannot be undone."
        )

        msg.setIcon(
            QMessageBox.Warning
        )

        delete_btn = msg.addButton(
            "Delete",
            QMessageBox.AcceptRole
        )

        cancel_btn = msg.addButton(
            "Cancel",
            QMessageBox.RejectRole
        )

        msg.exec()

        if msg.clickedButton() != delete_btn:
            return

        path = f"database/{name}.npy"

        if os.path.exists(path):
            os.remove(path)

        if name in self.database:
            del self.database[name]

        self.show_people()

    def rename_person(self, old_name):

        new_name, ok = QInputDialog.getText(
            self,
            "Rename Person",
            "New name:",
            text=old_name
        )

        if not ok:
            return

        new_name = new_name.strip()

        if not new_name:
            return

        if new_name == old_name:
            return

        if new_name in self.database:
            QMessageBox.warning(
                self,
                "Rename Person",
                "A person with that name already exists."
            )

            return


        # Rename database file

        old_path = os.path.join(
            "database",
            f"{old_name}.npy"
        )

        new_path = os.path.join(
            "database",
            f"{new_name}.npy"
        )

        if os.path.exists(old_path):
            os.rename(
                old_path,
                new_path
            )

        # Reload recognizer database
        self.database = self.load_database()
        self.recognizer.database = self.database

        # Update event metadata
        metadata = os.path.join(
            "events",
            "events.json"
        )

        if os.path.exists(metadata):

            with open(metadata, "r") as f:
                events = json.load(f)

            for event in events:

                # Update people list
                event["people"] = [
                    new_name if person == old_name else person
                    for person in event["people"]
                ]

                # Rename video if needed
                old_file = event["file"]
                new_file = old_file.replace(
                    old_name,
                    new_name
                )

                if old_file != new_file:

                    old_video = os.path.join(
                        "events",
                        old_file
                    )

                    new_video = os.path.join(
                        "events",
                        new_file
                    )

                    if os.path.exists(old_video):
                        os.rename(
                            old_video,
                            new_video
                        )

                    # Rename thumbnail
                    old_thumb = old_video.replace(
                        ".mp4",
                        ".jpg"
                    )

                    new_thumb = new_video.replace(
                        ".mp4",
                        ".jpg"
                    )

                    if os.path.exists(old_thumb):
                        os.rename(
                            old_thumb,
                            new_thumb
                        )

                    event["file"] = new_file
                    event["thumbnail"] = new_file.replace(
                        ".mp4",
                        ".jpg"
                    )

            with open(metadata, "w") as f:
                json.dump(
                    events,
                    f,
                    indent=4
                )

        # Refresh UI
        self.load_event_filters()
        self.show_people()

    def get_last_seen(self, person):

        metadata = os.path.join(
            "events",
            "events.json"
        )

        if not os.path.exists(metadata):
            return "Never"

        with open(metadata, "r") as f:
            events = json.load(f)

        latest = None

        for event in events:

            if person in event["people"]:

                timestamp = datetime.strptime(
                    event["timestamp"],
                    "%Y%m%d_%H%M%S"
                )

                if latest is None or timestamp > latest:
                    latest = timestamp

        if latest is None:
            return "Never"

        return latest.strftime(
            "%b %d, %Y • %I:%M %p"
        )

    def confirm_unsaved_changes(self):

        if not self.unsaved_changes:
            return True

        msg = QMessageBox(self)

        msg.setWindowTitle(
            "Unsaved Changes"
        )

        msg.setText(
            "You have unsaved changes."
        )

        msg.setInformativeText(
            "Apply them before leaving?"
        )

        msg.setIcon(
            QMessageBox.Warning
        )

        save_btn = msg.addButton(
            "Save",
            QMessageBox.AcceptRole
        )

        discard_btn = msg.addButton(
            "Discard",
            QMessageBox.DestructiveRole
        )

        cancel_btn = msg.addButton(
            "Cancel",
            QMessageBox.RejectRole
        )

        msg.exec()

        clicked = msg.clickedButton()

        if clicked == save_btn:
            self.apply_settings()
            return True

        elif clicked == discard_btn:
            self.clear_unsaved()
            return True

        return False

    def show_dashboard(self):

        if (
                self.pages.currentWidget() ==
                self.settings_widget
        ):
            if not self.confirm_unsaved_changes():
                return

        self.pages.setCurrentIndex(0)

    def show_recordings(self):

        if self.pages.currentWidget() == self.settings_widget:
            if not self.confirm_unsaved_changes():
                return

        while self.recordings_layout.count():

            item = self.recordings_layout.takeAt(0)

            widget = item.widget()

            if widget:
                widget.deleteLater()

        if os.path.exists("events"):

            with open("events/events.json") as f:
                events = json.load(f)

            events = sorted(
                events,
                key=lambda x: x["timestamp"],
                reverse=True
            )

            for event in events:

                # Person filter
                selected_person = self.person_filter.currentData()

                if (
                        selected_person
                        and selected_person not in event["people"]
                ):
                    continue

                # Date filter
                timestamp = event["timestamp"]

                dt = datetime.strptime(
                    timestamp,
                    "%Y%m%d_%H%M%S"
                )

                year_text = self.year_filter.currentText()

                if (
                        year_text
                        and year_text != "Any Year"
                        and dt.year != int(year_text)
                ):
                    continue

                month_text = self.month_filter.currentText()

                if (
                        month_text
                        and month_text != "Any Month"
                        and dt.strftime("%B") != month_text
                ):
                    continue

                day_text = self.day_filter.currentText()

                if (
                        day_text
                        and day_text != "Any Day"
                        and dt.day != int(day_text)
                ):
                    continue

                card = EventCard(
                    os.path.join(
                        "events",
                        event["file"]
                    )
                )

                self.recordings_layout.addWidget(card)

        self.recordings_layout.addStretch()

        self.pages.setCurrentWidget(
            self.events_page
        )

    def show_settings(self):

        self.pages.setCurrentWidget(
            self.settings_widget
        )

    def apply_settings(self):

        new_camera = self.camera_selector.currentData()

        self.face_recognition_enabled = (
            self.face_detection_toggle.isChecked()
        )

        was_enabled = self.event_recorder.enabled

        self.event_recorder.enabled = (
            self.save_events_toggle.isChecked()
        )

        if not was_enabled and self.event_recorder.enabled and not self.event_recorder.recording:

            names = {
                det["name"]
                for det in self.detections
                if det["name"] != "Unknown"
            }

            if names:
                first = next(iter(names))

                unknown_count = sum(1 for det in self.detections if det["name"] == "Unknown")

                self.event_recorder.start(
                    first,
                    self.current_frame,
                    unknown_count
                )

                for person in names - {first}:
                    self.event_recorder.add_person(person)

        if not self.event_recorder.enabled and self.event_recorder.recording:
            self.event_recorder.finish()

        if new_camera != self.current_camera:
            self.start_camera(
                new_camera
            )

        self.clear_unsaved()

        self.settings_status.setText(
            "✓ Settings saved"
        )

        QTimer.singleShot(
            3000,
            lambda: self.settings_status.setText("")
        )

    def show_add_face(self):

        if self.pages.currentWidget() == self.settings_widget:
            if not self.confirm_unsaved_changes():
                return

        name, ok = QInputDialog.getText(
            self,
            "Register Person",
            "Enter person's name:"
        )

        if not ok:
            return

        name = name.strip()

        if not name:
            return

        self.registration_name = name

        self.registration.reset()

        self.instruction_label.setText(
            self.registration.instructions[0]
        )

        self.pages.setCurrentWidget(
            self.add_face_widget
        )

    def capture_face(self):

        # Make sure we have a camera frame
        if not hasattr(self, "current_frame"):
            return

        # Send the current frame to FaceRegistration
        finished = self.registration.capture(
            self.current_frame
        )

        # Update the instruction text
        if not finished:
            self.instruction_label.setText(
                self.registration.instructions[
                    self.registration.step
                ]
            )

            return

        # Registration complete
        self.registration.save(
            self.registration_name
        )

        QMessageBox.information(
            self,
            "Registration Complete",
            f"{self.registration_name} has been added."
        )

        # Reset registration state
        self.registration.reset()

        # Reload the face database
        self.database = self.load_database()

        # Update the recognizer
        self.recognizer.database = self.database

        # Return to People page
        self.show_people()

    def start_camera(self, device):

        if self.camera:
            self.camera.stop()

        self.camera = QCamera(device)

        # Pick a non-square format
        formats = device.videoFormats()

        best_format = None

        for fmt in formats:

            resolution = fmt.resolution()

            w = resolution.width()
            h = resolution.height()

            # Prefer 16:9 formats
            if w / h > 1.5:
                best_format = fmt
                break

        # Fallback to first format
        if best_format is None and formats:
            best_format = formats[0]

        if best_format:
            self.camera.setCameraFormat(
                best_format
            )

        self.capture_session.setCamera(
            self.camera
        )

        self.current_camera = device

        self.camera.start()

    def process_camera_frame(self, video_frame):

        if not video_frame.isValid():
            return

        image = video_frame.toImage()

        image = image.convertToFormat(
            QImage.Format.Format_RGB888
        )

        width = image.width()
        height = image.height()

        ptr = image.bits()

        arr = np.frombuffer(
            ptr,
            np.uint8
        )

        arr = arr.reshape(
            height,
            width,
            3
        )

        frame = cv2.cvtColor(
            arr,
            cv2.COLOR_RGB2BGR
        )

        self.current_frame = frame.copy()


    def get_available_cameras(self):

        return QMediaDevices.videoInputs()

    def load_event_filters(self):

        # Reset filters
        self.person_filter.clear()
        self.year_filter.clear()
        self.month_filter.clear()
        self.day_filter.clear()

        # People
        self.person_filter.addItem(
            "All People"
        )

        people = set()

        if os.path.exists("events/events.json"):

            people = set()

            for file in os.listdir("database"):
                if file.endswith(".npy"):
                    people.add(file[:-4])

            people.add("Unknown")

        for person in sorted(people):

            self.person_filter.addItem(
                person,
                person
            )

        # Years
        current_year = datetime.now().year

        self.year_filter.addItem("Any Year")

        for year in range(current_year, 1979, -1):
            self.year_filter.addItem(str(year))


        # Months
        self.month_filter.addItem("Any Month")

        months = [
            "January", "February", "March", "April",
            "May", "June", "July", "August",
            "September", "October", "November", "December"
        ]

        for month in months:
            self.month_filter.addItem(month)


        # Days
        self.update_day_filter()

    def update_day_filter(self):

        self.day_filter.clear()
        self.day_filter.addItem("Any Day")

        month_name = self.month_filter.currentText()
        year_text = self.year_filter.currentText()

        if not month_name or month_name == "Any Month":
            max_days = 31

        else:

            month_number = [
                               "January", "February", "March", "April",
                               "May", "June", "July", "August",
                               "September", "October", "November", "December"
                           ].index(month_name) + 1

            if not year_text or year_text == "Any Year":

                if month_number == 2:
                    max_days = 29
                else:
                    max_days = calendar.monthrange(2024, month_number)[1]

            else:
                year = int(year_text)
                max_days = calendar.monthrange(year, month_number)[1]

        for day in range(1, max_days + 1):
            self.day_filter.addItem(f"{day:02d}")

    def show_person_appearances(self, person):

        index = self.person_filter.findData(person)

        if index == -1:
            index = self.person_filter.findText(person)

        if index != -1:
            self.person_filter.setCurrentIndex(index)

        self.show_recordings()


# RUN
class PersonCard(QWidget):

    remove_requested = Signal(str)

    view_appearances_requested = Signal(str)

    rename_requested = Signal(str)

    def __init__(self, name, last_seen):
        super().__init__()

        self.setAttribute(
            Qt.WA_StyledBackground,
            True
        )

        self.name = name

        self.setFixedHeight(80)

        self.setObjectName("personCard")

        self.setStyleSheet("""
            #personCard {
                background:#202020;
                border-radius:15px;
            }

            QLabel {
                color:white;
                background:transparent;
            }

            QPushButton {
                border:none;
                color:white;
                font-size:24px;
                background:transparent;
            }

            QPushButton:hover {
                background:#333333;
                border-radius:10px;
            }
        """)

        layout = QHBoxLayout()

        self.setLayout(layout)

        # Profile circle
        avatar = QLabel()

        avatar.setFixedSize(
            50,
            50
        )

        avatar.setStyleSheet("""
            background:white;
            border-radius:25px;
        """)

        # Name
        info_layout = QVBoxLayout()

        name_label = QLabel(
            name
        )

        name_label.setStyleSheet("""
            font-size:20px;
            font-weight:bold;
        """)

        last_seen_label = QLabel(
            f"Last seen: {last_seen}"
        )

        last_seen_label.setStyleSheet("""
            color:gray;
            font-size:13px;
        """)

        info_layout.addWidget(
            name_label
        )

        info_layout.addWidget(
            last_seen_label
        )

        # Three dots button
        menu_btn = QPushButton(
            "⋮"
        )

        menu_btn.clicked.connect(
            self.show_menu
        )


        layout.addWidget(
            avatar
        )

        layout.addSpacing(
            15
        )

        layout.addLayout(
            info_layout
        )

        layout.addStretch()

        layout.addWidget(
            menu_btn
        )

    def show_menu(self):

        menu = QMenu(
            self
        )

        appearances = menu.addAction(
            "View past appearances"
        )

        rename = menu.addAction(
            "Rename person"
        )

        remove = menu.addAction(
            "Remove person"
        )

        action = menu.exec(
            self.mapToGlobal(
                QPoint(
                    self.width()-30,
                    50
                )
            )
        )

        if action == appearances:
            self.view_appearances_requested.emit(self.name)

        elif action == rename:
            self.rename_requested.emit(self.name)

        elif action == remove:
            self.remove_requested.emit(self.name)


class EventCard(QWidget):

    def __init__(self, path):
        super().__init__()

        self.setAttribute(
            Qt.WA_StyledBackground,
            True
        )

        self.path = path

        self.setFixedHeight(80)

        self.setObjectName("eventCard")

        self.setStyleSheet("""
            #eventCard {
                background:#202020;
                border-radius:15px;
            }

            QLabel {
                color:white;
                background:transparent;
                font-size:16px;
            }

            QPushButton {
                background:transparent;
                color:white;
                padding:6px;
                border:none;
            }

            QPushButton:hover {
                background:#333333;
                border-radius:10px;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setAlignment(Qt.AlignVCenter)

        layout.setContentsMargins(
            12, 6, 10, 6
        )

        # Load video thumbnails
        thumbnail = QLabel()
        thumbnail.setFixedSize(120, 68)
        thumbnail.setAlignment(Qt.AlignCenter)

        thumbnail_path = path.replace(".mp4", ".jpg")

        if os.path.exists(thumbnail_path):
            thumbnail.setPixmap(
                QPixmap(thumbnail_path).scaled(
                    thumbnail.size(),
                    Qt.KeepAspectRatioByExpanding,
                    Qt.SmoothTransformation
                )
            )

        filename = os.path.splitext(
            os.path.basename(path)
        )[0]

        parts = filename.split("_")

        timestamp = parts[0] + "_" + parts[1]

        raw_names = parts[2:]

        known_people = []
        unknown_count = 0

        for name in raw_names:
            if name.startswith("Unknown"):
                unknown_count += 1
            else:
                known_people.append(name)

        display_names = known_people.copy()

        if unknown_count == 1:
            display_names.append("1 unknown person")

        elif unknown_count > 1:
            display_names.append(
                f"{unknown_count} unknown people"
            )

        if len(display_names) == 1:
            names = display_names[0]

        elif len(display_names) == 2:
            names = f"{display_names[0]} and {display_names[1]}"

        else:
            names = ", ".join(display_names[:-1])
            names += f", and {display_names[-1]}"

        dt = datetime.strptime(
            timestamp,
            "%Y%m%d_%H%M%S"
        )

        pretty_time = dt.strftime(
            "%b %d, %Y • %I:%M %p"
        )

        name_label = QLabel(names)

        name_label.setStyleSheet("""
            font-size:18px;
            font-weight:bold;
        """)

        time_label = QLabel(pretty_time)

        time_label.setStyleSheet("""
            color:gray;
            font-size:12px;
        """)

        info_layout = QVBoxLayout()

        info_layout.addWidget(name_label)
        info_layout.addWidget(time_label)

        play_btn = QPushButton("Play")
        delete_btn = QPushButton("Delete")

        play_btn.clicked.connect(self.play)
        delete_btn.clicked.connect(self.delete)

        layout.addWidget(thumbnail, alignment=Qt.AlignVCenter)

        layout.addSpacing(15)

        layout.addLayout(info_layout)

        layout.addStretch()

        layout.addWidget(play_btn)
        layout.addWidget(delete_btn)

    def play(self):

        system = platform.system()

        if system == "Windows":
            os.startfile(self.path)

        elif system == "Darwin":
            subprocess.call(["open", self.path])

        else:
            subprocess.call(["xdg-open", self.path])

    def delete(self):
        msg = QMessageBox(self)

        msg.setWindowTitle(
            "Delete Recording"
        )

        msg.setText(
            "Are you sure you want to delete this recording?"
        )

        msg.setInformativeText(
            "This action cannot be undone."
        )

        msg.setIcon(
            QMessageBox.Warning
        )

        delete_btn = msg.addButton(
            "Delete",
            QMessageBox.AcceptRole
        )

        cancel_btn = msg.addButton(
            "Cancel",
            QMessageBox.RejectRole
        )

        msg.exec()

        if msg.clickedButton() != delete_btn:
            return

        # Delete video file
        if os.path.exists(self.path):
            os.remove(self.path)

        # Delete thumbnail if it exists
        thumbnail_path = self.path.replace(".mp4", ".jpg")

        if os.path.exists(thumbnail_path):
            os.remove(thumbnail_path)

        # Remove from JSON
        json_path = "events/events.json"

        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                events = json.load(f)

            filename = os.path.basename(self.path)

            events = [
                event for event in events
                if event["file"] != filename
            ]

            with open(json_path, "w") as f:
                json.dump(
                    events,
                    f,
                    indent=4
                )

        # Remove UI card
        self.setParent(None)
        self.deleteLater()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
