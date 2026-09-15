import os
import cv2
import json
from collections import deque
from datetime import datetime


class EventRecorder:

    def __init__(self, fps=15, pre_seconds=5, post_seconds=3, output_folder="events"):

        self.fps = fps
        self.pre_frames = fps * pre_seconds
        self.post_frames = fps * post_seconds

        self.buffer = deque(maxlen=self.pre_frames)

        self.recording = False
        self.enabled = True
        self.post_counter = 0

        self.writer = None
        self.current_names = set()
        self.unknown_count = 0

        self.current_path = None
        self.current_timestamp = None
        self.thumbnail_frame = None

        os.makedirs(output_folder, exist_ok=True)
        self.output_folder = output_folder

        self.metadata_file = os.path.join(
            output_folder,
            "events.json"
        )

        if not os.path.exists(self.metadata_file):
            with open(self.metadata_file, "w") as f:
                json.dump([], f, indent=4)

    # Feed every camera frame into update
    def update(self, frame):

        self.buffer.append(frame.copy())

        if self.recording:

            self.writer.write(frame)

            if self.post_counter > 0:

                self.post_counter -= 1

                if self.post_counter == 0:
                    self.finish()

    # Run when a person appears in frame
    def start(self, person_name, frame, unknown_count):

        if not self.enabled:
            return

        if self.recording:
            return

        h, w = frame.shape[:2]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Add first detected person
        self.current_names.add(person_name)

        self.current_timestamp = timestamp

        self.unknown_count = unknown_count

        # Temporary filename
        filename = f"{timestamp}_recording.mp4"

        self.thumbnail_frame = frame.copy()

        path = os.path.join(
            self.output_folder,
            filename
        )

        self.current_path = path

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        self.writer = cv2.VideoWriter(
            path,
            fourcc,
            self.fps,
            (w, h)
        )

        self.recording = True
        self.current_names.add(person_name)
        self.post_counter = 0

        # Save buffered frames first
        for buffered_frame in self.buffer:
            self.writer.write(buffered_frame)

        print("Started recording:", filename)

    # Person still visible
    def keep_alive(self):

        if self.recording:
            self.post_counter = 0

    def add_person(self, person_name):

        if self.recording:
            self.current_names.add(person_name)

    # Person disappeared
    def stop(self):

        if self.recording:
            self.post_counter = self.post_frames

    # Finish recording
    def finish(self):

        if self.writer is not None:
            self.writer.release()

        # Rename file with all detected people
        if self.current_path is not None:

            names_list = sorted(self.current_names)

            for _ in range(self.unknown_count):
                names_list.append("Unknown")

            if not names_list:
                names_list = ["Unknown"]

            names = "_".join(names_list)

            new_filename = f"{self.current_timestamp}_{names}.mp4"

            new_path = os.path.join(
                self.output_folder,
                new_filename
            )

            thumbnail_name = new_filename.replace(
                ".mp4",
                ".jpg"
            )

            thumbnail_path = os.path.join(
                self.output_folder,
                thumbnail_name
            )

            os.rename(self.current_path, new_path)

            cv2.imwrite(
                thumbnail_path,
                self.thumbnail_frame
            )

            people = list(self.current_names)

            for _ in range(self.unknown_count):
                people.append("Unknown")

            if not people:
                people = ["Unknown"]

            event_data = {

                "file": new_filename,

                "thumbnail": thumbnail_name,

                "people": people,

                "timestamp": self.current_timestamp,

                "created": datetime.now().isoformat()

            }

            self.save_metadata(event_data)

        self.writer = None
        self.current_path = None
        self.current_timestamp = None
        self.recording = False
        self.post_counter = 0
        self.current_names.clear()
        self.unknown_count = 0

    def shutdown(self):

        if self.recording:

            self.finish()

        else:

            if self.writer is not None:
                self.writer.release()
                self.writer = None

        self.recording = False
        self.post_counter = 0

    def save_metadata(self, data):

        with open(self.metadata_file, "r") as f:
            events = json.load(f)

        events.append(data)

        with open(self.metadata_file, "w") as f:
            json.dump(events, f, indent=4)