from insightface.app import FaceAnalysis
import numpy as np
import os
import cv2


class FaceRegistration:

    def __init__(self):

        # Instructions for user to follow when registering a face
        self.instructions = [
            "Look straight",
            "Smile",
            "Turn slightly left",
            "Turn further left",
            "Turn slightly right",
            "Turn further right",
            "Look slightly up",
            "Look slightly down",
            "Move closer",
            "Move farther away",
        ]

        self.step = 0
        self.embeddings = []

        self.face = None

        self.app = FaceAnalysis()
        self.app.prepare(ctx_id=0)

    def process_frame(self, frame):
        faces = self.app.get(frame)

        self.face = None

        display = frame.copy()

        if len(faces) == 1:
            self.face = faces[0]

            x1, y1, x2, y2 = self.face.bbox.astype(int)

            cv2.rectangle(display,
                          (x1, y1),
                          (x2, y2),
                          (0, 255, 0),
                          2)

        return display

    def capture(self, frame):

        faces = self.app.get(frame)

        if len(faces) != 1:
            return False

        self.embeddings.append(
            faces[0].embedding
        )

        self.step += 1

        return self.step >= len(self.instructions)

    def save(self, name):

        embeddings = np.array(self.embeddings)

        os.makedirs("database", exist_ok=True)

        np.save(
            os.path.join("database", f"{name}.npy"),
            embeddings
        )

        self.reset()

    def reset(self):

        self.step = 0
        self.embeddings.clear()
        self.face = None

    def current_instruction(self):

        if self.step >= len(self.instructions):
            return "Finished"

        return self.instructions[self.step]