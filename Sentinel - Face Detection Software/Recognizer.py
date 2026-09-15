import numpy as np
from insightface.app import FaceAnalysis


class Recognizer:

    def __init__(self, database):

        self.database = database

        self.app = FaceAnalysis(
                        allowed_modules=[
                            "detection",
                            "recognition"
                        ]
                    )

        self.app.prepare(ctx_id=0, det_size=(224, 224))

    # Embedding matching only
    def recognize_embedding(self, embedding, threshold=25):

        best_name = "Unknown"
        best_dist = float("inf")

        for name, embeddings in self.database.items():

            distances = np.linalg.norm(
                embeddings - embedding,
                axis=1
            )

            dist = np.min(distances)

            if dist < best_dist:
                best_dist = dist
                best_name = name

        if best_dist > threshold:
            return "Unknown", best_dist

        return best_name, best_dist

    # Detect a person's face using AI FaceAnalysis and return raw detections
    def detect(self, frame_small, scale_x, scale_y):

        faces = self.app.get(frame_small)

        detections = []

        for face in faces:

            x1, y1, x2, y2 = face.bbox.astype(int)

            # scale back
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            name, dist = self.recognize_embedding(face.embedding)

            detections.append({
                "name": name,
                "bbox": (x1, y1, x2, y2),
                "distance": dist
            })

        return detections
