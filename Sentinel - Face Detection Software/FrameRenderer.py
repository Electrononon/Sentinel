import cv2


class FrameRenderer:

    def draw(self, frame, detections):

        for d in detections:

            x1, y1, x2, y2 = d["bbox"]

            color = (0, 255, 0) if d["name"] != "Unknown" else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(
                frame,
                d["name"],
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        return frame