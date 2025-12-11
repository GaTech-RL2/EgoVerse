import cv2
import numpy as np
import pupil_apriltags as apriltag


class AprilTagDetector:
    def __init__(self, families="tag36h11"):

        self.detector = apriltag.Detector(
            families=families, quad_decimate=3.0, decode_sharpening=0.25
        )
        self.results = []

    def detect(self, img: np.ndarray, intrinsics: dict, tag_size: int):
        img = img.astype(np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.results = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=[intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]],
            tag_size=tag_size,
        )
        return self.results

    def __len__(self):
        return len(self.results)

    def vis_tag(self, img: np.ndarray):
        img = img.astype(np.uint8)
        for r in self.results:
            (ptA, ptB, ptC, ptD) = r.corners
            ptB = (int(ptB[0]), int(ptB[1]))
            ptC = (int(ptC[0]), int(ptC[1]))
            ptD = (int(ptD[0]), int(ptD[1]))
            ptA = (int(ptA[0]), int(ptA[1]))

            # draw the bounding box of the AprilTag detection
            cv2.line(img, ptA, ptB, (0, 255, 0), 2)
            cv2.line(img, ptB, ptC, (0, 255, 0), 2)
            cv2.line(img, ptC, ptD, (0, 255, 0), 2)
            cv2.line(img, ptD, ptA, (0, 255, 0), 2)
            # draw the center (x, y)-coordinates of the AprilTag
            (cX, cY) = (int(r.center[0]), int(r.center[1]))
            cv2.circle(img, (cX, cY), 5, (0, 0, 255), -1)
            # draw the tag family on the image
            tagFamily = r.tag_family.decode("utf-8")
            cv2.putText(
                img,
                f"{tagFamily}-{r.tag_id}-{cX}-{cY}",
                (ptA[0], ptA[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
        return img

    def tags_centroid(self):
        """Return a dictionary, keys being tag id and values being positions of the centroid"""
        centroid_dict = {}
        for r in self.results:
            centroid_dict[r.tag_id] = [int(r.center[0]), int(r.center[1])]
        return centroid_dict
