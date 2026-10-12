# Copyright (C) Joe Cincotta
#
# SPDX-License-Identifier: MIT
import cv2
import numpy as np
import onnxruntime as ort


def _box_to_xywh(box):
    x1, y1, x2, y2 = (float(v) for v in box)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1


def _xywh_to_box(cx, cy, w, h):
    return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]


class KalmanFilter:
    """Compact SORT-style linear Kalman filter for a bounding-box track.
    based on concepts from https://github.com/abewley/sort and the paper doi 10.1109/ICIP.2016.7533003
    
    State vector is ``[cx, cy, ar, h, vx, vy, var, vh]``; the measurement is the
    observed ``[cx, cy, ar, h]``. Velocities are inferred, which lets the
    prediction bridge short occlusions between detections.
    """

    dt = 1.0 # one step per frame

    def __init__(self) -> None:
        self.x = np.zeros(8, dtype=np.float64)
        self.P = np.eye(8, dtype=np.float64) * 10.0 # a few pixels of uncertainty (heuristic based on SORT)
        self.Q = np.eye(8, dtype=np.float64) * 1e-2 # tiny per-frame jitter SORT implementation's default
        self.R = np.eye(4, dtype=np.float64) * 0.5 # detector localizes within roughly a sub-pixel to a pixel (heuristic, smaller than SORT)

        self.F = np.array(
            [
                [1, 0, 0, 0, self.dt, 0, 0, 0],
                [0, 1, 0, 0, 0, self.dt, 0, 0],
                [0, 0, 1, 0, 0, 0, self.dt, 0],
                [0, 0, 0, 1, 0, 0, 0, self.dt],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        self.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ],
            dtype=np.float64,
        )

    def initialize_from_box(self, box) -> None:
        cx, cy, w, h = _box_to_xywh(box)
        ar = w / h if h > 0 else 0.0
        self.x = np.array([cx, cy, ar, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.P = np.eye(8, dtype=np.float64) * 10.0

    def predict(self) -> None:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def predicted_box(self) -> list:
        cx, cy, ar, h = (float(v) for v in self.x[:4])
        ar, h = max(ar, 1e-3), max(h, 1e-3)  # guard against degenerate state
        return [max(0.0, v) for v in _xywh_to_box(cx, cy, ar * h, h)]

    def update(self, box) -> None:
        cx, cy, w, h = _box_to_xywh(box)
        ar = w / h if h > 0 else 0.0
        z = np.array([cx, cy, ar, h], dtype=np.float64)

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P

    def to_state(self) -> dict:
        return {"x": self.x.tolist(), "P": self.P.tolist()}

    @classmethod
    def from_state(cls, state: dict) -> "KalmanFilter":
        kf = cls()
        x = np.asarray(state["x"], dtype=np.float64).ravel()
        P = np.asarray(state["P"], dtype=np.float64)
        if x.shape != (8,) or P.shape != (8, 8):
            raise ValueError("invalid Kalman state")
        kf.x, kf.P = x, P
        return kf


class Track:
    """A single tracked object."""

    def __init__(self, kf: KalmanFilter, track_id: int, label: str = "unknown", score: float = 0.0):
        self.kf = kf
        self.track_id = track_id
        self.label = label
        self.score = float(score)
        self.age = 0
        self.hits = 0
        self.misses = 0

    @property
    def box(self) -> list:
        return self.kf.predicted_box()

    def to_state(self) -> dict:
        return {
            "id": int(self.track_id),
            "label": str(self.label),
            "score": float(self.score),
            "age": int(self.age),
            "hits": int(self.hits),
            "misses": int(self.misses),
            **self.kf.to_state(),
        }

    @classmethod
    def from_state(cls, st: dict) -> "Track":
        track = cls(KalmanFilter.from_state(st), 
                    int(st["id"]),
                    label=str(st.get("label", "unknown")), 
                    score=float(st.get("score", 0.0))
                    )
        track.age = int(st.get("age", 0))
        track.hits = int(st.get("hits", 0))
        track.misses = int(st.get("misses", 0))
        return track


class SortTracker:
    """Detection-driven multi-object tracker with stable integer IDs.
    ``update`` is stateless across frames in the sense that *everything* needed
    to continue tracking is returned in the serialised state dict; the caller
    passes it back on the next frame.
    """

    def __init__(self, max_misses: int = 5, iou_threshold: float = 0.3):
        self.max_misses = max_misses
        self.iou_threshold = iou_threshold
        self._next_id = 0

    def _alloc_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid

    @staticmethod
    def _iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        ih = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter = iw * ih
        union = (max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
                 + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter)
        return inter / union if union > 0 else 0.0

    def _associate(self, track_labels, det_labels, boxes, dets, iou_threshold):
        """Greedy, class-aware IoU association.

        Cross-class IoU is forced to 0 so a detection can only match a track
        of the same label.
        """
        if len(boxes) == 0 or len(dets) == 0:
            return [], [], list(range(len(boxes)))

        ious = np.zeros((len(boxes), len(dets)), dtype=np.float64)
        for i in range(len(boxes)):
            for j in range(len(dets)):
                if track_labels[i] == det_labels[j]:
                    ious[i, j] = self._iou(boxes[i], dets[j])

        pairs = sorted(
            ((ious[i, j], i, j) for i in range(ious.shape[0]) for j in range(ious.shape[1])),
            reverse=True,
        )
        used_track, used_det, matches = [False] * len(boxes), [False] * len(dets), []
        for score, i, j in pairs:
            if score < iou_threshold:
                break
            if used_track[i] or used_det[j]:
                continue
            used_track[i] = used_det[j] = True
            matches.append((i, j))

        unmatched_track = [i for i in range(len(boxes)) if not used_track[i]]
        matched_det = [j for j in range(len(dets)) if used_det[j]]
        return matches, matched_det, unmatched_track

    @staticmethod
    def _restore(state):
        if not state:
            return [], 0
        next_id = 0
        if isinstance(state, dict):
            next_id = state["next_id"]

        tracks = []
        for entry in state["tracks"]:
            if not isinstance(entry, dict):
                continue
            try:
                tracks.append(Track.from_state(entry))
            except Exception:
                continue
        return tracks, next_id

    def update(self, detections, state=None):
        tracks, next_id = self._restore(state)
        self._next_id = max(next_id, max((t.track_id for t in tracks), default=-1)) + 1

        dets = []
        for d in detections or []:
            if isinstance(d, dict):
                dets.append({"box": [float(v) for v in d["box"]],
                            "label": str(d.get("label", "unknown")),
                            "score": float(d.get("score", 0.0))})
            else:
                dets.append({"box": [float(v) for v in d], "label": "unknown", "score": 0.0})

        if not tracks:
            # First frame (or all tracks expired): seed one track per detection.
            for det in dets:
                kf = KalmanFilter()
                kf.initialize_from_box(det["box"])
                tracks.append(Track(kf, self._alloc_id(), det["label"], det["score"]))
        else:
            for t in tracks:
                t.age += 1
                t.kf.predict()

            predicted = np.array([t.kf.predicted_box() for t in tracks], dtype=np.float64)
            det_boxes = np.array([d["box"] for d in dets], dtype=np.float64) if dets else np.zeros((0, 4))

            matches, matched_dets, unmatched = self._associate(
                [t.label for t in tracks], [d["label"] for d in dets],
                predicted, det_boxes, self.iou_threshold)

            for t_pos, d_pos in matches:
                t, det = tracks[t_pos], dets[d_pos]
                t.kf.update(det["box"])
                t.hits += 1
                t.misses = 0
                t.label = det["label"]  # follow the detector's current call
                t.score = det["score"]

            # Fresh tracks from detections that matched nothing.
            for d_pos, det in enumerate(dets):
                if d_pos in matched_dets:
                    continue
                kf = KalmanFilter()
                kf.initialize_from_box(det["box"])
                tracks.append(Track(kf, self._alloc_id(), det["label"], det["score"]))

            # Age stale tracks and drop the ones unobserved too long.
            stale = {tracks[i].track_id for i in unmatched}
            kept = []
            for t in tracks:
                if t.track_id in stale:
                    t.misses += 1
                if t.misses <= self.max_misses:
                    kept.append(t)
            tracks = kept
        results = [[round(v) for v in t.box] for t in tracks]
        new_state = {"next_id": int(self._next_id), "tracks": [t.to_state() for t in tracks]}
        return results, new_state


class ModelHandler:
    def __init__(self, labels, max_misses: int = 5, iou_threshold: float = 0.3):
        self.model = None
        self.load_network(model="yolov7-w6-nms-1280.onnx")
        self.labels = labels
        self.tracker = SortTracker(max_misses=max_misses, iou_threshold=iou_threshold)

    def load_network(self, model):
        device = ort.get_device()
        cuda = True if device == "GPU" else False
        try:
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if cuda
                else ["CPUExecutionProvider"]
            )
            so = ort.SessionOptions()
            so.log_severity_level = 3

            self.model = ort.InferenceSession(model, providers=providers, sess_options=so)
            self.output_details = [i.name for i in self.model.get_outputs()]
            self.input_details = [i.name for i in self.model.get_inputs()]
        except Exception as e:
            raise Exception(f"Cannot load model {model}: {e}")

    def letterbox(
        self, im, new_shape=(1280, 1280), color=(114, 114, 114), auto=True, scaleup=True, stride=32
    ):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(
            im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )  # add border
        return im, r, (dw, dh)

    def _infer(self, inputs: np.ndarray):
        img = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB)
        image = img.copy()
        image, ratio, dwdh = self.letterbox(image, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)

        im = image.astype(np.float32)
        im /= 255

        inp = {self.input_details[0]: im}
        # ONNX inference
        output = list()
        detections = self.model.run(self.output_details, inp)[0]

        # for det in detections:
        boxes = detections[:, 1:5]
        labels = detections[:, 5]
        scores = detections[:, -1]

        boxes -= np.array(dwdh * 2)
        boxes /= ratio
        boxes = boxes.round().astype(np.int32)
        output.append(boxes)
        output.append(labels)
        output.append(scores)
        return output

    def infer(self, image, threshold, shapes=None, states=None):
        image = np.array(image)
        image = image[:, :, ::-1].copy()
        h, w, _ = image.shape
        detections = self._infer(image)
        dets = []
        if detections:
            boxes = detections[0]
            labels = detections[1]
            scores = detections[2]

            for label, score, box in zip(labels, scores, boxes):
                if score >= threshold:
                    xtl = max(int(box[0]), 0)
                    ytl = max(int(box[1]), 0)
                    xbr = min(int(box[2]), w)
                    ybr = min(int(box[3]), h)
                    if xbr > xtl and ybr > ytl:
                        dets.append(
                            {
                                "box": [xtl, ytl, xbr, ybr],
                                "label": self.labels.get(int(label), "unknown"),
                                "score": float(score),
                            }
                        )

        old_state = None
        if isinstance(states, list) and len(states) > 0:
            old_state = states[0]
        results, new_state = self.tracker.update(dets, old_state)
        return {"shapes": results, "states": [new_state]}
