import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from mediapipe_utils import PrepInputs, get_unified_model

# -----------------------------------------------------
# Defina o mapa de índices igual ao usado na criação
# do modelo (mesmo formato de frame_map)
# -----------------------------------------------------
frame_map = {
    "face":  list(range(0, 468)),
    "left_hand":  list(range(468, 489)),
    "pose":  list(range(489, 522)),
    "right_hand": list(range(522, 543)),
}

# -----------------------------------------------------
# 1) Monte seu modelo e carregue pesos
# -----------------------------------------------------
n_labels = 250
model = get_unified_model(frame_map, n_labels=n_labels)
model.load_weights("../test/models/asl_model.keras")  # .save_weights depois de treinar

# -----------------------------------------------------
# 2) Inicie o Holistic do MediaPipe
# -----------------------------------------------------
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------------------------------
# 3) Inicie a captura em background
# -----------------------------------------------------
class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.grabbed, self.frame = self.cap.read()
        self.stopped = False
        import threading
        threading.Thread(target=self.update, daemon=True).start()
    def update(self):
        while not self.stopped:
            self.grabbed, self.frame = self.cap.read()
    def read(self):
        return self.grabbed, self.frame
    def stop(self):
        self.stopped = True
        self.cap.release()

vs = VideoStream(0)

# -----------------------------------------------------
# 4) Loop de inferência
# -----------------------------------------------------
prep = PrepInputs()
try:
    while True:
        ret, frame = vs.read()
        if not ret:
            break

        # a) converte e processa landmarks
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(img_rgb)

        # b) concatena todos os landmarks em um array (543×3)
        #    caso não detecte alguma parte, preencha com zeros
        def to_array(landmarks, n_pts):
            if landmarks:
                pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
            else:
                pts = np.zeros((n_pts, 3), dtype=np.float32)
            return pts

        face = to_array(results.face_landmarks, 468)
        lh   = to_array(results.left_hand_landmarks, 21)
        pose = to_array(results.pose_landmarks, 33)
        rh   = to_array(results.right_hand_landmarks, 21)

        all_pts = np.vstack([face, lh, pose, rh])[None, ...]  # shape (1,543,3)

        # c) pré-processa e infere
        x_feat = prep(tf.constant(all_pts))          # tf.Tensor (1,feat_len)
        probs = model(x_feat, training=False).numpy()  # (1, n_labels)
        pred = np.argmax(probs[0])

        # d) desenha o label na tela
        label = f"Pred: {pred}"
        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # e) exibe
        cv2.imshow("Libras Live", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    vs.stop()
    cv2.destroyAllWindows()
