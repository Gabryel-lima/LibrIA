import cv2
import mediapipe as mp
import math

# --- Função utilitária de depuração de landmarks ---
def debug_landmarks(frame, hand_landmarks, threshold_overlap=20):
    h, w, _ = frame.shape
    # Mapeia cada landmark a coordenadas em pixels
    coords = {}
    for idx, lm in enumerate(hand_landmarks.landmark):
        px, py = int(lm.x * w), int(lm.y * h)
        pz = lm.z  # profundidade normalizada
        coords[idx] = (px, py, pz)
        # Desenha o índice e a profundidade ao lado do ponto
        cv2.circle(frame, (px, py), 4, (0,255,0), -1)
        cv2.putText(frame, f'{idx}:{pz:.2f}', (px+5, py-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1,
                    cv2.LINE_AA)

    # Exemplo: distância entre ponta do polegar (4) e ponta do indicador (8)
    if 4 in coords and 8 in coords:
        x1, y1, z1 = coords[4]
        x2, y2, z2 = coords[8]
        dist = math.hypot(x2-x1, y2-y1)
        # desenha a linha e o valor
        cv2.line(frame, (x1, y1), (x2, y2), (255,0,0), 2)
        cv2.putText(frame, f'Dist: {dist:.1f}px', 
                    ((x1+x2)//2, (y1+y2)//2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2,
                    cv2.LINE_AA)
        # sinaliza sobreposição se muito próximos
        if dist < threshold_overlap:
            cv2.putText(frame, f'Overlap!', 
                        (x1, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2,
                        cv2.LINE_AA)

    # Avalia proximidade geral à câmera por um ponto de referência (por ex., punho: 0)
    if 0 in coords:
        _, _, z0 = coords[0]
        # z é negativo quanto mais acima do plano da câmera,
        # então valores com maior módulo indicam mais proximidade.
        prox_text = f'Z0: {z0:.3f}'
        cv2.putText(frame, prox_text, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2,
                    cv2.LINE_AA)

    return frame

def cam(device_index=2):
    cap = cv2.VideoCapture(device_index) # Verifique o valor correto do droidcam
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir a câmera no índice {device_index}")
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False, 
        max_num_hands=2, 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )

    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2, color=(0, 0, 255))

    try:
        while True:
            # Capture video from the camera
            ret, frame = cap.read()

            # Convert the BGR image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the image and find hands
            results = hands.process(image)

            # Draw hand landmarks on the image
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        drawing_spec, drawing_spec
                    )
                    
                    # Chama nossa rotina de debug
                    frame = debug_landmarks(frame, hand_landmarks,
                                            threshold_overlap=30)

            # Display the image with hand landmarks
            cv2.imshow('Hand Tracking', frame)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Release the camera and close all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

if __name__ == "__main__":
    cam()
