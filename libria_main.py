import cv2 as cv
import numpy as np

# Conectar o DroidCam e o celular

# Carregar os nomes das classes
with open("", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Configurações do YOLO
net = cv.dnn.readNet(model="", config="")
layer_names = net.getLayerNames()
output_layers = [layer_names[i- 1] for i in net.getUnconnectedOutLayers()]

# Capturar vídeo da câmera
cap = cv.VideoCapture(0)

# Verificar se a captura de vídeo foi inicializada com sucesso
if not cap.isOpened():
    print("Erro ao abrir a câmera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Não foi possível receber frame (stream end?). Saindo ...")
        break

    # Redimensionar para melhorar a performance e converter para blob
    blob = cv.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Inicializar listas de detecção
    class_ids = []
    confidences = []
    boxes = []

    # Analisar as saídas
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Coordenadas do centro e dimensões da caixa delimitadora
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                
                # Coordenadas da caixa delimitadora
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Aplicar Non-Max Suppression para evitar múltiplas caixas para o mesmo objeto
    indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Desenhar as caixas delimitadoras e os rótulos na imagem
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)  # Cor verde para a caixa
            cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv.putText(frame, f'{label} {confidence:.2f}', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Mostrar o frame resultante
    cv.imshow("YOLO Object Detection", frame)

    # Pressione 'q' no teclado para sair do loop
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a captura de vídeo e fechar janelas
cap.release()
cv.destroyAllWindows()
