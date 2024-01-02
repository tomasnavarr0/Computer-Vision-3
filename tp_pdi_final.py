import cv2
import numpy as np

def filter_color_hsv(image, color_lower1, color_upper1, color_lower2, color_upper2):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, color_lower1, color_upper1)
    mask2 = cv2.inRange(hsv, color_lower2, color_upper2)
    mask = cv2.bitwise_or(mask1, mask2)
    return cv2.bitwise_and(image, image, mask=mask)

video_path = "tirada_1.mp4"

cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter('dados_quietos.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

previous_list_centroids = []

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break

    frame = cv2.resize(frame, dsize=(int(frame.shape[1] / 3), int(frame.shape[0] / 3)))

    lower_red1 = np.array([0, 100, 50])
    upper_red1 = np.array([5, 255, 255])
    lower_red2 = np.array([170, 100, 50])
    upper_red2 = np.array([190, 255, 255])
    red_mask = filter_color_hsv(frame, lower_red1, upper_red1, lower_red2, upper_red2)
    red_mask_gray = cv2.cvtColor(red_mask, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('', red_mask_gray)

    # Filtrar por área y obtener centroides
    labels, _, stats, centroids = cv2.connectedComponentsWithStats(red_mask_gray)

    # Crear la máscara una sola vez fuera del bucle
    dice_mask = np.zeros_like(red_mask_gray)

    # Visualizar objetos encontrados
    for stat in stats[1:]:  # Excluir el fondo (índice 0)
        x, y, w, h, area = map(int, stat)

        # Calcular el aspect ratio del rectángulo
        aspect_ratio = w / float(h)

        # Filtrar por área y aspect ratio
        if 200 < area < 600 and 0.5 < aspect_ratio < 1.5:
            # Acumular la región en la máscara
            dice_mask[y:y+h, x:x+w] += red_mask_gray[y:y+h, x:x+w]

    # Aplicar cierre morfológico sin dilatación
    kernel_cierre = np.ones((1, 1), np.uint8)  # Ajusta el tamaño del kernel
    dice_mask = cv2.morphologyEx(dice_mask, cv2.MORPH_CLOSE, kernel_cierre, iterations=1)  # Ajusta el número de iteraciones
    #cv2.imshow('', dice_mask)

    # Umbralizar la máscara
    _, binary_mask = cv2.threshold(dice_mask, 20, 255, cv2.THRESH_BINARY)
    #cv2.imshow('Binary mask', binary_mask)
    
    # Encontrar contornos en la máscara binaria
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crear una lista vacía para guardar los contornos válidos
    valid_contours = []

    # Crear una lista vacía para guardar los centroides de cada dado
    list_centroids = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Filtrar contornos por área
        if cv2.contourArea(contour) > 50:
            valid_contours.append(contour)

            # Calcular los centroides
            centroid_x = x + w // 2
            centroid_y = y + h // 2
            dice_centroid = (centroid_x, centroid_y)

            # Agregar el centroide a la lista
            list_centroids.append(dice_centroid)
    
    umbral = 3
    if all(all(abs(x[i] - y[i]) <= umbral for i in range(len(x))) for x, y in zip(list_centroids, previous_list_centroids)):
        if len(list_centroids) == 5:
            # Visualizar y contar puntos en los dados
            for contour in valid_contours:
                points_inside_dice = 0
                x, y, w, h = cv2.boundingRect(contour)
                             
                dice_region = frame[y:y+h, x:x+w]

                # Convertir la región del dado a escala de grises
                dice_gray = cv2.cvtColor(dice_region, cv2.COLOR_BGR2GRAY)
                #cv2.imshow("",dice_gray)

                # Umbralizar la región del dado
                _, binary_dice = cv2.threshold(dice_gray, 175, 255, cv2.THRESH_BINARY)
                
                # Encontrar contornos en la región binarizada
                contours_inside_dice, _ = cv2.findContours(binary_dice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Filtrar los puntos dentro del dado por área y contarlos
                for dice_contour in contours_inside_dice:
                    if 0 < cv2.contourArea(dice_contour) < 15:
                            points_inside_dice += 1

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f'{points_inside_dice}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    
            cv2.imwrite("dados_quietos.png", frame)
            nuevas_dimensiones = (width, height)
            resize_frame = cv2.resize(frame, nuevas_dimensiones)
            out.write(resize_frame)

    previous_list_centroids = list_centroids
    
    cv2.imshow('Frame', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()