import cv2
import numpy as np
from PIL import Image
import os

face_net = cv2.dnn.readNetFromCaffe('Sticker_on_Face/model/deploy.prototxt', 'Sticker_on_Face/model/res10_300x300_ssd_iter_140000.caffemodel')
directory_hat = 'Sticker_on_Face/imgs/hat_img'
directory_glass = 'Sticker_on_Face/imgs/glass_img'

hat_img = []
glass_img = []

for filename in os.listdir(directory_hat):
    if os.path.isfile(os.path.join(directory_hat, filename)):
        hat_img.append(filename)

for filename in os.listdir(directory_glass):
    if os.path.isfile(os.path.join(directory_glass, filename)):
        glass_img.append(filename)


eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

overlay_image_glasses = Image.open('Sticker_on_Face/imgs/glass_img/'+glass_img[-1])
overlay_image_hat = Image.open('Sticker_on_Face/imgs/hat_img/'+hat_img[-1])
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            face_width = endX - startX
            face_height = endY - startY

            new_width = int(face_width * 2)
            new_height = int(new_width)

            resized_overlay = overlay_image_hat.resize((new_width, new_height), Image.LANCZOS)

            overlay_array = np.array(resized_overlay)

            startX_overlay = startX + (face_width // 2) - (new_width // 2)
            startY_overlay = startY - new_height//2-25
            if startY_overlay < 0:
                startY_overlay = 0

            if overlay_array.shape[2] == 4:
                bgr_overlay = overlay_array[:, :, :3]
                alpha_overlay = overlay_array[:, :, 3]

                for c in range(3):
                    try:
                        frame[startY_overlay:startY_overlay + new_height, startX_overlay:startX_overlay + new_width, c] = (
                            frame[startY_overlay:startY_overlay + new_height, startX_overlay:startX_overlay + new_width, c] * (1 - alpha_overlay / 255.0) +
                            bgr_overlay[:, :, c] * (alpha_overlay / 255.0)
                        )
                    except ValueError as e:
                        print("ValueError:", e)

            else:
                bgr_overlay = overlay_array  
                for c in range(3):
                    frame[startY_overlay:startY_overlay + new_height, startX_overlay:startX_overlay + new_width, c] = bgr_overlay[:, :, c]

            face_roi_gray = cv2.cvtColor(frame[startY:endY, startX:endX], cv2.COLOR_BGR2GRAY)
            eyes = eye_cascade.detectMultiScale(face_roi_gray)

            if len(eyes) == 2:
                eye_1 = eyes[0]
                eye_2 = eyes[1]

                eye_1_center = (startX + eye_1[0] + eye_1[2] // 2, startY + eye_1[1] + eye_1[3] // 2)
                eye_2_center = (startX + eye_2[0] + eye_2[2] // 2, startY + eye_2[1] + eye_2[3] // 2)

                eye_distance = abs(eye_1_center[0] - eye_2_center[0])

                new_width = int(eye_distance * 2)
                new_height = int(overlay_image_glasses.height * (new_width / overlay_image_glasses.width))

                overlay_position_x = min(eye_1_center[0], eye_2_center[0]) - new_width // 4
                overlay_position_y = min(eye_1_center[1], eye_2_center[1]) - new_height // 2

                resized_overlay = overlay_image_glasses.resize((new_width, new_height), Image.LANCZOS)

                overlay_array = np.array(resized_overlay)

                if overlay_array.shape[2] == 4:
                    bgr_overlay = overlay_array[:, :, :3]
                    alpha_overlay = overlay_array[:, :, 3]

                    for c in range(3):
                        try:
                            frame[overlay_position_y:overlay_position_y + new_height,
                                  overlay_position_x:overlay_position_x + new_width, c] = (
                                frame[overlay_position_y:overlay_position_y + new_height,
                                      overlay_position_x:overlay_position_x + new_width, c] * (1 - alpha_overlay / 255.0) +
                                bgr_overlay[:, :, c] * (alpha_overlay / 255.0)
                            )
                        except ValueError:
                            pass
    frame =  cv2.flip(frame, 1)
    cv2.imshow('Face and Glasses Overlay', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
