import cv2
from ultralytics import YOLO
import easyocr
import time

# load model detect biển số
model = YOLO("best.pt")

# load OCR
reader = easyocr.Reader(['en'])

cap = cv2.VideoCapture("plate2.mp4")

# lưu biển số + timestamp
detected_plates = {}

DISPLAY_TIME = 3

frame_count = 0


while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    results = model(frame)

    for result in results:

        for box in result.boxes.xyxy:

            x1, y1, x2, y2 = map(int, box)

            plate_crop = frame[y1:y2, x1:x2]

            if plate_crop.size == 0:
                continue

            plate_text = ""

            if frame_count % 5 == 0:

                ocr_result = reader.readtext(plate_crop)

                for text in ocr_result:

                    plate_text = text[1].strip()

                    if len(plate_text) >= 5:
                        detected_plates[plate_text] = time.time()

            # vẽ bbox
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

            # hiển thị text trên bbox
            if plate_text != "":
                cv2.putText(
                    frame,
                    plate_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2
                )


    # ================= PANEL TRẮNG MỜ =================

    panel_x1 = 20
    panel_y1 = 20
    panel_x2 = 320
    panel_y2 = 220

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (panel_x1, panel_y1),
        (panel_x2, panel_y2),
        (255,255,255),
        -1
    )

    alpha = 0.4

    frame = cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1 - alpha,
        0
    )

    # viền panel
    cv2.rectangle(
        frame,
        (panel_x1, panel_y1),
        (panel_x2, panel_y2),
        (0,255,0),
        2
    )


    # ================= HEADER PANEL =================

    cv2.putText(
        frame,
        "Detected Plates",
        (panel_x1 + 10, panel_y1 + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,0,0),
        2
    )


    # ================= TEXT TRONG PANEL =================

    y_offset = panel_y1 + 70

    current_time = time.time()

    max_lines = 4


    for plate in list(detected_plates.keys())[:max_lines]:

        if current_time - detected_plates[plate] < DISPLAY_TIME:

            cv2.putText(
                frame,
                plate,
                (panel_x1 + 10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,0,0),
                2
            )

            y_offset += 35

        else:
            del detected_plates[plate]


    cv2.imshow("Plate Detection", frame)

    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows()