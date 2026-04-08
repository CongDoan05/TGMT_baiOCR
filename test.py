import cv2 as cv
import easyocr

#data
img_path = "image.png"

#read image
img = cv.imread(img_path, cv.IMREAD_COLOR)
#convert to gray
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)#0 ->255

#BINARY IMAGE 0, 1
_, binary = cv.threshold(gray, 128, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
#clean image, noise cancelation
clean_img = cv.fastNlMeansDenoising(binary, h=10)

#find all edges
edges = cv.Canny(clean_img, 30, 128)

#find closed bbox
keypoints = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
closed_bbox = cv.morphologyEx(edges, cv.MORPH_CLOSE, keypoints)

#find contours
contours, _ = cv.findContours(closed_bbox.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

#find box by box to find candidate of plate
plates = []
img_size = img.shape[0]*img.shape[1]

for cnt in contours:
    x, y, w, h = cv.boundingRect(cnt)
    aspect_ratio = w/h # tỷ lệ của chiều dài và chiều cao
    area_ratio = (w*h)/img_size #tỷ lệ diện tích của biển số trên khung hình
    if (1.0 < aspect_ratio < 6.0) and (0.005 < area_ratio <0.15):
        plates.append((x,y,w,h))
        cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)

def crop_plate(x,y,w,h, pad = 5):
    x1 = max(0, x-pad)
    y1 = max(0, y-pad)
    x2 = min(img.shape[1], x+w+pad)
    y2 = min(img.shape[1], y+h+pad)
    return img[y1:y2, x1:x2]

for i, (x,y,w,h) in enumerate(plates):
    plate_img = crop_plate(x,y,w,h)
    reader = easyocr.Reader(["en"], gpu=False)
    gimg = cv.cvtColor(plate_img, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gimg, 128, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    # cv.imshow("plate_img", gimg)
    # cv.waitKey(0)
    
    results = reader.readtext(binary, allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXY", detail=1)
    for (bbox, text, conf) in results:
        print(f"Text: {text}, Confidence: {conf}")

# cv.waitKey(0)
# cv.destroyAllWindows()