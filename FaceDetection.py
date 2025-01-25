import cv2 as cv

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

img = cv.imread('face.png')
resized_img = cv.resize(img, None, fx=0.5, fy=0.5)

gray = cv.cvtColor(resized_img, cv.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.1, 4)

idCard = cv.imread('card.png')
width, height = 252, 272

if len(faces) > 0:
    (x, y, w, h) = faces[0]

    padding_x = int(w * 1.5)
    padding_y = int(h * 1.5)
    
    x_new = max(0, x - padding_x)
    y_new = max(0, y - padding_y)
    x_end = min(resized_img.shape[1], x + w + padding_x)
    y_end = min(resized_img.shape[0], y + h + padding_y)

    center = (float(x + w / 2), float(y + h / 2))
    patch_size = (min(width, resized_img.shape[1] - x), min(height, resized_img.shape[0] - y))
    patch = cv.getRectSubPix(resized_img, (padding_x,padding_y), center)
    resized_patch = cv.resize(patch, (width, height), interpolation=cv.INTER_AREA)
    
    cv.rectangle(resized_img, (x, y), (x + w, y + h), (255, 0, 0), 2)


start_x, start_y = 687, 216
end_x, end_y = start_x + width, start_y + height

idCard[start_y:end_y, start_x:end_x] = resized_patch

cv.imshow('Image', resized_img)
cv.imshow('Card', idCard)
cv.imshow('Patch', patch)
cv.waitKey(0)
cv.destroyAllWindows()
