import cv2 as cv

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

img = cv.imread('faceGroup.png')
resized_img = cv.resize(img, None, fx=1, fy=1 )

gray = cv.cvtColor(resized_img, cv.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.1, 4)

idCard = cv.imread('card.png')
width, height = 252, 272
start_x, start_y = 687, 216
end_x, end_y = start_x + width, start_y + height

if len(faces) > 0:
    largest_face = None
    max_area = 0
    
    for (x, y, w, h) in faces:
        value = (x, y, w, h , (w * h)) 
        area = w * h
        
        cv.rectangle(resized_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if area > max_area:
            max_area = area
            largest_face = (x, y, w, h)
            
        face_area = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(face_area)
        print(eyes)
        for (ex, ey, ew, eh) in eyes:
            eye_center = x + ex + ew // 2, y + ey + eh // 2
            radius = max(1, ew // 4)
            cv.circle(resized_img, eye_center,  radius, (0, 255, 0), 2)
            
    if largest_face:
        (x, y, w, h) = largest_face
        
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

        idCard[start_y:end_y, start_x:end_x] = resized_patch
        
        cv.rectangle(resized_img, (x, y), (x + w, y + h), (0, 255, 0), 2)     
        
else:
    print("No face detected.")
    cv.waitKey(0)
    exit()

cv.imshow('Image', resized_img)
cv.imshow('Card', idCard)
cv.imshow('Patch', patch)
cv.waitKey(0)
cv.destroyAllWindows()
