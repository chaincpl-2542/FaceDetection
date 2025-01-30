import cv2 as cv
import math
import argparse

def auto_resize_image(image, max_width, max_height):
    height, width = image.shape[:2]
    
    if width > max_width or height > max_height:
        scaling_factor = min(max_width / width, max_height / height)
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        
        resized_image = cv.resize(image, (new_width, new_height), interpolation=cv.INTER_AREA)
        
        return resized_image
    
    return image

def main() -> None:
    parser = argparse.ArgumentParser(description="Face detection")
    parser.add_argument("-i", "--image", required=True, help="Path to the image to be scanned")
    args = parser.parse_args()
    
    image = cv.imread(args.image)
    if image is None:
        raise ValueError("Could not read the image.")

    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
    mouth_cascade = cv.CascadeClassifier('haarcascade_mouth.xml')

    MAX_WIDTH = 800
    MAX_HEIGHT = 800

    img = cv.imread(args.image)
    resized_img = auto_resize_image(img, MAX_WIDTH, MAX_HEIGHT)
    gray = cv.cvtColor(resized_img, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    mouths = mouth_cascade.detectMultiScale(gray, 1.3, 5)

    idCard = cv.imread('card.png')
    width, height = 252, 272
    start_x, start_y = 687, 216
    end_x, end_y = start_x + width, start_y + height

    if len(faces) > 0:
        largest_face = None
        max_area = 0
        
        for (x, y, w, h) in faces:
            area = w * h

            if area > max_area:
                max_area = area
                largest_face = (x, y, w, h)
                
        if largest_face:
            (x, y, w, h) = largest_face
            
            padding_x = int(w * 1.5)
            padding_y = int(h * 1.5)
            
            face_center = (float(x + w / 2), float(y + h / 2))
            
            face_area = gray[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(face_area)
            
            # Draw face
            cv.rectangle(resized_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            if len(eyes) >= 2:
                eye_1 = eyes[0]  
                eye_2 = eyes[1]
                
                if eye_1[0] < eye_2[0]:
                    eye_1, eye_2 = eye_2, eye_1
                
                eye_1_center = (x + eye_1[0] + eye_1[2] // 2, y + eye_1[1] + eye_1[3])
                eye_2_center = (x + eye_2[0] + eye_2[2] // 2, y + eye_2[1] + eye_2[3])
                
                dx = eye_1_center[0] - eye_2_center[0]
                dy = eye_1_center[1] - eye_2_center[1]
                angle = math.degrees(math.atan2(dy ,dx))
                
                # Draw circle at eyes position
                for (ex, ey, ew, eh) in eyes:
                    eye_center = (x + ex + ew // 2, y + ey + eh // 2)
                    radius = max(1, ew // 4)  # ลดรัศมีของวงกลม
                    cv.circle(resized_img, eye_center, radius, (0, 255, 0), 2)

                eyes_center = ((float)(eye_1_center[0] + eye_2_center[0]) // 2, (float)(eye_1_center[1] + eye_2_center[1]) // 2)
                eyes_center_y = eyes_center[1]
                
                # Draw rectangle at mouth position
                for (x, y, w, h) in mouths:
                    if y > eyes_center_y:
                        cv.rectangle(resized_img, (x, y), (x + w, y + h), (255, 255, 0), 2)

                rotation_matrix = cv.getRotationMatrix2D(face_center, angle, 1)
                rotated_img = cv.warpAffine(resized_img, rotation_matrix, (resized_img.shape[1],resized_img.shape[0]))     
                
                resized_img = rotated_img
                gray = cv.cvtColor(resized_img, cv.COLOR_BGR2GRAY)  
                
                ###################################################################################################
                
                patch = cv.getRectSubPix(resized_img, (padding_x,padding_y), face_center)
                resized_patch = cv.resize(patch, (width, height), interpolation=cv.INTER_AREA)

                idCard[start_y:end_y, start_x:end_x] = resized_patch
                
            else:
                print("Less than 2 eyes detected.")
                
    else:
        print("No face detected.")
        cv.waitKey(0)
        exit()

    cv.imshow('Card', idCard)
    cv.imshow('Patch', resized_patch)
    cv.imshow('Rotated Image', rotated_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
if __name__ == "__main__":
    main()
