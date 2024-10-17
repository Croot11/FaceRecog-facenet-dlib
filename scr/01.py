import cv2
import os
import dlib

# Khởi tạo camera
cam = cv2.VideoCapture(0)
count = 0

# Nhập tên và tạo thư mục
nameD = str(input("Name: ")).lower()
path = "C:\\Users\\Acer\\OneDrive\\Documents\\dataLe\\" + nameD

detector = dlib.get_frontal_face_detector()

# Kiểm tra xem thư mục đã tồn tại chưa
isExists = os.path.exists(path)
if not isExists:
    os.makedirs(path)

# Vòng lặp chụp ảnh
while True:
    ret, frame = cam.read()
    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Phát hiện khuôn mặt trong frame
    faces = detector(rgbFrame)
    
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        count += 1

        # Tạo tên và lưu ảnh
        name = path + '\\' + str(count) + '.jpg'
        
        # Kiểm tra giới hạn của khuôn mặt
        if 0 <= y < rgbFrame.shape[0] and 0 <= y + h < rgbFrame.shape[0] and 0 <= x < rgbFrame.shape[1] and 0 <= x + w < rgbFrame.shape[1] and rgbFrame is not None and rgbFrame.shape[0] > 0 and rgbFrame.shape[1] > 0:
            print(f'Created Image: {count}.jpg')
            imgg = frame[y-30:y + h + 20, x - 30:x + w + 20]
            imggg = cv2.resize(imgg, (160, 160))
            cv2.imwrite(name, imggg)
            
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 240), 2)
    
    # Hiển thị frame
    cv2.imshow("Dataset", frame)
    
    # Thoát khỏi vòng lặp khi nhấn 'q' hoặc đủ số ảnh
    if cv2.waitKey(1) & 0xFF == ord('q') or count > 159:
        break

# Giải phóng và đóng cửa sổ
cam.release()
cv2.destroyAllWindows()
