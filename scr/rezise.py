import cv2

path = r"C:\Users\Acer\Downloads\126-e1649907727656.png"
new_path = r"C:\Users\Acer\Downloads\qllh.jpg"
img = cv2.imread(path)

resize_img = cv2.resize(img, (855, 433))
cv2.imwrite(new_path, resize_img)
cv2.imshow("hhh", resize_img)
cv2.waitKey()