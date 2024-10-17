from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from imutils.video import VideoStream
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import dlib
import subprocess
#import argparse
import facenet
#import imutils
import os
#import sys
#import math
import pickle
#import align.detect_face
import numpy as np
import cv2
#import collections
from sklearn.svm import SVC
from tkinter import messagebox
import tkinter as tk
from PIL import Image, ImageTk
import pyodbc




path = "C://Users//Acer//OneDrive//Documents//TDH//projects//NCKH_2024//FaceRecog//datalog//"
detector = dlib.get_frontal_face_detector()

tf.compat.v1.disable_eager_execution()

CLASSIFIER_PATH = 'Models/facemodel.pkl'
FACENET_MODEL_PATH = 'Models/20180402-114759.pb'
with open(CLASSIFIER_PATH, 'rb') as file:
    model, class_names = pickle.load(file)

# Tạo session TensorFlow
sess = tf.compat.v1.Session()

# Load mô hình Facenet
facenet.load_model(FACENET_MODEL_PATH)
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

path = "C://Users//Acer//OneDrive//Documents//TDH//projects//NCKH_2024//FaceRecog//datalog//"
detector = dlib.get_frontal_face_detector()

global current_password
current_password = "123456"

def camera():
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (385, 280))
        
        #img = Image.fromarray(img)
        photo = ImageTk.PhotoImage(Image.fromarray(frame))
        label_camera.config(image=photo)
        label_camera.photo = photo
    label_camera.after(10, camera)

   
def recognized():
    IMAGE_DIR = r"C:\Users\Acer\OneDrive\Documents\TDH\projects\NCKH_2024\FaceRecog\datalog"
    files = os.listdir(IMAGE_DIR)
    # Lặp qua tất cả các ảnh trong thư mục
    total_prob = 0.0
    count = 0  # Biến đếm số lượng ảnh
    first_best_name = None  # Biến lưu trữ best_name của ảnh đầu tiên
    if not files:
        print("No images found in the directory.")
        get_object()
        return False  # Trả về False nếu không có ảnh trong thư mục

    for filename in os.listdir(IMAGE_DIR):
        img_path = os.path.join(IMAGE_DIR, filename)
        img = cv2.imread(img_path)
        if img is not None:
            scaled = cv2.resize(img, (160, 160), interpolation=cv2.INTER_CUBIC)
            scaled = facenet.prewhiten(scaled)
            
            scaled_reshape = scaled.reshape(-1, 160, 160, 3)
            feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
            emb_array = sess.run(embeddings, feed_dict=feed_dict)
            
            predictions = model.predict_proba(emb_array)
            best_class_indices = np.argmax(predictions, axis=1)
            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
            best_name = class_names[best_class_indices[0]]

            if count == 0:
                first_best_name = best_name
            else:
                # So sánh best_name của ảnh này với first_best_name
                if best_name != first_best_name:
                    print("Fail to recognize: Different best_name detected in images.")
                    break
            
            # Cộng tổng xác suất cho mỗi tên
            total_prob += best_class_probabilities[0]
            
            count += 1  # Tăng biến đếm số lượng ảnh
        else:
            print("Could not read image:", img_path)

    avg_prob = total_prob / count  # Tính toán xác suất trung bình
    print("Name: {}, Average Probability: {}".format(best_name, avg_prob))
    def remove_label():
        success_label.destroy()
    if(avg_prob > 0.8):
        success_label = tk.Label(root, text="Đăng nhập thành công!\n{}".format(best_name), fg="green", font=("Helvetica", 16, "bold"))
        success_label.place(x = 100, y = 150)
        root.after(3000, remove_label)


    else :
        success_label = tk.Label(root, text="Vui lòng đăng nhập lại!", fg="red", font=("Helvetica", 16, "bold"))
        success_label.place(x = 100, y = 150)
        root.after(3000, remove_label)
        
  
    for filename in os.listdir(IMAGE_DIR):
        file_path = os.path.join(IMAGE_DIR, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print("Error while deleting file:", e)

    # Đóng session TensorFlow
    #sess.close()
def  get_object():
    
    for i in range(5):
        ret, frame = cap.read()
        if ret:
            rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Phát hiện khuôn mặt trong frame
            faces = detector(rgbFrame)
            for face in faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                # Tạo tên và lưu ảnh
                name = path + '\\' + str(i) + '.jpg'
                
                # Kiểm tra giới hạn của khuôn mặt
                if 0 <= y < rgbFrame.shape[0] and 0 <= y + h < rgbFrame.shape[0] and 0 <= x < rgbFrame.shape[1] and 0 <= x + w < rgbFrame.shape[1] and rgbFrame is not None and rgbFrame.shape[0] > 0 and rgbFrame.shape[1] > 0:
                    print(f'Created Image: {i}.jpg')
                    imgg = frame[y-30:y + h + 20, x - 30:x + w + 20]
                    imggg = cv2.resize(imgg, (160, 160))
                    cv2.imwrite(name, imggg)
def change_password():
    root.withdraw()
    def thay_doi(password, new_password, confirm_new_password):
        global current_password  # Sử dụng biến toàn cục current_password
        def remove_label():
            success_label.destroy()
        if password == current_password:
            if new_password == confirm_new_password:
                current_password = new_password  # Cập nhật mật khẩu mới
                #messagebox.showinfo("Thông báo", "Đổi mật khẩu thành công!")
                change_password_window.destroy()
                root.deiconify()
                success_label = tk.Label(root, text="Thay đổi mật khẩu thành công!", fg="green", font=("Helvetica", 16, "bold"))
                success_label.place(x = 50, y = 150)
                root.after(3000, remove_label)
            else:
                messagebox.showerror("Lỗi", "Mật khẩu mới không khớp. Vui lòng nhập lại!")
        else:
            messagebox.showerror("Lỗi", "Mật khẩu hiện tại không đúng!")
    def huy():
        change_password_window.destroy()
        root.deiconify()
    # Tạo cửa sổ giao diện đổi mật khẩu
    change_password_window = tk.Toplevel(root)
    change_password_window.title("Thay đổi mật khẩu")
    change_password_window.geometry("480x320")
    label_password = tk.Label(change_password_window, text="Nhập mật khẩu hiện tại:")
    label_password.place(x = 90, y = 70)
    entry_password = tk.Entry(change_password_window, show="*")
    entry_password.place(x = 230, y = 70)
    label_new_password = tk.Label(change_password_window, text="Nhập mật khẩu mới:")
    label_new_password.place(x = 90, y = 100)
    entry_new_password = tk.Entry(change_password_window, show="*")
    entry_new_password.place(x = 230, y = 100)
    label_confirm_new_password = tk.Label(change_password_window, text="Nhập lại mật khẩu mới:")
    label_confirm_new_password.place(x = 90, y = 130)
    entry_confirm_new_password = tk.Entry(change_password_window, show="*")
    entry_confirm_new_password.place(x = 230, y = 130)
    button_xac_nhan = tk.Button(change_password_window, text="Xác nhận", command=lambda: thay_doi(entry_password.get(), entry_new_password.get(), entry_confirm_new_password.get()), width = 9, height = 1)
    button_xac_nhan.place(x = 230, y = 160)
    button_huy = tk.Button(change_password_window, text="Hủy", command=huy, width=9, height=1)
    button_huy.place(x = 230, y = 190)

def mat_ma():
    root.withdraw()
    def huy_nhap():
        login_window2.destroy()
        root.deiconify()
    def check(password,):
        global current_password
        def remove_label():
            success_label.destroy()
        if password == current_password:
            login_window2.destroy()
            root.deiconify()

            success_label = tk.Label(root, text="Đăng nhập thành công!", fg="green", font=("Helvetica", 16, "bold"))
            success_label.place(x = 100, y = 150)
            root.after(3000, remove_label)
            
        else:
            tk.messagebox.showerror("Lỗi", "mật khẩu không đúng!")
    # Tạo cửa sổ mới cho đăng nhập
    login_window2 = tk.Toplevel(root)
    login_window2.title("Đăng nhập bằng mật mã")
    login_window2.geometry("480x320")
    
    label_mat_ma = tk.Label(login_window2, text="Mật khẩu:")
    label_mat_ma.place(x = 130, y = 130)
    entry_mat_ma = tk.Entry(login_window2, show="*")
    entry_mat_ma.place(x = 200, y = 130)
    login_mat_ma = tk.Button(login_window2, text="Đăng nhập", command=lambda: check(entry_mat_ma.get()), width = 9, height = 1)
    login_mat_ma.place(x = 200, y = 160)
    button_huy_nhap = tk.Button(login_window2, text="Hủy", command=huy_nhap, width=9, height=1)
    button_huy_nhap.place(x = 200, y = 190)
    
def login_window():
    
    root.withdraw()
    def huy_sign():
        login_window1.destroy()
        root.deiconify()
    def check(password):
        global current_password
        if password == current_password:
            login_window1.destroy()
            sign()
        else:
            tk.messagebox.showerror("Lỗi", "Mật khẩu không đúng!")
    # Tạo cửa sổ mới cho đăng nhập
    login_window1 = tk.Toplevel(root)
    login_window1.title("Đăng nhập")
    login_window1.geometry("480x320")
    # Tạo nhãn và ô nhập cho mật khẩu
    label_password = tk.Label(login_window1, text="Mật khẩu:")
    label_password.place(x = 130, y = 130)
    entry_password = tk.Entry(login_window1, show="*")
    entry_password.place(x = 200, y = 130)
    # Tạo nút đăng nhập
    login_button = tk.Button(login_window1, text="Đăng nhập", command=lambda: check(entry_password.get()), width = 9, height = 1)
    login_button.place(x = 200, y = 160)
    button_huy_sign = tk.Button(login_window1, text="Hủy", command=huy_sign, width=9, height=1)
    button_huy_sign.place(x = 200, y = 190)
    

def sign():
        try:
            data_window = tk.Toplevel(root)
            data_window.title("Nhập thông tin")
            data_window.geometry("480x320")
            label_name = tk.Label(data_window, text="Tên:")
            label_name.place(x=130, y=100)
            entry_name = tk.Entry(data_window)
            entry_name.place(x=200, y=100)
            label_id = tk.Label(data_window, text="Mã:")
            label_id.place(x=130, y=130)
            entry_id = tk.Entry(data_window)
            entry_id.place(x=200, y=130)
            def save_data():
                name = entry_name.get()
                id = entry_id.get()
                count = 0
                data_window.destroy()

                path = "C:\\Users\\Acer\\OneDrive\\Documents\\TDH\\projects\\NCKH_2024\\FaceRecog\\Dataset\\" + name + str(id)
                isExists = os.path.exists(path)
                if isExists:
                    print("Aldready taken")
                else:
                    os.makedirs(path)
                detector = dlib.get_frontal_face_detector()

                while True:
                    ret, frame = cap.read()
                    if ret:
                        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        faces = detector(rgbFrame)
                        for face in faces:
                            x, y, w, h = face.left(), face.top(), face.width(), face.height()
                            count += 1

                            name1 = path + '\\' + str(count) + '.jpg'
                            if 0 <= y < rgbFrame.shape[0] and 0 <= y + h < rgbFrame.shape[0] and 0 <= x < rgbFrame.shape[1] and 0 <= x + w < rgbFrame.shape[1]:
                                print(f'Created Image : {count}.jpg')
                                imgg = frame[y-30:y + h + 20, x - 30:x + w + 20]
                                imggg = cv2.resize(imgg, (160, 160))
                                cv2.imwrite(name1, imggg)
                            cv2.rectangle(frame, (x,y), (x + w, y +h), (0,255,240), 2)
                        cv2.imshow("Dataset", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q') or count > 159:
                            break
                cv2.destroyAllWindows()
                root.deiconify()
            def train_classifier():
                command = [
                    "python", 
                    "C:\\Users\\Acer\\OneDrive\\Documents\\TDH\\projects\\NCKH_2024\\FaceRecog\\scr\\classifier.py", 
                    "TRAIN", 
                    "C:\\Users\\Acer\\OneDrive\\Documents\\TDH\\projects\\NCKH_2024\\FaceRecog\\Dataset",
                    "Models/20180402-114759.pb",
                    "Models/facemodel.pkl",
                    "--batch_size",
                    "1000"
                ]
                
                try:
                    # Thực thi lệnh
                    subprocess.run(command, check=True)
                    print("Training completed successfully.")
                except subprocess.CalledProcessError as e:
                    # Xử lý lỗi nếu có
                    print("Error:", e)
            def dang_ky():
                save_data()
                train_classifier()
            def huy_luu():
                data_window.destroy()
                root.deiconify()
            # Tạo nút để lưu thông tin
            save_button = tk.Button(data_window, text="Lưu", command=dang_ky, width = 9, height = 1)
            save_button.place(x=200, y=160)
            button_huy_luu = tk.Button(data_window, text="Hủy", command=huy_luu, width = 9, height = 1)
            button_huy_luu.place(x = 200, y = 190)
        except Exception as e:
            messagebox.showerror("Lỗi", str(e))

def nhan_dien():
    get_object()
    recognized()

root = tk.Tk()
root.title("Camera")
root.geometry("480x320")

cap = cv2.VideoCapture(0)

label_camera = tk.Label(root)
label_camera.place(x = 0, y = 30)
label1_camera = tk.Label(root)
label1_camera.place(x = 0, y = 30)
login_button = tk.Button(root, text="Đăng nhập", command=nhan_dien, width=9, height=2)
login_button.place(x = 400, y = 30)
'''login_mat_ma = tk.Button(root, text="Mật mã", command=mat_ma, width=9, height=2)
login_mat_ma.place(x = 400, y = 75)
signup_button = tk.Button(root, text="Đăng kí", command=login_window, width=9, height=2)
signup_button.place(x = 400, y = 120)
'''
def show_options(event=None):
    options_menu.post(menu_button.winfo_rootx(), menu_button.winfo_rooty()+menu_button.winfo_height())

options_menu = tk.Menu(root, tearoff=0)
options_menu.add_command(label="Đăng ký", command=login_window)
options_menu.add_command(label="Đăng nhập mật mã", command=mat_ma)
options_menu.add_command(label="Thay đổi mật mã", command = change_password)

menu_button = tk.Label(root, text="...", width=3, relief="raised")
menu_button.place(x = 400, y = 75)
menu_button.bind("<Button-1>", show_options)

label1 = tk.Label(root, text="Xin chào!", font=("Helvetica", 16, "bold"))
label1.place(x = 150, y = 0)
camera()

root.mainloop()

cap.release()
cv2.destroyAllWindows()

