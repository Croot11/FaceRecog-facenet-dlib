import os
import tkinter as tk
from tkinter import messagebox

# Đường dẫn đến thư mục chứa tệp ảnh của người dùng
folder_path = r"C:\Users\Acer\OneDrive\Documents\NCKH\datatrain"

def get_user_files(folder_path):
    user_files = os.listdir(folder_path)
    return user_files

def show_user_list():
    user_window = tk.Toplevel(root)
    user_window.title("Danh sách tệp người dùng")

    # Lấy danh sách các tệp trong thư mục chứa ảnh của người dùng
    user_files = get_user_files(folder_path)

    # Tạo danh sách người dùng trong giao diện
    user_listbox = tk.Listbox(user_window)
    for file in user_files:
        user_listbox.insert(tk.END, file)
    user_listbox.pack()

    # Tạo nút xóa người dùng
    delete_button = tk.Button(user_window, text="Xóa người dùng", command=lambda: delete_user(user_listbox.get(tk.ACTIVE)))
    delete_button.pack()

def delete_user(selected_file):
    if selected_file:
        file_path = os.path.join(folder_path, selected_file)
        os.remove(file_path)
        messagebox.showinfo("Thông báo", "Đã xóa tệp '{}' thành công.".format(selected_file))
    else:
        messagebox.showerror("Lỗi", "Vui lòng chọn một tệp để xóa.")

root = tk.Tk()
root.geometry("300x200")

# Tạo nút hiển thị danh sách người dùng và xóa người dùng
show_user_button = tk.Button(root, text="Hiển thị danh sách người dùng và xóa", command=show_user_list)
show_user_button.pack()

root.mainloop()
