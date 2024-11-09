"""
2024秋季学期《Python编程技术》  指导教师：陈建文
第七组  题目：图像主体计数系统
组长：王艺澄
组员：易俊哲 武晨曦 胡玮承 张俊哲 龙文振 胡波 郑帅 刘柏煜 钟轶群
"""

import tkinter as tk
from tkinter import messagebox

import cv2
from PIL import Image, ImageTk
from tkinterdnd2 import DND_FILES, TkinterDnD
from ultralytics import YOLO

model = YOLO('yolov8n-face.pt', verbose=False)


def detect_faces(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            messagebox.showerror("错误", "无法读取图片")
            return 0, None

        # 进行检测，并获取检测结果
        results = model.predict(img)
        result_img = results[0].plot()  # 获取带检测框的图像
        face_count = 0
        # 提取检测结果
        for result in results:
            boxes = result.boxes.xyxy  # 边界框坐标
            scores = result.boxes.conf  # 置信度分数
            classes = result.boxes.cls  # 类别索引

        # 从索引中获取类型名称
        class_names = [model.names[int(cls)] for cls in classes]

        # 打印检测结果
        for box, score, class_name in zip(boxes, scores, class_names):
            if class_name == 'face' and score > 0.3:
                face_count += 1
        print(f"{face_count}")

        print("finish test")

        return face_count, result_img

    except Exception as e:
        messagebox.showerror("错误", f"处理图片时发生错误: {str(e)}")
        return 0, None


def drop(event):
    file_path = event.data
    count, output_img = detect_faces(file_path)

    # 显示计数结果
    result_label.config(text=f"检测到 {count} 个人")

    if output_img is not None:
        # 显示图像
        img = Image.fromarray(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
        img.thumbnail((400, 400))  # 调整图像大小
        imgtk = ImageTk.PhotoImage(image=img)
        panel.imgtk = imgtk  # 保留引用，防止垃圾回收
        panel.config(image=imgtk)


# 创建GUI窗口
root = TkinterDnD.Tk()
root.attributes("-topmost", True)
root.title("人脸计数系统")
# 获取屏幕尺寸
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# 计算窗口在屏幕中的位置
window_width = 600
window_height = 400
x = (screen_width - window_width) // 2
y = (screen_height - window_height) // 2

# 设置窗口位置和尺寸
root.geometry(f"{window_width}x{window_height}+{x}+{y}")

panel = tk.Label(root)
panel.pack(padx=10, pady=10)

result_label = tk.Label(root, text="请拖入图片", font=("Arial", 16))
result_label.pack(padx=10, pady=10)

# 设置拖放功能
root.drop_target_register(DND_FILES)
root.dnd_bind('<<Drop>>', drop)

root.mainloop()
