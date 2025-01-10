"""Модуль для домашней работы по теме "Обзор сторонних библиотек Python"
   Задача: на фотографиях керна найти область где находится желтая линейка
   отобразить исходную фотографию с выделенной областью желтой линейки на форме tkinter
   и сохранить выделенную область в файл .jpg в папку ./YellowRulerPhoto/
   *керн - это цилиндрические образцы горных пород получаемые при бурении скважин для их дальнейшего изучения
   Как работает: Пользователь выбирает файл с исходным изображением керна из папки ./PhotoCore/
   создается форма на которой размещена выбранная фотография с выделенной областью желтой линейки,
   в папку ./YellowRulerPhoto/ сохраняется изображение jpg с выделенной областью желтой линейки
"""

import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import os

yellow_ruler_dir = './YellowRulerPhoto/'


def get_file_name():
    """функция для получения имени файла с фотографией керна"""
    file_name = filedialog.askopenfilename(initialdir="./PhotoCore", title="выберите файл")
    return file_name

def get_pure_colors_image(image):
    """функция преобразует изображение в палитру из 8 цветов"""
    return  (image[:, :, :] >= 127).astype(int) * 255

def find_yellow_ruler(image):
    """функция ищет область с желтой линейкой и координаты левого верхнего угла области возвращает, а также ширину и
      высоту этой области"""
    # переводим исходное изображение в палитру из 8 цветов
    img = get_pure_colors_image(image)
    # cv2.imwrite(f"./_8colors.jpg", img)
    img = cv2.convertScaleAbs(img)
    #с помощью функция из библиотеки CV2 конвертируем изображение из цветового пространства RGB в HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #задаем уровни желтого и с помощью функции из библиотеки CV2 получаем список контуров с преобладающим желтым цветом
    lower_yellow = np.array([25, 50, 70])
    upper_yellow = np.array([35, 255, 255])
    mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # в цикле находим контур с максимальной высотой и возвращаем кортеж с его координатами левого верхнего угла, шириной
    # и высотой
    yellow_ruler_region = ()
    max_h = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h > max_h:
            max_h = h
            yellow_ruler_region =(x, y, w, h)
    return yellow_ruler_region

# выполняем программу
file_name = get_file_name()
img = cv2.imread(file_name)
yellow_ruler_region = find_yellow_ruler(img)
left_x = yellow_ruler_region[0]
left_y = yellow_ruler_region[1]
right_x = yellow_ruler_region[0]  + yellow_ruler_region[2]
right_y = yellow_ruler_region[1]  + yellow_ruler_region[3]
yellow_ruler_img = img[left_y:right_y, left_x:right_x]
number = len([f for f in  os.listdir(yellow_ruler_dir)])
cv2.imwrite(f"./YellowRulerPhoto/yellow_ruler_img_{number}.jpg", yellow_ruler_img)
cv2.rectangle(img,(left_x, left_y), (right_x, right_y), (0, 0, 255), thickness=10, lineType=cv2.LINE_AA )

# создаём окно и выводим результат
main_window = tk.Tk()
main_window.title('Результаты поиска жёлтой линейки')
img = img.astype(np.uint8)
pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
pil_image = pil_image.resize((pil_image.size[0]//5, pil_image.size[1]//5))
tk_image = ImageTk.PhotoImage(pil_image)
photo_box = tk.Label(main_window, image=tk_image)
photo_box.pack(side="left", padx=10, pady=10)
main_window.mainloop()