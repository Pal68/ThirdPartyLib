"""Модуль для домашней работы по теме "Обзор сторонних библиотек Python"
   Задача: на полученных с помощью модуля get_yellow_ruler изображениях желтой линейки
   найти, преобразовать к размеру 32х32 пикселя и сохранить в отдельные файлы jpg в папку ./NumberPhoto/
   области изображения содержащие двухзначные числа.
   Примечание: Алгоритм ищет двузначные числа в метрической системе линейки (т.е. с левой стороны линейки)
   Как работает: Пользователь выбирает файл с изображением желтой линейки из папки ./YellowRulerPhoto/
   выполняется программа, в папку ./NumberPhoto/ сохраняется изображение jpg с найденными двухзначными числами"""

import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os

num_image_directory = './NumberPhoto/'

def get_file_name():
    """функция для получения имени файла с фотографией керна"""
    file_name = filedialog.askopenfilename(initialdir="./YellowRulerPhoto", title="выберите файл")
    return file_name

def find_objects(binary_image):
    """ищет объекты на изображении и убирает объекты с площадью менее 25"""
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, 8, cv2.CV_32S)
    # get areas of all components except the background (first label)
    areas = stats[1:, cv2.CC_STAT_AREA]
    result = np.zeros(labels.shape, np.uint8)
    for i, area in enumerate(areas):
        if area >= 25:
            result[labels == i+1] = 255
    return result

def get_all_contours(binary_image):
    """функция возвращает все найденные контуры объектов"""
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours, hierarchy

def get_number_countours(all_countours):
    """функция возвращает только контуры объектов которые она приняла за двухзначные числа и которые находятся
    в метрической части линейки"""
    min_x = []
    max_x = []
    min_y = []
    max_y = []
    num_i = 0
    for c in all_contours:
        # получаем высоту контура, минимальную и максимальную координату по х контура.
        rx = np.max(c[:, 0:c.shape[1], 0])
        lx = np.min(c[:, 0:c.shape[1], 0])
        h = np.max(c[:, 0:c.shape[1], 1]) - np.min(c[:, 0:c.shape[1], 1])
        # если контур относится к нужной нам цифре, если его высота больше 5 и он находится слева от центра исходного
        # изображения, т.е в метрической части линейки, то добавляем в список координаты прямоугольника описывающего
        # контур
        if rx < origin_img.shape[1] / 2 and lx > 0 and h > 5:
            min_x.append(np.min(c[:, 0:c.shape[1], 0]))
            max_x.append(np.max(c[:, 0:c.shape[1], 0]))
            min_y.append(np.min(c[:, 0:c.shape[1], 1]))
            max_y.append(np.max(c[:, 0:c.shape[1], 1]))
    # получаем гистограмму распределения левых верхних углов прямоугольников описывающих контур
    v, b = np.histogram(min_x, range(0, 300))
    bound_not_zero = [b[i + 1] if v[i + 1] > 0 else b[i] for i in range(0, len(v) - 1)
                      if v[i] == 0 and v[i + 1] > 0 or (v[i] > 0 and v[i + 1] == 0)]
    count_not_zero = [sum(v[bound_not_zero[i]: bound_not_zero[i + 1] + 1]) for i in
                      range(0, len(list(bound_not_zero)) - 1, 2)]
    # ищем интервал с наибольшей частотой попадания координат левых верхних углов прямоугольников описывающих контур
    k = 0
    max_count = 0
    left_x_interval = ()
    for i in range(0, len(bound_not_zero) - 1, 2):
        if count_not_zero[k] > max_count:
            max_count = count_not_zero[k]
            left_x_interval = (bound_not_zero[i], bound_not_zero[i + 1])
        print(f"{bound_not_zero[i]} - {bound_not_zero[i + 1]}   {count_not_zero[k]}")
        k = k + 1
    # фильтруем список всех найденных контуров
    # оставляем только те у которых х к левого верхнего угла лежит в найденном на предыдущем шаге интервале
    number_contours = [c for c in all_contours if np.min(c[:, 0:c.shape[1], 0]) >= left_x_interval[0]
                       and np.min(c[:, 0:c.shape[1], 0]) <= left_x_interval[1]
                       and (np.max(c[:, 0:c.shape[1], 1]) - np.min(c[:, 0:c.shape[1], 1])) < 200
                       and (np.max(c[:, 0:c.shape[1], 1]) - np.min(c[:, 0:c.shape[1], 1])) > 5]
    number_contours.sort(key=lambda x: np.min(x[:, 0, 1]))
    # так как необходимо найти двухзначные числа получаем список индексов контуров у которых расстояние между ними
    # меньше половины их ширины, чтобы в дальнейшем объединить их в одно изображение
    intervals = [(i, i + 1) for i in range(0, len(number_contours) - 1)
        if np.min(number_contours[i + 1][:, 0:c.shape[1], 1]) - np.max(number_contours[i][:, 0:c.shape[1], 1])
        < (np.max(number_contours[i + 1][:, 0:c.shape[1], 1]) - np.min(number_contours[i + 1][:, 0:c.shape[1], 1]))/2
        and (np.max(number_contours[i + 1][:, 0:c.shape[1], 1]) - np.min(number_contours[i + 1][:, 0:c.shape[1], 1]))>0]
    return number_contours, intervals

def get_num_images(number_contours, num_item):
    """функция возвращает изображения двухзначных чисел"""
    c = number_contours[num_item[0]]
    c1 = number_contours[num_item[1]]
    num_image = binary_ruler[np.min(c[:, 0:c.shape[1], 1]) - 2: np.max(c1[:, 0:c.shape[1], 1]) + 2,
                 np.min(c[:, 0:c.shape[1], 0]) - 2: np.max(c1[:, 0:c.shape[1], 0]) + 2]
    num_image = cv2.resize(num_image, (32, 32))
    return num_image

#получаем исходное изображение
img_name = get_file_name()
origin_img = cv2.imread(img_name)
img = cv2.imread(img_name)
#преобразуем в цветовое пространство HSV
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
# задаем уровни черного и бинаризируем изображение
lower1 = np.array([0, 0, 0])
upper1 = np.array([180, 255, 150])
binary_ruler = cv2.inRange(hsv, lower1, upper1)
# cv2.imwrite('./tmp.jpg', binary_ruler)
binary_ruler = find_objects(binary_ruler)
all_contours, hierarchy = get_all_contours(binary_ruler)

# получаем контура цифр и сохраняем их в файлы .jpg
number_contours, intervals1 = get_number_countours(all_contours)
for i in intervals1:
    curr_num_image = get_num_images(number_contours,i)
    number = len([f for f in os.listdir(num_image_directory)])
    cv2.imwrite(f'{num_image_directory}num_image{number}.jpg', curr_num_image)



