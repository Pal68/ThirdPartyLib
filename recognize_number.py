"""Модуль для домашней работы по теме "Обзор сторонних библиотек Python"
   Задача: распознать числа на изображениях полученных в модуле get_num_from_yellow_ruler
   Как работает: Пользователь сначала должен выполнить модуль get_yellow_ruler и get_num_from_yellow_ruler
   после выполнения этих модулей в папке ./NumberPhoto/ будут изображеня с числами для распознавания
   при запуске модуля из папки ./NumberTrainingSet/ загрузится тренировочный набор изображений чисел,
   отобразиться форма с изображением тестирумого числа и полем с результатом распознавания кнопками вперед, назад
   можно перебирать изображнеия для тестирования"""

import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk


test_path = './NumberTrainingSet/'
work_path = './NumberPhoto/'

def copyPartMatrix(matr, leftX, rightX, topY, baseY):
    result = matr[topY:baseY+1, leftX:rightX+1, 0:3]
    return result

def getMeanDelta(matr):
    mean_delta=0.0
    for i in range(1, matr.shape[0] -1):
        for j in range(1, matr.shape[1]-1):
            mean_delta =mean_delta+ (abs(matr[i, j] - matr[i, j - 1]) + abs(matr[i, j] - matr[i - 1, j - 1]) +
                           abs(matr[i, j] - matr[i - 1, j]) + abs(matr[i, j] - matr[i - 1, j + 1])) / 4.0
    mean_delta /= (matr.shape[1]-2 ) * (matr.shape[0]-2 )  # normalizing
    return mean_delta

def getLBPMatrix(img,chanel):
    """функция возвращает LBP матрицу для изображения"""
    if chanel in range(0,3):
        result_matrix=img[:,:,chanel]*1.0
    else:
        result_matrix = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

    mean_delta = getMeanDelta(result_matrix)
    lbp = np.zeros((result_matrix.shape[0], result_matrix.shape[1]))
    for i in range(1,result_matrix.shape[0]-1):
       for j in range(1,result_matrix.shape[1]-1):

            data=((result_matrix[i, j] - result_matrix[max(0, i-1):min(result_matrix.shape[0], i+2), max(0, j-1):
                                                            min(result_matrix.shape[1], j+2)]) <= mean_delta)
            c=np.zeros((3,3))
            s = data.size
            match s :
                case 4:
                    if i==0 and j==0:#Левый верхний угол
                       c[1:3, 1:3] = data
                    elif i!=0 and j!=0:#правый нижний угол
                       c[0:2, 0:2] = data
                    elif i!=0 and j==0:#Левый нижний угол
                        c[0:2, 1:3] = data
                    else:
                        c[1:3, 0:2] = data#правый верхний угол
                case 6 :
                    if i==result_matrix.shape[0]-1 and j>0:#нижняя сторона
                        c[0:2,0:3]=data
                    elif i==0 and j>0:#верхняя сторона
                        c[1:3, 0:4] = data
                    elif i!=0 and j==0:
                        c[0:4, 1:4] = data#левая сторона
                    elif i!=0 and j==result_matrix.shape[1]-1:
                        c[0:4, 0:2] = data  # правая сторона
                case _:
                    c[0:4,0:4]=data

            tmpV=np.zeros(8)
            jj=0
            for ii in range(0,c.shape[1]):
                tmpV[jj]=c[0,ii]
                jj=jj+1
            for ii in range(1, c.shape[0]):
                tmpV[jj]=c[ii,c.shape[0]-1]
                jj=jj+1
            for ii in range(c.shape[1]-2,-1,-1):
                tmpV[jj]=c[c.shape[0]-1,ii]
                jj=jj+1
            for ii in range(c.shape[0]-2,0,-1):
                tmpV[jj] = c[ii,0]
            mnojiteli = (128, 64, 32, 16, 8, 4, 2, 1)
            lbp[i, j] = np.sum(tmpV * mnojiteli)
    return lbp

def getUniFormLBPHist(matr):
    """функция возвращает гитограмму распределения униформных LBP """
    uniform_lbp= [1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31, 32, 48, 56, 60, 62, 63, 64, 96, 112, 120,
                  124, 126, 127, 128, 129, 131, 135, 143, 159, 191, 192, 193, 195, 199, 207, 223, 224, 225, 227,
                  231, 239, 240, 241, 243, 247, 248, 249, 251, 252, 253, 254]
    bins = [1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31, 32, 48, 56, 60, 62, 63, 64, 96, 112, 120,
            124, 126, 127, 128, 129, 131, 135, 143, 159, 191, 192, 193, 195, 199, 207, 223, 224, 225, 227,
            231, 239, 240, 241, 243, 247, 248, 249, 251, 252, 253, 254,255,256]
    for i in range(0, matr.shape[0]):
        for j in range(0,matr.shape[1]):
            if matr[i, j] not in uniform_lbp:
                matr[i,j]=255
    lbpHist, bins = np.histogram(a=matr,bins=bins,density = True)
    return lbpHist, bins

def getLBPPropsVector(matr, blok_size):
    """функция возвращает LBP вектор свойств изображения"""
    # Определяем количество итераций по ширине и высоте
    end_w_iteration = (matr.shape[1] // blok_size) * 2 - 1
    end_h_iteration = (matr.shape[0] // blok_size) * 2 - 1

    w_over = matr.shape[1] % blok_size
    h_over = matr.shape[0] % blok_size

    gray_lbp_list = []
    def process_block(ii, jj):
        cyr_blok_matrix = copyPartMatrix(matr, jj * blok_size // 2, jj * blok_size // 2 + (blok_size - 1),
                                         ii * blok_size // 2, ii * blok_size // 2 + (blok_size - 1))
        gray_blok_matrix = getLBPMatrix(cyr_blok_matrix, 1)
        gray_LBP_hist, _ = getUniFormLBPHist(gray_blok_matrix)
        gray_lbp_list.extend(gray_LBP_hist)

    # Основные итерации
    for ii in range(end_h_iteration + 1):
        for jj in range(end_w_iteration + 1):
            process_block(ii, jj)

        # Обработка остатка по ширине
        if w_over > 0 and jj == end_w_iteration:
            process_block(ii, end_w_iteration)

    # Обработка остатка по высоте
    if h_over > 0:
        for jj in range(end_w_iteration + 1):
            process_block(end_h_iteration, jj)
    return gray_lbp_list

def getCOSSimilarity(v1,v2):
    """функция возвращает косинусную меру сходства двух векторов свойств"""
    result = np.dot(v1,v2)/(np.linalg.norm(v1) * np.linalg.norm(v2))
    return result


test_files = [f"{test_path}{f}" for f in listdir(test_path) if isfile(join(test_path, f))]
num_images = [(f, cv2.imread(f)) for f in test_files]
test_num_arr = [(n[0], getLBPPropsVector(n[1],8)) for n in num_images]

cur_files = [f"{work_path}{f}" for f in listdir(work_path) if isfile(join(work_path, f))]
cur_images = [(f, cv2.imread(f)) for f in cur_files]
cur_num_arr = [(n[0], getLBPPropsVector(n[1],16)) for n in cur_images]
print(test_num_arr[0])

item_ = 0
n1 = f'{cur_num_arr[0][0]}'

def main():
    global item_
    global cur_num_arr
    global cur_images
    global test_path

    def recognize_number(cur_num_file):
        cur_num = [(n[0], getLBPPropsVector(n[1], 8)) for n in cur_images if n[0] == cur_num_file]
        for t in cur_num:
            max_sim = 0
            for i in test_num_arr:
                v1 = i[1]
                v2 = t[1]
                if getCOSSimilarity(v1, v2) > max_sim:
                    max_sim = getCOSSimilarity(v1, v2)
                    s = f"{t[0]}, {i[0]} - {max_sim}"
                    return_val =((t[0], i[0][len(test_path):len(test_path)+2], max_sim))
            if max_sim < 0.991:
                return_string = 'не могу определить число'
            else:
                return_string = f'с вероятностью {round(max_sim,3)} это {return_val[1]}'
            print(f"{return_string}")
        return return_string

    def on_next_button_click():
        global item_
        global cur_num_arr
        global n1
        global n2
        if item_ < len(cur_num_arr)-1:
            n1 = f'{cur_num_arr[item_][0]}'
            # n2 = f'{tmp[item_][1]}'
            item_ = item_ + 1
        update_images()
        recognize_number(n1)
        label2.delete(0, tk.END)
        label2.insert(0, recognize_number(n1))

    def on_prev_button_click():
        global item_
        global cur_num_arr
        global n1
        global n2
        if item_ > 1:
            n1 = f'{cur_num_arr[item_][0]}'
            # n2 = f'{tmp[item_][1]}'
            item_ = item_ - 1
        update_images()
        label2.delete(0, tk.END)
        label2.insert(0,recognize_number(n1))

    def update_images():
        image1 = Image.open(n1).resize((64, 64))
        photo1 = ImageTk.PhotoImage(image1)
        label1.config(image=photo1)
        label1.image = photo1  # Сохраняем ссылку на изображение

    # Создаем основное окно
    root = tk.Tk()
    root.title("Распознаём число")

    image1 = Image.open(n1).resize((64, 64))  # Замените "image1.png" на путь к вашему изображению

    photo1 = ImageTk.PhotoImage(image1)

    # Создаем метки для изображения
    label1 = tk.Label(root, image=photo1)
    label1.pack(side="left", padx=10, pady=10)  # Размещаем первое изображение

    # поле для вывода результата распознавания
    label2 = tk.Entry(root, width=50)
    label2.pack(side="left", padx=10, pady=10)  #поле для вывода числа

    button = tk.Button(root, text="<<", command=on_prev_button_click)
    button.pack(pady=20)

    button = tk.Button(root, text=">>", command=on_next_button_click)
    button.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()