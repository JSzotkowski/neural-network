import numpy as np

from NeuralNetwork import NeuralNetwork
import tkinter as tk
from load_digit import load_digit

"""entire file is GPT stuff tbh"""


def paint(event):
    """GPT stuff"""
    x1, y1 = (event.x - 2), (event.y - 2)
    x2, y2 = (event.x + 2), (event.y + 2)
    canvas.create_oval(x1, y1, x2, y2, fill="black", width=5)
    for i in range(-2, 3):
        for j in range(-2, 3):
            x, y = event.x + i, event.y + j
            x = x // 10
            y = y // 10
            canvas_array[y][x] = 1
    print(np.argmax(net.feedforward(np.reshape(canvas_array, (784, 1)))))


def clear_canvas():
    """GPT stuff"""
    canvas.delete("all")
    global canvas_array
    canvas_array = [[0 for x in range(28)] for y in range(28)]


net = NeuralNetwork([784, 30, 10])
net.init_from_json('/home/jiri/PycharmProjects/NeuralNetwork/runs/mon_02_october_09_35_26/trained_network_SGD.txt')
my_digit = load_digit()
print(np.argmax(net.feedforward(my_digit)))

root = tk.Tk()

close_button = tk.Button(root, text="Clear", command=clear_canvas)
close_button.pack(side=tk.LEFT, padx=10, pady=10)

canvas = tk.Canvas(root, bg="white", height=280, width=280)
canvas.pack(expand=True, fill="both")
canvas.bind("<B1-Motion>", paint)

canvas_array = [[0 for x in range(28)] for y in range(28)]

root.mainloop()
