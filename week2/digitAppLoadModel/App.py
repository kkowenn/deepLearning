from keras.models import load_model
import tkinter as tk
from PIL import ImageGrab, Image
import numpy as np

# Load the pre-trained model
model = load_model('/Users/kritsadakruapat/Desktop/Collage/CSX4208DL/week2/digitAppLoadModel/mnist_improved.h5')

def predict_digit(img):
    # Resize image to 28x28 pixels
    img = img.resize((28, 28))
    # Convert RGB to grayscale
    img = img.convert('L')
    img = np.array(img)
    # Reshape for model normalization
    img = img.reshape(1, 28, 28, 1)
    img = img / 255.0
    # Predicting the class
    res = model.predict([img])[0]
    return np.argmax(res), max(res)

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        # Creating elements
        self.canvas = tk.Canvas(self, width=200, height=200, bg="white", cursor="cross")
        self.label = tk.Label(self, text="Analyzing...", font=("Helvetica", 24))
        self.classify_btn = tk.Button(self, text="Predict", command=self.classify_handwriting)
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_all)
        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=tk.W)
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")

    def classify_handwriting(self):
        # Get the handle of the canvas
        x = self.winfo_rootx() + self.canvas.winfo_x()
        y = self.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        # Grab image from canvas
        im = ImageGrab.grab().crop((x, y, x1, y1))
        digit, acc = predict_digit(im)
        self.label.configure(text=str(digit) + ', ' + str(int(acc * 100)) + '%')

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 12
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill='black')

if __name__ == "__main__":
    app = App()
    app.mainloop()
