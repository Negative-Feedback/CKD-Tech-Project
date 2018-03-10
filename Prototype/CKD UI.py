from tkinter import *

def sel():
   selection = "You selected the option " + str(var.get())
   label.config(text = selection)

root = Tk()

root.geometry("700x400")
root.resizable(0, 0)
root.title("Chronic Kidney Disease - Prototype")
var = IntVar()
R1 = Radiobutton(root, text="Decision Tree", variable=var, value=1,
                  command=sel)
R1.pack( anchor = W )

R2 = Radiobutton(root, text="KNN", variable=var, value=2,
                  command=sel)
R2.pack( anchor = W )

R3 = Radiobutton(root, text="LR", variable=var, value=3,
                  command=sel)
R3.pack( anchor = W )

R4 = Radiobutton(root, text="NN", variable=var, value=4,
                  command=sel)
R4.pack( anchor = W)
R5 = Radiobutton(root, text="RF", variable=var, value=5,
                  command=sel)
R5.pack( anchor = W)
R6 = Radiobutton(root, text="SVM", variable=var, value=6,
                  command=sel)
R6.pack( anchor = W)

label = Label(root)
label.pack()
root.mainloop()