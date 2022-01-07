from tkinter import *
from tkinter import messagebox
from Iris_training import *

app = Tk()

app.title("Iris")
app.geometry("600x600")


def checkClasses():
    global c1

    global c2
    if check_C1.get() == 1 and check_C2.get() == 1 and check_C3.get() == 0:
        c1 = 0
        c2 = 1
        return True
    elif check_C1.get() == 1 and check_C2.get() == 0 and check_C3.get() == 1:
        c1 = 0
        c2 = 2
        return True
    elif check_C1.get() == 0 and check_C2.get() == 1 and check_C3.get() == 1:
        c1 = 1
        c2 = 2
        return True
    return False


def startCallBack():
    print(check_C1.get(), check_C2.get(), check_C3.get())
    # Check the selected Features
    if check_X1.get() == 1 and check_X2.get() == 1 and check_X3.get() == 0 and check_X4.get() == 0:
        if checkClasses():
            feature1 = 0
            feature2 = 1
            setClassesAndFeatures(c1, c2, feature1, feature2)
    elif check_X1.get() == 0 and check_X2.get() == 1 and check_X3.get() == 1 and check_X4.get() == 0:
        if checkClasses():
            feature1 = 1
            feature2 = 2
            setClassesAndFeatures(c1, c2, feature1, feature2)
    elif check_X1.get() == 0 and check_X2.get() == 0 and check_X3.get() == 1 and check_X4.get() == 1:
        if checkClasses():
            feature1 = 2
            feature2 = 3
            setClassesAndFeatures(c1, c2, feature1, feature2)
    elif check_X1.get() == 1 and check_X2.get() == 0 and check_X3.get() == 1 and check_X4.get() == 0:
        if checkClasses():
            feature1 = 0
            feature2 = 2
            setClassesAndFeatures(c1, c2, feature1, feature2)
    elif check_X1.get() == 0 and check_X2.get() == 1 and check_X3.get() == 0 and check_X4.get() == 1:
        if checkClasses():
            feature1 = 1
            feature2 = 3
            setClassesAndFeatures(c1, c2, feature1, feature2)
    elif check_X1.get() == 1 and check_X2.get() == 0 and check_X3.get() == 0 and check_X4.get() == 1:
        if checkClasses():
            feature1 = 0
            feature2 = 3
            setClassesAndFeatures(c1, c2, feature1, feature2)
    else:
        msg = messagebox.showinfo("Info", "Please Choose only 2 classes and 2 features and epochs less than 100")
        return
    if not checkClasses():
        msg = messagebox.showinfo("Info", "Please Choose only 2 classes and 2 features and epochs less than 100")
        return
    if 100 >= int(e_epochs.get()) > 0:
        finalize_data(e_learn.get(), e_epochs.get(), check_bais.get())
        read_data()


var_features = StringVar()
l_features = Label(app, textvariable=var_features)
var_features.set("Choose only 2 Features")
l_features.grid(row=0, column=0, sticky=W)

check_X1 = IntVar()
Checkbutton(app, text="X1", variable=check_X1).grid(row=3, column=0, sticky=W)
check_X2 = IntVar()
Checkbutton(app, text="X2", variable=check_X2).grid(row=4, column=0, sticky=W)

check_X3 = IntVar()
Checkbutton(app, text="X3", variable=check_X3).grid(row=5, column=0, sticky=W)
check_X4 = IntVar()
Checkbutton(app, text="X4", variable=check_X4).grid(row=6, column=0, sticky=W)

var_Classes = StringVar()
l_Classes = Label(app, textvariable=var_Classes)
var_Classes.set("Choose only 2 Classes")
l_Classes.grid(row=7, column=0, sticky=W)

check_C1 = IntVar()
Checkbutton(app, text="C1", variable=check_C1).grid(row=9, column=0, sticky=W)
check_C2 = IntVar()
Checkbutton(app, text="C2", variable=check_C2).grid(row=10, column=0, sticky=W)

check_C3 = IntVar()
Checkbutton(app, text="C3", variable=check_C3).grid(row=11, column=0, sticky=W)

var_learn = StringVar()
l_learn = Label(app, textvariable=var_learn)
var_learn.set("Enter Learning Rate")
l_learn.grid(row=12, column=0, sticky=W)
e_learn = Entry(app)
e_learn.grid(row=13, column=0, sticky=W)

var_epochs = StringVar()
l_epochs = Label(app, textvariable=var_epochs)
var_epochs.set("Enter Epochs")
l_epochs.grid(row=14, column=0, sticky=W)
e_epochs = Entry(app)
e_epochs.grid(row=15, column=0, sticky=W)

check_bais = IntVar()
Checkbutton(app, text="Bias", variable=check_bais).grid(row=16, column=0, sticky=W)

btn_start = Button(app, text="Start", command=startCallBack)
btn_start.grid(row=17, column=0, sticky=W)
app.mainloop()
