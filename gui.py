from tkinter import *

def button1():
    novi = Toplevel()
    canvas = Canvas(novi, width = 900, height = 900)
    canvas.pack(expand = YES, fill = BOTH)
    gif1 = PhotoImage(file = 'BTC-USD-result.png')
                                #image not visual
    canvas.create_image(50, 10, image = gif1, anchor = NW)
    #assigned the gif1 to the canvas object
    canvas.gif1 = gif1

def button2():
    novi = Toplevel()
    canvas = Canvas(novi, width = 900, height = 900)
    canvas.pack(expand = YES, fill = BOTH)
    gif2 = PhotoImage(file = 'BCH-USD-result.png')
                                #image not visual
    canvas.create_image(50, 10, image = gif2, anchor = NW)
    #assigned the gif1 to the canvas object
    canvas.gif2 = gif2

def button3():
    novi = Toplevel()
    canvas = Canvas(novi, width = 900, height = 900)
    canvas.pack(expand = YES, fill = BOTH)
    gif3 = PhotoImage(file = 'ETH-USD-result.png')
                                #image not visual
    canvas.create_image(50, 10, image = gif3, anchor = NW)
    #assigned the gif1 to the canvas object
    canvas.gif3 = gif3

def button4():
    novi = Toplevel()
    canvas = Canvas(novi, width = 900, height = 900)
    canvas.pack(expand = YES, fill = BOTH)
    gif4 = PhotoImage(file = 'LTC-USD-result.png')
                                #image not visual
    canvas.create_image(50, 10, image = gif4, anchor = NW)
    #assigned the gif1 to the canvas object
    canvas.gif4 = gif4

mGui = Tk()


button1 = Button(mGui,text ='Bitcoin',command = button1, height=5, width=20).pack()
button2 = Button(mGui,text ='Bitcoin Cash',command = button2, height=5, width=20).pack()
button3 = Button(mGui,text ='Ethereum',command = button3, height=5, width=20).pack()
button4 = Button(mGui,text ='Litecoin',command = button4, height=5, width=20).pack()
button5 = Button(mGui, text = "EXIT", command = mGui.destroy, height=5, width=20).pack()



mGui.mainloop()
