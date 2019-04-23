from tkinter import *
from GAN_v1 import *
import threading


def synchronize(tid, func):
    thd = MyThread(tid, func)
    thd.start()


class MyThread(threading.Thread):
    def __init__(self, tid, func):
        threading.Thread.__init__(self)
        self.threadID = tid
        self.func = func

    def run(self):
        self.func()


class Window:
    def __init__(self):
        self.proc = GAN()
        self.win = Tk()

        self.widget = []

        self.win_main()

    def unable_widget(self):
        for n in self.widget:
            n['state'] = DISABLED

    def enable_widget(self):
        for n in self.widget:
            n['state'] = NORMAL

    def train(self):
        self.unable_widget()
        self.proc.train()
        self.enable_widget()

    def test(self):
        self.unable_widget()
        self.proc.generate()
        self.enable_widget()

    def func1(self):
        synchronize(1, self.train)

    def func2(self):
        synchronize(2, self.test)

    def load_widget(self):
        btn1 = Button(self.win, text="点我来训练", command=self.func1)
        btn1.place(x=160, y=200)
        self.widget.append(btn1)
        btn2 = Button(self.win, text="点我来测试", command=self.func2)
        btn2.place(x=360, y=200)
        self.widget.append(btn2)

    def win_proc(self):
        self.win.geometry("600x480+200+200")
        self.load_widget()

    def win_main(self):
        self.win_proc()
        self.win.mainloop()


if __name__ == "__main__":
    Window()
