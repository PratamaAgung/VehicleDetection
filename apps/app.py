import tkinter
import cv2
from tkinter import filedialog

class App:
    def __init__(self):
        self.root = tkinter.Tk()
        self.screen_w, self.screen_h = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry("%dx%d+0+0" % (self.screen_w, self.screen_h))

        self.init_gui(window_title="Vehicle Detection")
        self.root.mainloop()

    def init_gui(self, window_title):
        self.root.title(window_title)
        self.video_path_label = tkinter.Text(master=self.root, state="disabled", width=100, height=2)
        self.video_path_label.grid(row=0, column=1)
        self.get_vid_path_button = tkinter.Button(text="Browse", command=self.set_video_path)
        self.get_vid_path_button.grid(row=0, column=2)

    def set_video_path(self):
        file_name = filedialog.askdirectory()

        self.video_path_label.configure(state="normal")
        self.video_path_label.delete(1.0, tkinter.END)
        self.video_path_label.insert(tkinter.END, file_name)
        self.video_path_label.configure(state='disabled')

    def get_video_path(self):
        return self.video_path_label.get(1.0, tkinter.END)



if __name__ == "__main__":
    app = App()