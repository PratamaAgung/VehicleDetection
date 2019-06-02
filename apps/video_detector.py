from tkinter import Tk, Text, BOTH, W, N, E, S, filedialog, END, BooleanVar, Checkbutton, Entry, StringVar
from tkinter.ttk import Frame, Button, Label, Style
import cv2
import threading
import PIL.Image, PIL.ImageTk

class Detector(Frame):

    def __init__(self):
        super().__init__()

        self.initUI()
        self.video = None


    def initUI(self):

        self.master.title("Video Detector")
        self.pack(fill=BOTH, expand=True)

        self.columnconfigure(2, weight=1)
        self.columnconfigure(3, pad=7)
        self.rowconfigure(9, weight=1)
        self.rowconfigure(10, pad=7)

        path_lbl = Label(self, text="Video Source")
        path_lbl.grid(sticky=W, pady=4, padx=5)

        self.video_filepath = Text(self, height=1, padx=5, state='disabled')
        self.video_filepath.grid(row=0, column=1, sticky=W) 

        self.set_vid_path_button = Button(self, text="Browse", command=self.set_video_path)
        self.set_vid_path_button.grid(row=0, column=2, sticky=W, padx=5)

        self.video_area = Label(self, borderwidth=1, relief='ridge')
        self.video_area.grid(row=1, column=0, columnspan=3, rowspan=9, padx=5, sticky=E+W+S+N)

        show_label = Label(self, text="Show")
        show_label.grid(row=1, column=3, sticky=W, columnspan=2)

        self.show_det_area = BooleanVar()
        self.show_det_area.set(True)
        Checkbutton(self, text="Detection Area", variable=self.show_det_area).grid(row=2, column=3, sticky=W, columnspan=2)

        self.show_bb = BooleanVar()
        self.show_bb.set(True)
        Checkbutton(self, text="Bounding Box", variable=self.show_bb).grid(row=3, column=3, sticky=W, columnspan=2)

        det_label = Label(self, text="Detection Area")
        det_label.grid(row=4, column=3, sticky=W, pady=5, columnspan=2)

        self.min_x = StringVar()
        self.min_x.set("0.0")
        min_x_label = Label(self, text="Min X").grid(row=5, column=3, sticky=W)
        min_x_text = Entry(self, width=5, textvariable=self.min_x).grid(row=5, column=4, sticky=W)

        self.max_x = StringVar()
        self.max_x.set("1.0")
        max_x_label = Label(self, text="Max X").grid(row=6, column=3, sticky=W)
        max_x_text = Entry(self, width=5, textvariable= self.max_x).grid(row=6, column=4, sticky=W)

        self.min_y = StringVar()
        self.min_y.set("0.0")
        min_y_label = Label(self, text="Min Y").grid(row=7, column=3, sticky=W)
        min_y_text = Entry(self, width=5, textvariable=self.min_y).grid(row=7, column=4, sticky=W)

        self.max_y = StringVar()
        self.max_y.set("0.9")
        max_y_label = Label(self, text="Max Y").grid(row=8, column=3, sticky=W)
        max_y_text = Entry(self, width=5, textvariable=self.max_y).grid(row=8, column=4, sticky=W)

        self.ctrl_btn = Button(self, text="Play", command=self.play_video)
        self.ctrl_btn.grid(row=10, column=0, padx=5)

        self.close_btn = Button(self, text="Close")
        self.close_btn.grid(row=10, column=3, columnspan=2)

    def set_video_path(self):
        file_name = filedialog.askopenfilename()

        self.video_filepath.configure(state="normal")
        self.set_input(self.video_filepath, file_name)
        self.video_filepath.configure(state='disabled')

    def set_input(self, text_widget, text):
        text_widget.delete(1.0, END)
        text_widget.insert(END, text)

    def get_video_path(self):
        return self.video_filepath.get(1.0, END)[:-1]

    def play_video(self):
        if not self.video:
            file_name = self.get_video_path()
            if file_name and len(file_name) > 0:
                self.video = cv2.VideoCapture(file_name)
                self.is_stopped = False
                if not self.video.isOpened():
                    raise ValueError("Unable to open video source", file_name)
                
                self.ctrl_btn.configure(text='Stop')
                self.ctrl_btn.configure(command=self.stop_video)
                self.play_thread = threading.Thread(target=self.thread_play_video, args=())
                self.play_thread.start()
            else:
                print("Video path not set")

    def stop_video(self):
        self.is_stopped = True
        self.ctrl_btn.configure(text='Start')
        self.ctrl_btn.configure(command=self.play_video)

    def thread_play_video(self):
        if self.video.isOpened():
            while not self.is_stopped:
                ret, frame = self.video.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    label_h, label_w = self.video_area.winfo_height(), self.video_area.winfo_width()
                    frame = cv2.resize(frame, (label_w, label_h))
                    tk_image = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
                    self.video_area.configure(image=tk_image)
                else:
                    continue
            self.video.release()
            self.video = None


def main():

    root = Tk()
    root.attributes('-fullscreen', True)
    app = Detector()
    root.mainloop()


if __name__ == '__main__':
    main()