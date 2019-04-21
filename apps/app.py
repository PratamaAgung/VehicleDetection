import tkinter
import cv2
from tkinter import filedialog
import threading
import PIL.Image, PIL.ImageTk

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
        self.video_path_label.grid(row=0, column=0)

        self.set_vid_path_button = tkinter.Button(text="Browse", command=self.set_video_path)
        self.set_vid_path_button.grid(row=0, column=1)

        self.play_video_button = tkinter.Button(text="Play", command=self.play_video)
        self.play_video_button.grid(row=0, column=2)
        self.video = None

        self.pause_video_button = tkinter.Button(text="Pause", command=self.pause_video)
        self.pause_video_button.grid(row=0, column=3)
        self.is_paused = False

        self.video_canvas = tkinter.Label(image=None)
        self.video_canvas.grid(row=1, column=0)

    def set_video_path(self):
        file_name = filedialog.askopenfilename()

        self.video_path_label.configure(state="normal")
        self.video_path_label.delete(1.0, tkinter.END)
        self.video_path_label.insert(tkinter.END, file_name)
        self.video_path_label.configure(state='disabled')

    def get_video_path(self):
        return self.video_path_label.get(1.0, tkinter.END)

    def play_video(self):
        if not self.video:
            file_name = self.get_video_path()
            self.video = cv2.VideoCapture(file_name[:-1])
            if not self.video.isOpened():
                raise ValueError("Unable to open video source", file_name)
            
            self.play_thread = threading.Thread(target=self.thread_play_video, args=())
            self.play_thread.start()
        else:
            self.is_paused = False

    def pause_video(self):
        self.is_paused = True

    def thread_play_video(self):
        if self.video.isOpened():
            is_playing = True
            while is_playing:
                if not self.is_paused:
                    ret, frame = self.video.read()
                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = cv2.resize(frame, (480, 270))
                        tk_image = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
                        self.video_canvas.configure(image=tk_image)
                        self.video_canvas.image = tk_image
                    else:
                        is_playing = False
                else:
                    continue
        



if __name__ == "__main__":
    app = App()