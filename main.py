from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
import os
from student import Student
from train import Train
from face_recognition import Face_Recognition
from attendance import Attendance
from developer import Developer
from helpsupport import Helpsupport

class Face_Recognition_System:
    def __init__(self, root):
        self.root = root
        self.root.attributes("-fullscreen", True)  # Start in full-screen mode
        self.root.title("Face Recognition System")

        # Dictionary to keep track of open windows
        self.open_windows = {}

        # Bind keys for toggling full-screen and minimizing
        self.root.bind("<F11>", self.toggle_fullscreen)
        self.root.bind("<Escape>", self.exit_fullscreen)

        # Get screen dimensions
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()

        # Header Image
        header_img = Image.open(r"C:\Users\ANIQUE\Documents\Python_Test_Projects\Images_GUI\banner.jpg")
        header_img = header_img.resize((self.screen_width, int(self.screen_height * 0.15)), Image.Resampling.LANCZOS)
        self.header_photo = ImageTk.PhotoImage(header_img)

        header_label = Label(self.root, image=self.header_photo)
        header_label.place(x=0, y=0, width=self.screen_width, height=int(self.screen_height * 0.15))

        # Background Image
        bg_img = Image.open(r"C:\Users\ANIQUE\Documents\Python_Test_Projects\Images_GUI\bg3.jpg")
        bg_img = bg_img.resize((self.screen_width, self.screen_height), Image.Resampling.LANCZOS)
        self.bg_photo = ImageTk.PhotoImage(bg_img)

        bg_label = Label(self.root, image=self.bg_photo)
        bg_label.place(x=0, y=int(self.screen_height * 0.15), width=self.screen_width, height=int(self.screen_height * 0.85))

        # Title
        title_label = Label(bg_label, text="Attendance Management System Using Facial Recognition",
                            font=("verdana", 33, "bold"), bg="white", fg="navyblue")
        title_label.place(x=0, y=0, width=self.screen_width, height=55)

        # Buttons
        self.create_buttons(bg_label)

    def create_buttons(self, bg_label):
        # Button data
        button_data = [
            ("Student Panel", r"C:\Users\ANIQUE\Documents\Python_Test_Projects\Images_GUI\std1.jpg", self.student_panel),
            ("Face Detector", r"C:\Users\ANIQUE\Documents\Python_Test_Projects\Images_GUI\det1.jpg", self.face_rec),
            ("Attendance", r"C:\Users\ANIQUE\Documents\Python_Test_Projects\Images_GUI\att.jpg", self.attendance_panel),
            ("Help Support", r"C:\Users\ANIQUE\Documents\Python_Test_Projects\Images_GUI\hlp.jpg", self.help_support),
            ("Data Train", r"C:\Users\ANIQUE\Documents\Python_Test_Projects\Images_GUI\tra1.jpg", self.train_panel),
            ("Database", r"C:\Users\ANIQUE\Documents\Python_Test_Projects\Images_GUI\db.jpg", self.open_img),
            ("Developers", r"C:\Users\ANIQUE\Documents\Python_Test_Projects\Images_GUI\dev.jpg", self.developers),
            ("Exit", r"C:\Users\ANIQUE\Documents\Python_Test_Projects\Images_GUI\ex.webp", self.close_system),
        ]

        # Define button sizes
        button_width = 190
        button_height = 195
        spacing_x = (self.screen_width - (button_width * 4)) // 5  # Space between buttons (horizontal)
        spacing_y = 90  # Space between rows (vertical)

        # Button placement
        for i, (text, img_path, command) in enumerate(button_data):
            col = i % 4
            row = i // 4
            x = spacing_x + col * (button_width + spacing_x)
            y = 100 + row * (button_height + spacing_y)

            # Load and resize images dynamically
            btn_img = Image.open(img_path)
            btn_img = btn_img.resize((button_width, button_height), Image.Resampling.LANCZOS)
            btn_photo = ImageTk.PhotoImage(btn_img)

            # Button with image
            Button(bg_label, image=btn_photo, cursor="hand2", command=command).place(x=x, y=y, width=button_width, height=button_height)

            # Button with text
            Button(bg_label, text=text, cursor="hand2", font=("tahoma", 15, "bold"), bg="white", fg="navyblue",
                   command=command).place(x=x, y=y + button_height + 5, width=button_width, height=46)

            # Retain a reference to images (prevent garbage collection)
            setattr(self, f"btn_photo_{i}", btn_photo)

    # Button Functions
    def open_img(self):
        os.startfile("data_img")

    def student_panel(self):
        self.open_or_raise_window("Student", Student)

    def train_panel(self):
        self.open_or_raise_window("Train", Train)

    def face_rec(self):
        self.open_or_raise_window("Face_Recognition", Face_Recognition)

    def attendance_panel(self):
        self.open_or_raise_window("Attendance", Attendance)

    def developers(self):
        self.open_or_raise_window("Developer", Developer)

    def help_support(self):
        self.open_or_raise_window("Helpsupport", Helpsupport)

    def open_or_raise_window(self, window_name, window_class):
        if window_name in self.open_windows:
            # Window is already open
            window = self.open_windows[window_name]
            if window.state() == "iconic":  # Check if the window is minimized
                window.deiconify()  # Restore the window
            window.lift()  # Bring the window to the front
        else:
            # Window is not open, create a new one
            self.new_window = Toplevel(self.root)
            self.app = window_class(self.new_window)
            self.open_windows[window_name] = self.new_window
            # Bind the close event to remove the window from the dictionary
            self.new_window.protocol("WM_DELETE_WINDOW", lambda: self.on_window_close(window_name))

    def on_window_close(self, window_name):
        # Remove the window from the dictionary when it is closed
        self.open_windows[window_name].destroy()
        del self.open_windows[window_name]

    def close_system(self):
        self.root.destroy()

    def toggle_fullscreen(self, event=None):
        """Toggle full-screen mode."""
        is_fullscreen = self.root.attributes("-fullscreen")
        self.root.attributes("-fullscreen", not is_fullscreen)

    def exit_fullscreen(self, event=None):
        """Exit full-screen mode."""
        self.root.attributes("-fullscreen", False)


if __name__ == "__main__":
    root = Tk()
    app = Face_Recognition_System(root)
    root.mainloop()