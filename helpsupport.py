from tkinter import *
from PIL import Image, ImageTk
import webbrowser


class Helpsupport:
    def __init__(self, root):
        self.root = root
        self.root.state('zoomed')
        self.root.title("Face Recognition System")

        # First header image
        img = Image.open(r"C:\Users\ANIQUE\Documents\Python_Test_Projects\Images_GUI\banner.jpg")
        img = img.resize((1566, 150), Image.Resampling.LANCZOS)
        self.photoimg = ImageTk.PhotoImage(img)

        # Set image as label
        f_lb1 = Label(self.root, image=self.photoimg)
        f_lb1.place(relx=0, rely=0, relwidth=1, height=150)

        # Background image
        bg1 = Image.open(r"C:\Users\ANIQUE\Documents\Python_Test_Projects\Images_GUI\bg4.png")
        bg1 = bg1.resize((1566, 799), Image.Resampling.LANCZOS)
        self.photobg1 = ImageTk.PhotoImage(bg1)

        # Set image as label
        bg_img = Label(self.root, image=self.photobg1)
        bg_img.place(relx=0, rely=0.17, relwidth=1, relheight=0.83)

        # Title section
        title_lb1 = Label(bg_img, text="Help Support", font=("verdana", 30, "bold"), bg="white", fg="navyblue")
        title_lb1.place(relx=0, rely=0, relwidth=1, height=55)

        # Button configuration
        button_config = [
            ("Website", self.website, r"C:\Users\ANIQUE\Documents\Python_Test_Projects\Images_GUI\wb.png"),
            ("Facebook", self.facebook, r"C:\Users\ANIQUE\Documents\Python_Test_Projects\Images_GUI\fb.jpg"),
            ("Youtube", self.youtube, r"C:\Users\ANIQUE\Documents\Python_Test_Projects\Images_GUI\ytbt.png"),
            ("Gmail", self.gmail, r"C:\Users\ANIQUE\Documents\Python_Test_Projects\Images_GUI\MAIL.webp"),
        ]

        for idx, (text, command, img_path) in enumerate(button_config):
            # Load button image
            btn_img = Image.open(img_path)
            btn_img = btn_img.resize((180, 180), Image.Resampling.LANCZOS)
            photo_img = ImageTk.PhotoImage(btn_img)
            setattr(self, f"btn_img_{idx}", photo_img)  # Store reference to avoid garbage collection

            # Place button image
            btn = Button(bg_img, image=photo_img, command=command, cursor="hand2")
            btn.place(relx=0.18 + idx * 0.18, rely=0.25, relwidth=0.12, relheight=0.3)

            # Place button text
            btn_text = Button(bg_img, text=text, command=command, cursor="hand2", font=("tahoma", 15, "bold"), bg="white", fg="navyblue")
            btn_text.place(relx=0.18 + idx * 0.18, rely=0.57, relwidth=0.12, height=45)

    # Button functions
    def website(self):
        webbrowser.open("https://en.wikipedia.org/wiki/Facial_recognition_system#:~:text=These%20claims%20have%20led%20to,more%20than%20one%20billion%20users.")

    def facebook(self):
        webbrowser.open("https://www.facebook.com/")

    def youtube(self):
        webbrowser.open("https://www.youtube.com/watch?v=9H72QbYqxXU")

    def gmail(self):
        webbrowser.open("https://mail.google.com/mail/u/0/#inbox?compose=VpCqJXKRzSJzVflcPXXMfChLMSFLfCmsddlNDdmZRzpQwvrwqKbXCWWKhhZgRCXLnszVnrv")


if __name__ == "__main__":
    root = Tk()
    obj = Helpsupport(root)
    root.mainloop()
