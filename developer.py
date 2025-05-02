from tkinter import *
from PIL import Image, ImageTk


class Developer:
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
        bg1 = bg1.resize((1566, 778), Image.Resampling.LANCZOS)
        self.photobg1 = ImageTk.PhotoImage(bg1)

        # Set image as label
        bg_img = Label(self.root, image=self.photobg1)
        bg_img.place(relx=0, rely=0.17, relwidth=1, relheight=0.83)

        # Title section
        title_lb1 = Label(bg_img, text="Developer Panel", font=("verdana", 33, "bold"), bg="white", fg="navyblue")
        title_lb1.place(relx=0, rely=0, relwidth=1, height=55)

        # Button configuration
        developers = [
            ("ALHAN SHAIKH", r"C:\Users\ANIQUE\Documents\Python_Test_Projects\Images_GUI\alhan.jpg"),
            ("AYAAN KHAN", r"C:\Users\ANIQUE\Documents\Python_Test_Projects\Images_GUI\ayaan.jpg"),
            ("ANIQUE SHAIKH", r"C:\Users\ANIQUE\Documents\Python_Test_Projects\Images_GUI\anique.jpg"),
            ("ZAID QAZIM", r"C:\Users\ANIQUE\Documents\Python_Test_Projects\Images_GUI\log.jpg"),
        ]

        for idx, (name, img_path) in enumerate(developers):
            # Load button image
            dev_img = Image.open(img_path)
            dev_img = dev_img.resize((200, 200), Image.Resampling.LANCZOS)
            photo_img = ImageTk.PhotoImage(dev_img)
            setattr(self, f"dev_img_{idx}", photo_img)  # Store reference to avoid garbage collection

            # Place button image
            btn = Button(bg_img, image=photo_img, cursor="hand2")
            btn.place(relx=0.18 + idx * 0.18, rely=0.25, relwidth=0.13, relheight=0.3)

            # Place button text
            btn_text = Button(bg_img, text=name, cursor="hand2", font=("tahoma", 15, "bold"), bg="white", fg="navyblue")
            btn_text.place(relx=0.18 + idx * 0.18, rely=0.57, relwidth=0.13, height=45)


if __name__ == "__main__":
    root = Tk()
    obj = Developer(root)
    root.mainloop()
