from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import messagebox
from register import Register
import mysql.connector
import os
from train import Train
from student import Student
from face_recognition import Face_Recognition
from attendance import Attendance
from developer import Developer
from helpsupport import Helpsupport


class Login:
    def __init__(self, root):
        self.root = root
        self.root.state('zoomed')
        self.root.title("Login")

        # Variables
        self.var_ssq = StringVar()
        self.var_sa = StringVar()
        self.var_pwd = StringVar()

        # Background Image
        self.bg = ImageTk.PhotoImage(file=r"C:\Users\ANIQUE\Documents\Python_Test_Projects\Images_GUI\loginBg1.jpg")
        lb1_bg = Label(self.root, image=self.bg)
        lb1_bg.place(x=0, y=0, relwidth=1, relheight=1)

        # Login Frame
        frame1 = Frame(self.root, bg="#002B53")
        frame1.place(x=560, y=170, width=340, height=450)

        # Logo
        img1 = Image.open(r"C:\Users\ANIQUE\Documents\Python_Test_Projects\Images_GUI\log1.png")
        img1 = img1.resize((100, 100), Image.Resampling.LANCZOS)
        self.photoimage1 = ImageTk.PhotoImage(img1)
        lb1img1 = Label(image=self.photoimage1, bg="#002B53")
        lb1img1.place(x=690, y=175, width=100, height=100)

        # Title
        get_str = Label(frame1, text="Login", font=("times new roman", 20, "bold"), fg="white", bg="#002B53")
        get_str.place(x=140, y=100)

        # Username
        username = Label(frame1, text="Username:", font=("times new roman", 15, "bold"), fg="white", bg="#002B53")
        username.place(x=30, y=160)
        self.txtuser = ttk.Entry(frame1, font=("times new roman", 15, "bold"))
        self.txtuser.place(x=33, y=190, width=270)

        # Password
        pwd = Label(frame1, text="Password:", font=("times new roman", 15, "bold"), fg="white", bg="#002B53")
        pwd.place(x=30, y=230)
        self.txtpwd = ttk.Entry(frame1, show="*", font=("times new roman", 15, "bold"))
        self.txtpwd.place(x=33, y=260, width=270)

        # Login Button
        loginbtn = Button(frame1, command=self.login, text="Login", font=("times new roman", 15, "bold"), bd=0,
                          relief=RIDGE, fg="#002B53", bg="white", activeforeground="white", activebackground="#007ACC")
        loginbtn.place(x=33, y=320, width=270, height=35)

        # Register Button
        regbtn = Button(frame1, command=self.open_register, text="Register", font=("times new roman", 10, "bold"), bd=0,
                         relief=RIDGE, fg="white", bg="#002B53", activeforeground="orange", activebackground="#002B53")
        regbtn.place(x=33, y=370, width=50, height=20)

        # Forget Password Button
        forgetbtn = Button(frame1, command=self.forget_pwd, text="Forget", font=("times new roman", 10, "bold"), bd=0,
                           relief=RIDGE, fg="white", bg="#002B53", activeforeground="orange", activebackground="#002B53")
        forgetbtn.place(x=90, y=370, width=50, height=20)

        # Track registration window
        self.register_window = None

    def open_register(self):
        """Open the registration window if it doesn't already exist."""
        if self.register_window is None or not self.register_window.winfo_exists():
            self.register_window = Toplevel(self.root)
            self.app = Register(self.register_window)
            self.register_window.protocol("WM_DELETE_WINDOW", self.on_register_close)
        else:
            self.register_window.lift()

    def on_register_close(self):
        """Handle the closing of the registration window."""
        self.register_window = None

    def login(self):
        """Handle login functionality."""
        if self.txtuser.get() == "" or self.txtpwd.get() == "":
            messagebox.showerror("Error", "All fields are required!")
        elif self.txtuser.get() == "admin" and self.txtpwd.get() == "admin":
            messagebox.showinfo("Success", "Welcome to Attendance Management System Using Facial Recognition")
            self.root.withdraw()  # Hide the login window
            self.open_main_system()  # Open the main system
        else:
            try:
                conn = mysql.connector.connect(username='root', password='admin', host='localhost',
                                               database='face_recognition', port=3306)
                mycursor = conn.cursor()
                mycursor.execute("SELECT * FROM regteach WHERE email=%s AND pwd=%s", (
                    self.txtuser.get(),
                    self.txtpwd.get()
                ))
                row = mycursor.fetchone()
                if row is None:
                    messagebox.showerror("Error", "Invalid Username or Password!")
                else:
                    open_min = messagebox.askyesno("YesNo", "Access only Admin")
                    if open_min:
                        self.root.withdraw()  # Hide the login window
                        self.open_main_system()  # Open the main system
            except Exception as es:
                messagebox.showerror("Error", f"Due to: {str(es)}")
            finally:
                conn.close()

    def open_main_system(self):
        """Open the main system by calling main.py."""
        import subprocess
        subprocess.Popen(["python", "main.py"])  # Replace "main.py" with the correct path to your main.py file
        self.root.destroy()  # Close the login window after launching main.py

    def reset_pass(self):
        """Reset password functionality."""
        if self.var_ssq.get() == "Select":
            messagebox.showerror("Error", "Select the Security Question!", parent=self.root2)
        elif self.var_sa.get() == "":
            messagebox.showerror("Error", "Please enter the Answer!", parent=self.root2)
        elif self.var_pwd.get() == "":
            messagebox.showerror("Error", "Please enter the New Password!", parent=self.root2)
        else:
            try:
                conn = mysql.connector.connect(username='root', password='admin', host='localhost',
                                            database='face_recognition', port=3306)
                mycursor = conn.cursor()
                query = "SELECT * FROM regteach WHERE email=%s AND ssq=%s AND sa=%s"
                value = (self.txtuser.get(), self.var_ssq.get(), self.var_sa.get())
                mycursor.execute(query, value)
                row = mycursor.fetchone()
                if row is None:
                    messagebox.showerror("Error", "Please enter the correct Answer!", parent=self.root2)
                else:
                    query = "UPDATE regteach SET pwd=%s WHERE email=%s"
                    value = (self.var_pwd.get(), self.txtuser.get())
                    mycursor.execute(query, value)
                    conn.commit()
                    messagebox.showinfo("Info", "Successfully reset your password. Please login with the new password!",
                                    parent=self.root2)
                    self.root2.destroy()  # Close the forget password window
                    self.refresh_login_page()  # Refresh the login page
            except Exception as es:
                messagebox.showerror("Error", f"Due to: {str(es)}", parent=self.root2)
            finally:
                conn.close()

    def refresh_login_page(self):
        """Refresh the login page by clearing all fields and focusing on the username field."""
        self.txtuser.delete(0, END)  # Clear the username field
        self.txtpwd.delete(0, END)  # Clear the password field
        self.txtuser.focus_set()  # Set focus back to the username field
    def forget_pwd(self):
        """Open the forget password window."""
        if self.txtuser.get() == "":
            messagebox.showerror("Error", "Please enter the Email ID to reset Password!")
        else:
            try:
                conn = mysql.connector.connect(username='root', password='admin', host='localhost',
                                              database='face_recognition', port=3306)
                mycursor = conn.cursor()
                query = "SELECT * FROM regteach WHERE email=%s"
                value = (self.txtuser.get(),)
                mycursor.execute(query, value)
                row = mycursor.fetchone()
                if row is None:
                    messagebox.showerror("Error", "Please enter a valid Email ID!")
                else:
                    self.root2 = Toplevel()
                    self.root2.title("Forget Password")
                    self.root2.geometry("400x400+610+170")
                    l = Label(self.root2, text="Forget Password", font=("times new roman", 30, "bold"), fg="#002B53",
                              bg="#fff")
                    l.place(x=0, y=10, relwidth=1)

                    # Security Question
                    ssq = Label(self.root2, text="Select Security Question:", font=("times new roman", 15, "bold"),
                                fg="#002B53", bg="#F2F2F2")
                    ssq.place(x=70, y=80)
                    self.combo_security = ttk.Combobox(self.root2, textvariable=self.var_ssq,
                                                       font=("times new roman", 15, "bold"), state="readonly")
                    self.combo_security["values"] = ("Select", "Your Date of Birth", "Your Nick Name", "Your Favorite Book")
                    self.combo_security.current(0)
                    self.combo_security.place(x=70, y=110, width=270)

                    # Security Answer
                    sa = Label(self.root2, text="Security Answer:", font=("times new roman", 15, "bold"), fg="#002B53",
                               bg="#F2F2F2")
                    sa.place(x=70, y=150)
                    self.txtpwd = ttk.Entry(self.root2, textvariable=self.var_sa, font=("times new roman", 15, "bold"))
                    self.txtpwd.place(x=70, y=180, width=270)

                    # New Password
                    new_pwd = Label(self.root2, text="New Password:", font=("times new roman", 15, "bold"), fg="#002B53",
                                    bg="#F2F2F2")
                    new_pwd.place(x=70, y=220)
                    self.new_pwd = ttk.Entry(self.root2, textvariable=self.var_pwd, font=("times new roman", 15, "bold"))
                    self.new_pwd.place(x=70, y=250, width=270)

                    # Reset Password Button
                    reset_btn = Button(self.root2, command=self.reset_pass, text="Reset Password",
                                       font=("times new roman", 15, "bold"), bd=0, relief=RIDGE, fg="#fff", bg="#002B53",
                                       activeforeground="white", activebackground="#007ACC")
                    reset_btn.place(x=70, y=300, width=270, height=35)
            except Exception as es:
                messagebox.showerror("Error", f"Due to: {str(es)}")
            finally:
                conn.close()
if __name__ == "__main__":
    root = Tk()
    app = Login(root)
    root.mainloop()