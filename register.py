from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import messagebox
import mysql.connector
import subprocess
import sys

class Register:
    def __init__(self, root):
        self.root = root
        self.root.state('zoomed')
        self.root.title("Register")
        

        # ============ Variables =================
        self.var_fname = StringVar()
        self.var_lname = StringVar()
        self.var_cnum = StringVar()
        self.var_email = StringVar()
        self.var_ssq = StringVar()
        self.var_sa = StringVar()
        self.var_pwd = StringVar()
        self.var_cpwd = StringVar()
        self.var_check = IntVar()

        self.bg = ImageTk.PhotoImage(file=r"C:\Users\ANIQUE\Documents\Python_Test_Projects\Images_GUI\bgReg.jpg")
        
        lb1_bg = Label(self.root, image=self.bg)
        lb1_bg.place(x=0, y=0, relwidth=1, relheight=1)

        frame = Frame(self.root, bg="#F2F2F2")
        frame.place(x=100, y=80, width=900, height=580)

        get_str = Label(frame, text="Registration", font=("times new roman", 30, "bold"), fg="#002B53", bg="#F2F2F2")
        get_str.place(x=350, y=130)

        # First Name
        Label(frame, text="First Name:", font=("times new roman", 15, "bold"), fg="#002B53", bg="#F2F2F2").place(x=100, y=200)
        self.txt_fname = ttk.Entry(frame, textvariable=self.var_fname, font=("times new roman", 15, "bold"))
        self.txt_fname.place(x=103, y=225, width=270)

        # Last Name
        Label(frame, text="Last Name:", font=("times new roman", 15, "bold"), fg="#002B53", bg="#F2F2F2").place(x=100, y=270)
        self.txt_lname = ttk.Entry(frame, textvariable=self.var_lname, font=("times new roman", 15, "bold"))
        self.txt_lname.place(x=103, y=295, width=270)

        # Contact No
        Label(frame, text="Contact No:", font=("times new roman", 15, "bold"), fg="#002B53", bg="#F2F2F2").place(x=530, y=200)
        self.txt_cnum = ttk.Entry(frame, textvariable=self.var_cnum, font=("times new roman", 15, "bold"))
        self.txt_cnum.place(x=533, y=225, width=270)

        # Email
        Label(frame, text="Email:", font=("times new roman", 15, "bold"), fg="#002B53", bg="#F2F2F2").place(x=530, y=270)
        self.txt_email = ttk.Entry(frame, textvariable=self.var_email, font=("times new roman", 15, "bold"))
        self.txt_email.place(x=533, y=295, width=270)

        # Security Question
        Label(frame, text="Select Security Question:", font=("times new roman", 15, "bold"), fg="#002B53", bg="#F2F2F2").place(x=100, y=350)
        self.combo_security = ttk.Combobox(frame, textvariable=self.var_ssq, font=("times new roman", 15, "bold"), state="readonly")
        self.combo_security["values"] = ("Select", "Your Date of Birth", "Your Nick Name", "Your Favorite Book")
        self.combo_security.current(0)
        self.combo_security.place(x=103, y=375, width=270)

        # Security Answer
        Label(frame, text="Security Answer:", font=("times new roman", 15, "bold"), fg="#002B53", bg="#F2F2F2").place(x=100, y=420)
        self.txt_sa = ttk.Entry(frame, textvariable=self.var_sa, font=("times new roman", 15, "bold"))
        self.txt_sa.place(x=103, y=445, width=270)

        # Password
        Label(frame, text="Password:", font=("times new roman", 15, "bold"), fg="#002B53", bg="#F2F2F2").place(x=530, y=350)
        self.txt_pwd = ttk.Entry(frame, textvariable=self.var_pwd, show='*', font=("times new roman", 15, "bold"))
        self.txt_pwd.place(x=533, y=375, width=270)

        # Confirm Password
        Label(frame, text="Confirm Password:", font=("times new roman", 15, "bold"), fg="#002B53", bg="#F2F2F2").place(x=530, y=420)
        self.txt_cpwd = ttk.Entry(frame, textvariable=self.var_cpwd, show='*', font=("times new roman", 15, "bold"))
        self.txt_cpwd.place(x=533, y=445, width=270)

        # Checkbutton
        checkbtn = Checkbutton(frame, variable=self.var_check, text="I Agree to the Terms & Conditions", font=("times new roman", 15, "bold"), fg="#002B53", bg="#F2F2F2")
        checkbtn.place(x=280, y=485, width=350)

        # Register Button
        loginbtn = Button(frame, command=self.reg, text="Register", font=("times new roman", 15, "bold"), bd=0, relief=RIDGE, fg="#fff", bg="#002B53", activeforeground="white", activebackground="#007ACC")
        loginbtn.place(x=320, y=530, width=270, height=35)

    def reg(self):
        if (self.var_fname.get() == "" or self.var_lname.get() == "" or self.var_cnum.get() == "" or self.var_email.get() == "" or self.var_ssq.get() == "Select" or self.var_sa.get() == "" or self.var_pwd.get() == "" or self.var_cpwd.get() == ""):
            messagebox.showerror("Error", "All fields are required!")
        elif (self.var_pwd.get() != self.var_cpwd.get()):
            messagebox.showerror("Error", "Password and Confirm Password must be the same!")
        elif (self.var_check.get() == 0):
            messagebox.showerror("Error", "Please agree to the Terms and Conditions!")
        else:
            try:
                conn = mysql.connector.connect(username='root', password='admin', host='localhost', database='face_recognition', port=3306)
                mycursor = conn.cursor()
                query = ("SELECT * FROM regteach WHERE email=%s")
                value = (self.var_email.get(),)
                mycursor.execute(query, value)
                row = mycursor.fetchone()
                if row is not None:
                    messagebox.showerror("Error", "User already exists, please try another email")
                    self.reset_fields()  # Reset all fields
                else:
                    mycursor.execute("INSERT INTO regteach (fname, lname, cnum, email, ssq, sa, pwd) VALUES (%s, %s, %s, %s, %s, %s, %s)", (
                        self.var_fname.get(),
                        self.var_lname.get(),
                        self.var_cnum.get(),
                        self.var_email.get(),
                        self.var_ssq.get(),
                        self.var_sa.get(),
                        self.var_pwd.get()
                    ))
                    conn.commit()
                    conn.close()
                    messagebox.showinfo("Success", "Successfully Registered!", parent=self.root)
                    self.redirect_to_login()  # Redirect to login.py
            except Exception as es:
                messagebox.showerror("Error", f"Due to: {str(es)}", parent=self.root)

    def reset_fields(self):
        """Reset all input fields to empty"""
        self.var_fname.set("")
        self.var_lname.set("")
        self.var_cnum.set("")
        self.var_email.set("")
        self.var_ssq.set("Select")
        self.var_sa.set("")
        self.var_pwd.set("")
        self.var_cpwd.set("")
        self.var_check.set(0)

    def redirect_to_login(self):
        self.root.destroy()  # Close the registration window
        subprocess.run([sys.executable, "login.py"])  # Run login.py


if __name__ == "__main__":
    root = Tk()
    app = Register(root)
    root.mainloop()