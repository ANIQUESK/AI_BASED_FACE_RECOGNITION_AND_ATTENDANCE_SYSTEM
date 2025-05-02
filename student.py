import os
import cv2
import numpy as np
from tkinter import *
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import mysql.connector
import threading
import time

class Student:
    def __init__(self, root):
        self.root = root
        self.root.state('zoomed')
        self.root.title("Student Panel")

        # Variables
        self.var_dep = StringVar()
        self.var_course = StringVar()
        self.var_year = StringVar()
        self.var_semester = StringVar()
        self.var_std_id = StringVar()
        self.var_std_name = StringVar()
        self.var_div = StringVar()
        self.var_roll = StringVar()
        self.var_gender = StringVar()
        self.var_dob = StringVar()
        self.var_email = StringVar()
        self.var_mob = StringVar()
        self.var_address = StringVar()
        self.var_teacher = StringVar()
        self.var_radio1 = StringVar(value="No")  # Default to "No"
        self.var_search = StringVar()

        # Load DNN model for face detection
        self.face_detection_model = self.load_dnn_model()

        # Background Images
        self.setup_background()

        # Title
        title_lb1 = Label(self.bg_img, text="Welcome to Student Panel", font=("verdana", 33, "bold"), bg="white", fg="navyblue")
        title_lb1.place(x=0, y=0, width=1566, height=50)

        # Main Frame
        main_frame = Frame(self.bg_img, bd=2, bg="white")
        main_frame.place(x=5, y=55, width=1355, height=510)

        # Left Frame (Student Details)
        self.setup_left_frame(main_frame)

        # Right Frame (Student Table and Search)
        self.setup_right_frame(main_frame)

        # Fetch Data Initially
        self.fetch_data()

    def load_dnn_model(self):
        """Load the DNN-based face detection model."""
        model_path = "res10_300x300_ssd_iter_140000.caffemodel"
        config_path = "deploy.prototxt"

        if not os.path.exists(model_path) or not os.path.exists(config_path):
            messagebox.showerror("Error", "DNN model files not found!", parent=self.root)
            return None

        net = cv2.dnn.readNetFromCaffe(config_path, model_path)
        return net

    def setup_background(self):
        """Set up background images."""
        # Header Image
        img = Image.open(r"C:\Users\ANIQUE\Documents\Python_Test_Projects\Images_GUI\banner.jpg")
        img = img.resize((1566, 150), Image.Resampling.LANCZOS)
        self.photoimg = ImageTk.PhotoImage(img)
        f_lb1 = Label(self.root, image=self.photoimg)
        f_lb1.place(x=0, y=0, width=1566, height=150)

        # Background Image
        bg1 = Image.open(r"C:\Users\ANIQUE\Documents\Python_Test_Projects\Images_GUI\bg3.jpg")
        bg1 = bg1.resize((1566, 768), Image.Resampling.LANCZOS)
        self.photobg1 = ImageTk.PhotoImage(bg1)
        self.bg_img = Label(self.root, image=self.photobg1)
        self.bg_img.place(x=0, y=130, width=1566, height=768)

    def setup_left_frame(self, main_frame):
        """Set up the left frame for student details."""
        left_frame = LabelFrame(main_frame, bd=2, bg="white", relief=RIDGE, text="Student Details",
                                font=("verdana", 12, "bold"), fg="navyblue")
        left_frame.place(x=10, y=10, width=660, height=480)

        # Current Course Frame
        current_course_frame = LabelFrame(left_frame, bd=2, bg="white", relief=RIDGE, text="Current Course",
                                        font=("verdana", 12, "bold"), fg="navyblue")
        current_course_frame.place(x=10, y=5, width=635, height=150)

        # Department
        Label(current_course_frame, text="Department", font=("verdana", 12, "bold"), bg="white", fg="navyblue").grid(row=0, column=0, padx=5, pady=15)
        dep_combo = ttk.Combobox(current_course_frame, textvariable=self.var_dep, width=15, font=("verdana", 12, "bold"), state="readonly")
        dep_combo["values"] = ("Select Department", "AN", "AO", "CO", "CIVIL", "ET", "ME")
        dep_combo.current(0)
        dep_combo.grid(row=0, column=1, padx=5, pady=15, sticky=W)

        # Course
        Label(current_course_frame, text="Course", font=("verdana", 12, "bold"), bg="white", fg="navyblue").grid(row=0, column=2, padx=5, pady=15)
        cou_combo = ttk.Combobox(current_course_frame, textvariable=self.var_course, width=15, font=("verdana", 12, "bold"), state="readonly")
        cou_combo["values"] = ("Select Course", "ARTIFICIAL INTELLIGENCE", "AUTOMATION & ROBOTICS", "COMPUTER", "CIVIL", "ELECTRICAL", "MECHANICAL")
        cou_combo.current(0)
        cou_combo.grid(row=0, column=3, padx=5, pady=15, sticky=W)

        # Year
        Label(current_course_frame, text="Year", font=("verdana", 12, "bold"), bg="white", fg="navyblue").grid(row=1, column=0, padx=5, sticky=W)
        year_combo = ttk.Combobox(current_course_frame, textvariable=self.var_year, width=15, font=("verdana", 12, "bold"), state="readonly")
        year_combo["values"] = ("Select Year", "2019-22", "2020-23", "2021-24", "2022-25", "2023-26", "2024-27")
        year_combo.current(0)
        year_combo.grid(row=1, column=1, padx=5, pady=15, sticky=W)

        # Semester
        Label(current_course_frame, text="Semester", font=("verdana", 12, "bold"), bg="white", fg="navyblue").grid(row=1, column=2, padx=5, sticky=W)
        sem_combo = ttk.Combobox(current_course_frame, textvariable=self.var_semester, width=15, font=("verdana", 12, "bold"), state="readonly")
        sem_combo["values"] = ("Select Semester", "Semester-1", "Semester-2", "Semester-3", "Semester-4", "Semester-5", "Semester-6")
        sem_combo.current(0)
        sem_combo.grid(row=1, column=3, padx=5, pady=15, sticky=W)

        # Class Student Information Frame
        class_student_frame = LabelFrame(left_frame, bd=2, bg="white", relief=RIDGE, text="Class Student Information",
                                        font=("verdana", 12, "bold"), fg="navyblue")
        class_student_frame.place(x=10, y=160, width=635, height=230)

        # Student ID
        Label(class_student_frame, text="Std-ID:", font=("verdana", 12, "bold"), fg="navyblue", bg="white").grid(row=0, column=0, padx=5, pady=5, sticky=W)
        Entry(class_student_frame, textvariable=self.var_std_id, width=15, font=("verdana", 12, "bold")).grid(row=0, column=1, padx=5, pady=5, sticky=W)

        # Student Name
        Label(class_student_frame, text="Std-Name:", font=("verdana", 12, "bold"), fg="navyblue", bg="white").grid(row=0, column=2, padx=5, pady=5, sticky=W)
        Entry(class_student_frame, textvariable=self.var_std_name, width=15, font=("verdana", 12, "bold")).grid(row=0, column=3, padx=5, pady=5, sticky=W)

        # Class Division
        Label(class_student_frame, text="Class Division:", font=("verdana", 12, "bold"), fg="navyblue", bg="white").grid(row=1, column=0, padx=5, pady=5, sticky=W)
        div_combo = ttk.Combobox(class_student_frame, textvariable=self.var_div, width=13, font=("verdana", 12, "bold"), state="readonly")
        div_combo["values"] = ("Morning", "Evening")
        div_combo.current(0)
        div_combo.grid(row=1, column=1, padx=5, pady=5, sticky=W)

        # Roll No
        Label(class_student_frame, text="Roll-No:", font=("verdana", 12, "bold"), fg="navyblue", bg="white").grid(row=1, column=2, padx=5, pady=5, sticky=W)
        Entry(class_student_frame, textvariable=self.var_roll, width=15, font=("verdana", 12, "bold")).grid(row=1, column=3, padx=5, pady=5, sticky=W)

        # Gender
        Label(class_student_frame, text="Gender:", font=("verdana", 12, "bold"), fg="navyblue", bg="white").grid(row=2, column=0, padx=5, pady=5, sticky=W)
        gender_combo = ttk.Combobox(class_student_frame, textvariable=self.var_gender, width=13, font=("verdana", 12, "bold"), state="readonly")
        gender_combo["values"] = ("Male", "Female", "Others")
        gender_combo.current(0)
        gender_combo.grid(row=2, column=1, padx=5, pady=5, sticky=W)

        # DOB
        Label(class_student_frame, text="DOB:", font=("verdana", 12, "bold"), fg="navyblue", bg="white").grid(row=2, column=2, padx=5, pady=5, sticky=W)
        Entry(class_student_frame, textvariable=self.var_dob, width=15, font=("verdana", 12, "bold")).grid(row=2, column=3, padx=5, pady=5, sticky=W)

        # Email
        Label(class_student_frame, text="Email:", font=("verdana", 12, "bold"), fg="navyblue", bg="white").grid(row=3, column=0, padx=5, pady=5, sticky=W)
        Entry(class_student_frame, textvariable=self.var_email, width=15, font=("verdana", 12, "bold")).grid(row=3, column=1, padx=5, pady=5, sticky=W)

        # Mobile No
        Label(class_student_frame, text="Mob-No:", font=("verdana", 12, "bold"), fg="navyblue", bg="white").grid(row=3, column=2, padx=5, pady=5, sticky=W)
        Entry(class_student_frame, textvariable=self.var_mob, width=15, font=("verdana", 12, "bold")).grid(row=3, column=3, padx=5, pady=5, sticky=W)

        # Address
        Label(class_student_frame, text="Address:", font=("verdana", 12, "bold"), fg="navyblue", bg="white").grid(row=4, column=0, padx=5, pady=5, sticky=W)
        Entry(class_student_frame, textvariable=self.var_address, width=15, font=("verdana", 12, "bold")).grid(row=4, column=1, padx=5, pady=5, sticky=W)

        # Teacher Name
        Label(class_student_frame, text="Tutor Name:", font=("verdana", 12, "bold"), fg="navyblue", bg="white").grid(row=4, column=2, padx=5, pady=5, sticky=W)
        Entry(class_student_frame, textvariable=self.var_teacher, width=15, font=("verdana", 12, "bold")).grid(row=4, column=3, padx=5, pady=5, sticky=W)

        # Radio Buttons for Photo Sample
        Radiobutton(class_student_frame, text="Take Photo Sample", variable=self.var_radio1, value="Yes").grid(row=5, column=0, padx=5, pady=5, sticky=W)
        Radiobutton(class_student_frame, text="No Photo Sample", variable=self.var_radio1, value="No").grid(row=5, column=1, padx=5, pady=5, sticky=W)

        # Buttons Frame
        btn_frame = Frame(left_frame, bd=2, bg="white", relief=RIDGE)
        btn_frame.place(x=10, y=390, width=635, height=60)

        # Save Button
        Button(btn_frame, text="Save", command=self.add_data, width=7, font=("verdana", 12, "bold"), fg="white", bg="navyblue").grid(row=0, column=0, padx=5, pady=10, sticky=W)

        # Update Button
        Button(btn_frame, text="Update", command=self.update_data, width=7, font=("verdana", 12, "bold"), fg="white", bg="navyblue").grid(row=0, column=1, padx=5, pady=10, sticky=W)

        # Delete Button
        Button(btn_frame, text="Delete", command=self.delete_data, width=7, font=("verdana", 12, "bold"), fg="white", bg="navyblue").grid(row=0, column=2, padx=5, pady=10, sticky=W)

        # Reset Button
        Button(btn_frame, text="Reset", command=self.reset_data, width=7, font=("verdana", 12, "bold"), fg="white", bg="navyblue").grid(row=0, column=3, padx=5, pady=10, sticky=W)

        # Take Photo Button
        self.take_pic_button = Button(btn_frame, text="Take Pic", command=self.start_dataset_generation, width=9, font=("verdana", 12, "bold"), fg="white", bg="navyblue")
        self.take_pic_button.grid(row=0, column=4, padx=5, pady=10, sticky=W)

    def setup_right_frame(self, main_frame):
        """Set up the right frame for student table and search."""
        right_frame = LabelFrame(main_frame, bd=2, bg="white", relief=RIDGE, text="Student Details",
                                 font=("verdana", 12, "bold"), fg="navyblue")
        right_frame.place(x=680, y=10, width=660, height=480)

        # Search Frame
        search_frame = LabelFrame(right_frame, bd=2, bg="white", relief=RIDGE, text="Search System",
                                  font=("verdana", 12, "bold"), fg="navyblue")
        search_frame.place(x=10, y=5, width=635, height=80)

        # Search By Roll No
        Label(search_frame, text="Roll No:", font=("verdana", 12, "bold"), fg="navyblue", bg="white").grid(row=0, column=0, padx=5, pady=5, sticky=W)
        Entry(search_frame, textvariable=self.var_search, width=15, font=("verdana", 12, "bold")).grid(row=0, column=1, padx=5, pady=5, sticky=W)

        # Search Button
        Button(search_frame, text="Search", command=self.search_data, width=9, font=("verdana", 12, "bold"), fg="white", bg="navyblue").grid(row=0, column=2, padx=5, pady=10, sticky=W)

        # Show All Button
        Button(search_frame, text="Show All", command=self.fetch_data, width=8, font=("verdana", 12, "bold"), fg="white", bg="navyblue").grid(row=0, column=3, padx=5, pady=10, sticky=W)

        # Table Frame
        table_frame = Frame(right_frame, bd=2, bg="white", relief=RIDGE)
        table_frame.place(x=10, y=90, width=635, height=360)

        # Scrollbars
        scroll_x = ttk.Scrollbar(table_frame, orient=HORIZONTAL)
        scroll_y = ttk.Scrollbar(table_frame, orient=VERTICAL)

        # Student Table
        self.student_table = ttk.Treeview(table_frame, columns=("ID", "Name", "Dep", "Course", "Year", "Sem", "Div", "Gender", "DOB", "Mob-No", "Address", "Roll-No", "Email", "Teacher", "Photo"),
                                          xscrollcommand=scroll_x.set, yscrollcommand=scroll_y.set)
        scroll_x.pack(side=BOTTOM, fill=X)
        scroll_y.pack(side=RIGHT, fill=Y)
        scroll_x.config(command=self.student_table.xview)
        scroll_y.config(command=self.student_table.yview)

        # Table Headings
        self.student_table.heading("ID", text="StudentID")
        self.student_table.heading("Name", text="Name")
        self.student_table.heading("Dep", text="Department")
        self.student_table.heading("Course", text="Course")
        self.student_table.heading("Year", text="Year")
        self.student_table.heading("Sem", text="Semester")
        self.student_table.heading("Div", text="Division")
        self.student_table.heading("Gender", text="Gender")
        self.student_table.heading("DOB", text="DOB")
        self.student_table.heading("Mob-No", text="Mob-No")
        self.student_table.heading("Address", text="Address")
        self.student_table.heading("Roll-No", text="Roll-No")
        self.student_table.heading("Email", text="Email")
        self.student_table.heading("Teacher", text="Teacher")
        self.student_table.heading("Photo", text="PhotoSample")
        self.student_table["show"] = "headings"

        # Column Widths
        self.student_table.column("ID", width=100)
        self.student_table.column("Name", width=100)
        self.student_table.column("Dep", width=100)
        self.student_table.column("Course", width=100)
        self.student_table.column("Year", width=100)
        self.student_table.column("Sem", width=100)
        self.student_table.column("Div", width=100)
        self.student_table.column("Gender", width=100)
        self.student_table.column("DOB", width=100)
        self.student_table.column("Mob-No", width=100)
        self.student_table.column("Address", width=100)
        self.student_table.column("Roll-No", width=100)
        self.student_table.column("Email", width=100)
        self.student_table.column("Teacher", width=100)
        self.student_table.column("Photo", width=100)

        self.student_table.pack(fill=BOTH, expand=1)
        self.student_table.bind("<ButtonRelease>", self.get_cursor)

    def add_data(self):
        """Add student data to the database."""
        if (self.var_dep.get() == "Select Department" or self.var_course.get() == "Select Course" or
                self.var_year.get() == "Select Year" or self.var_semester.get() == "Select Semester" or
                self.var_std_id.get() == "" or self.var_std_name.get() == "" or self.var_div.get() == "" or
                self.var_roll.get() == "" or self.var_gender.get() == "" or self.var_dob.get() == "" or
                self.var_email.get() == "" or self.var_mob.get() == "" or self.var_address.get() == "" or
                self.var_teacher.get() == ""):
            messagebox.showerror("Error", "Please Fill All Fields are Required!", parent=self.root)
        else:
            try:
                conn = mysql.connector.connect(username='root', password='admin', host='localhost', database='face_recognition', port=3306)
                mycursor = conn.cursor()
                mycursor.execute("INSERT INTO student VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", (
                    self.var_std_id.get(),
                    self.var_std_name.get(),
                    self.var_dep.get(),
                    self.var_course.get(),
                    self.var_year.get(),
                    self.var_semester.get(),
                    self.var_div.get(),
                    self.var_gender.get(),
                    self.var_dob.get(),
                    self.var_mob.get(),
                    self.var_address.get(),
                    self.var_roll.get(),
                    self.var_email.get(),
                    self.var_teacher.get(),
                    self.var_radio1.get()
                ))
                conn.commit()
                self.fetch_data()
                conn.close()
                messagebox.showinfo("Success", "All Records are Saved!", parent=self.root)
            except Exception as es:
                messagebox.showerror("Error", f"Due to: {str(es)}", parent=self.root)

    def fetch_data(self):
        """Fetch data from the database and display it in the table."""
        conn = mysql.connector.connect(username='root', password='admin', host='localhost', database='face_recognition', port=3306)
        mycursor = conn.cursor()
        mycursor.execute("SELECT * FROM student")
        data = mycursor.fetchall()

        if len(data) != 0:
            self.student_table.delete(*self.student_table.get_children())
            for i in data:
                self.student_table.insert("", END, values=i)
            conn.commit()
        conn.close()

    def get_cursor(self, event=""):
        """Get the selected row data and populate the fields."""
        cursor_focus = self.student_table.focus()
        content = self.student_table.item(cursor_focus)
        data = content["values"]

        if data and len(data) >= 15:  # Ensure there are at least 15 elements in the data list
            self.var_std_id.set(data[0])
            self.var_std_name.set(data[1])
            self.var_dep.set(data[2])
            self.var_course.set(data[3])
            self.var_year.set(data[4])
            self.var_semester.set(data[5])
            self.var_div.set(data[6])
            self.var_gender.set(data[7])
            self.var_dob.set(data[8])
            self.var_mob.set(data[9])
            self.var_address.set(data[10])
            self.var_roll.set(data[11])
            self.var_email.set(data[12])
            self.var_teacher.set(data[13])
            self.var_radio1.set(data[14])
        else:
            self.reset_data()

    def update_data(self):
        """Update student data in the database."""
        if (self.var_dep.get() == "Select Department" or self.var_course.get() == "Select Course" or
                self.var_year.get() == "Select Year" or self.var_semester.get() == "Select Semester" or
                self.var_std_id.get() == "" or self.var_std_name.get() == "" or self.var_div.get() == "" or
                self.var_roll.get() == "" or self.var_gender.get() == "" or self.var_dob.get() == "" or
                self.var_email.get() == "" or self.var_mob.get() == "" or self.var_address.get() == "" or
                self.var_teacher.get() == ""):
            messagebox.showerror("Error", "Please Fill All Fields are Required!", parent=self.root)
        else:
            try:
                Update = messagebox.askyesno("Update", "Do you want to Update this Student Details?", parent=self.root)
                if Update > 0:
                    conn = mysql.connector.connect(username='root', password='admin', host='localhost', database='face_recognition', port=3306)
                    mycursor = conn.cursor()
                    mycursor.execute("UPDATE student SET Name=%s, Department=%s, Course=%s, Year=%s, Semester=%s, Division=%s, Gender=%s, DOB=%s, Mobile_No=%s, Address=%s, Roll_No=%s, Email=%s, Teacher_Name=%s, PhotoSample=%s WHERE Student_ID=%s", (
                        self.var_std_name.get(),
                        self.var_dep.get(),
                        self.var_course.get(),
                        self.var_year.get(),
                        self.var_semester.get(),
                        self.var_div.get(),
                        self.var_gender.get(),
                        self.var_dob.get(),
                        self.var_mob.get(),
                        self.var_address.get(),
                        self.var_roll.get(),
                        self.var_email.get(),
                        self.var_teacher.get(),
                        self.var_radio1.get(),
                        self.var_std_id.get()
                    ))
                else:
                    if not Update:
                        return
                conn.commit()
                self.fetch_data()
                conn.close()
                messagebox.showinfo("Success", "Successfully Updated!", parent=self.root)
            except Exception as es:
                messagebox.showerror("Error", f"Due to: {str(es)}", parent=self.root)

    def delete_data(self):
        """Delete student data from the database."""
        if self.var_std_id.get() == "":
            messagebox.showerror("Error", "Student Id Must be Required!", parent=self.root)
        else:
            try:
                delete = messagebox.askyesno("Delete", "Do you want to Delete?", parent=self.root)
                if delete > 0:
                    conn = mysql.connector.connect(username='root', password='admin', host='localhost', database='face_recognition', port=3306)
                    mycursor = conn.cursor()
                    sql = "DELETE FROM student WHERE Student_ID=%s"
                    val = (self.var_std_id.get(),)
                    mycursor.execute(sql, val)
                else:
                    if not delete:
                        return
                conn.commit()
                self.fetch_data()
                conn.close()
                messagebox.showinfo("Delete", "Successfully Deleted!", parent=self.root)
            except Exception as es:
                messagebox.showerror("Error", f"Due to: {str(es)}", parent=self.root)

    def reset_data(self):
        """Reset all input fields."""
        self.var_std_id.set(""),
        self.var_std_name.set(""),
        self.var_dep.set("Select Department"),
        self.var_course.set("Select Course"),
        self.var_year.set("Select Year"),
        self.var_semester.set("Select Semester"),
        self.var_div.set("Morning"),
        self.var_gender.set("Male"),
        self.var_dob.set(""),
        self.var_mob.set(""),
        self.var_address.set(""),
        self.var_roll.set(""),
        self.var_email.set(""),
        self.var_teacher.set(""),
        self.var_radio1.set("No")

    def search_data(self):
        """Search student data by Roll No."""
        if self.var_search.get() == "":
            messagebox.showerror("Error", "Please enter Roll No to search!", parent=self.root)
        else:
            try:
                conn = mysql.connector.connect(username='root', password='admin', host='localhost', database='face_recognition', port=3306)
                mycursor = conn.cursor()
                sql = "SELECT * FROM student WHERE Roll_No=%s"
                val = (self.var_search.get(),)
                mycursor.execute(sql, val)
                rows = mycursor.fetchall()
                if len(rows) != 0:
                    self.student_table.delete(*self.student_table.get_children())
                    for i in rows:
                        self.student_table.insert("", END, values=i)
                else:
                    messagebox.showerror("Error", "Data Not Found", parent=self.root)
                conn.commit()
                conn.close()
            except Exception as es:
                messagebox.showerror("Error", f"Due To: {str(es)}", parent=self.root)

    def start_dataset_generation(self):
        """Start dataset generation in a separate thread."""
        if self.var_std_id.get() == "":
            messagebox.showerror("Error", "Student ID is required!", parent=self.root)
            return

        # Disable the Take Pic button to prevent multiple clicks
        self.take_pic_button.config(state=DISABLED)

        # Start dataset generation in a separate thread
        dataset_thread = threading.Thread(target=self.generate_dataset, daemon=True)
        dataset_thread.start()
        
    def generate_dataset(self):
        """Generate dataset by capturing high-quality frontal face images."""
        try:
            if self.face_detection_model is None:
                messagebox.showerror("Error", "DNN model not loaded!", parent=self.root)
                return

            # Create directory if not exists
            os.makedirs("data_img", exist_ok=True)

            # Open camera with optimized settings
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower resolution for speed
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)  # Increase FPS for smooth capture

            if not cap.isOpened():
                messagebox.showerror("Error", "Webcam not accessible!", parent=self.root)
                return

            img_id = 0
            frame_skip = 2  # Process every 2nd frame to reduce load
            frame_count = 0

            while img_id < 100:  # Capture 150 high-quality images
                ret, frame = cap.read()
                if not ret:
                    messagebox.showerror("Error", "Failed to capture frame!", parent=self.root)
                    break

                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue  # Skip frames to optimize performance

                # Detect faces using DNN model
                blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False, crop=False)
                self.face_detection_model.setInput(blob)
                detections = self.face_detection_model.forward()

                faces = []
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.85:  # High confidence threshold for better accuracy
                        box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                        (x, y, x2, y2) = box.astype("int")
                        faces.append((x, y, x2 - x, y2 - y))

                # Ensure only one frontal face is detected
                if len(faces) == 1:
                    (x, y, w, h) = faces[0]

                    # Check if the face is centered in the frame
                    frame_center_x, frame_center_y = frame.shape[1] // 2, frame.shape[0] // 2
                    face_center_x, face_center_y = x + w // 2, y + h // 2
                    if abs(frame_center_x - face_center_x) > w // 2 or abs(frame_center_y - face_center_y) > h // 2:
                        cv2.putText(frame, "Center your face!", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.imshow("Capture Images", frame)
                        continue

                    # Adjust bounding box to be a perfect square
                    w = h = max(w, h)
                    x = max(0, x - (w - (x2 - x)) // 2)
                    y = max(0, y - (h - (y2 - y)) // 2)

                    # Ensure the bounding box fits within the frame
                    x, y, w, h = map(int, [x, y, min(frame.shape[1] - x, w), min(frame.shape[0] - y, h)])

                    face = frame[y:y + h, x:x + w]
                    if face.size == 0:
                        continue

                    # Resize and convert to grayscale
                    face = cv2.resize(face, (280, 280))  # Resize to a fixed size
                    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

                    # Save the face image with optimized quality
                    file_path = f"data_img/student.{self.var_std_id.get()}.{img_id}.jpg"
                    cv2.imwrite(file_path, gray_face, [cv2.IMWRITE_JPEG_QUALITY, 90])

                    # Display real-time capture feedback
                    cv2.putText(frame, f"Captured: {img_id + 1}", (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    img_id += 1

                else:
                    # Provide feedback if no valid face is detected
                    cv2.putText(frame, "No valid face detected!", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Show the frame with detected faces
                cv2.imshow("Capture Images", frame)

                # Exit on pressing 'Enter'
                if cv2.waitKey(1) == 13:
                    break

            cap.release()
            cv2.destroyAllWindows()
            messagebox.showinfo("Result", "Dataset generation completed successfully!", parent=self.root)

        except Exception as es:
            messagebox.showerror("Error", f"Due to: {str(es)}", parent=self.root)
        finally:
            self.take_pic_button.config(state=NORMAL)  # Re-enable button after completion

if __name__ == "__main__":
    root = Tk()
    obj = Student(root)
    root.mainloop()