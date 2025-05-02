import csv
import logging
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk
import os
import cv2
import numpy as np
from datetime import datetime, timedelta
import threading
import mysql.connector

class Face_Recognition:
    def __init__(self, root):
        self.root = root
        self.root.state('zoomed')
        self.root.title("Face Recognition Panel")

        # Database configuration
        self.db_config = {
            'user': 'root',
            'password': 'admin',
            'host': 'localhost',
            'database': 'face_recognition',
            'port': 3306
        }

        # Attendance tracking with a 1-hour cooldown
        self.attendance_tracker = {}
        self.attendance_cooldown = timedelta(hours=1)

        # Cache student data to minimize database calls
        self.student_cache = self.load_student_data()

        # UI Configuration
        self.FONT = ("verdana", 30, "bold")
        self.BUTTON_FONT = ("tahoma", 15, "bold")
        self.BG_COLOR = "white"
        self.FG_COLOR = "navyblue"

        # Video capture and recognition settings
        self.video_cap = None
        self.is_running = False
        self.frame_skip = 2  # Process every 2nd frame to reduce load
        self.frame_count = 0

        # Load OpenCV DNN face detection model
        self.face_net = cv2.dnn.readNetFromCaffe(
            "deploy.prototxt",  # Path to the prototxt file
            "res10_300x300_ssd_iter_140000.caffemodel"  # Path to the caffemodel file
        )

        # Enable GPU acceleration if available
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            self.face_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # Load face recognition model
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        if os.path.exists("clf.xml"):
            self.recognizer.read("clf.xml")
        else:
            messagebox.showerror("Error", "Classifier file 'clf.xml' not found!")
            self.root.destroy()

        self.setup_ui()

    def load_student_data(self):
        """Load student data from the database to reduce queries."""
        student_cache = {}
        try:
            with mysql.connector.connect(**self.db_config) as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT Student_ID, Name, Roll_No FROM student")
                    for student_id, name, roll_no in cursor.fetchall():
                        student_cache[student_id] = (name, roll_no)
        except mysql.connector.Error as err:
            messagebox.showerror("Database Error", f"Error loading student data: {err}")
        return student_cache

    def setup_ui(self):
        """Initialize UI components"""
        try:
            banner_img = Image.open("Images_GUI/banner.jpg").resize((1566, 150), Image.LANCZOS)
            self.photoimg = ImageTk.PhotoImage(banner_img)
            Label(self.root, image=self.photoimg).place(x=0, y=0, width=1566, height=150)

            bg_img = Image.open("Images_GUI/bg2.jpg").resize((1566, 799), Image.LANCZOS)
            self.photobg1 = ImageTk.PhotoImage(bg_img)
            bg_label = Label(self.root, image=self.photobg1)
            bg_label.place(x=0, y=130, width=1566, height=799)

            Label(bg_label, text="Welcome to Face Recognition Panel", 
                 font=self.FONT, bg=self.BG_COLOR, fg=self.FG_COLOR).place(x=0, y=0, width=1566, height=50)

            # Face Detector button
            btn_img = Image.open("Images_GUI/f_det.jpg").resize((200, 200), Image.LANCZOS)
            self.std_img1 = ImageTk.PhotoImage(btn_img)

            Button(bg_label, command=self.start_face_recognition_thread, 
                  image=self.std_img1, cursor="hand2").place(x=700, y=190, width=200, height=200)
            Button(bg_label, command=self.start_face_recognition_thread, text="Face Detector", 
                  cursor="hand2", font=self.BUTTON_FONT, bg=self.BG_COLOR, fg=self.FG_COLOR
                  ).place(x=700, y=390, width=200, height=45)

        except FileNotFoundError as e:
            messagebox.showerror("File Error", f"Required image file not found: {e.filename}")

    def start_face_recognition_thread(self):
        """Start face recognition in a separate thread."""
        self.is_running = True
        threading.Thread(target=self.face_recognition, daemon=True).start()

    def face_recognition(self):
        """Main face recognition method using OpenCV DNN for multiple faces."""
        self.video_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.video_cap.set(cv2.CAP_PROP_FPS, 15)

        while self.is_running:
            ret, frame = self.video_cap.read()
            if not ret:
                messagebox.showwarning("Camera Error", "Failed to capture frame")
                break

            self.frame_count += 1
            if self.frame_count % self.frame_skip != 0:
                continue

            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (300, 300))
            h, w = frame.shape[:2]

            # Prepare the frame for DNN face detection
            blob = cv2.dnn.blobFromImage(small_frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
            self.face_net.setInput(blob)
            detections = self.face_net.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:  # Confidence threshold
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x, y, x1, y1) = box.astype("int")

                    # Ensure bounding box is within frame dimensions
                    x, y, x1, y1 = max(0, x), max(0, y), min(w, x1), min(h, y1)

                    # Extract face ROI
                    face_roi = frame[y:y1, x:x1]
                    if face_roi.size == 0:
                        continue

                    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    gray_face = cv2.resize(gray_face, (100, 100))

                    # Recognize face
                    try:
                        student_id, confidence = self.recognizer.predict(gray_face)
                        confidence_pct = int(100 * (1 - confidence / 300))

                        if confidence_pct > 65 and student_id in self.student_cache:
                            self.draw_face_info(frame, x, y, x1 - x, y1 - y, student_id)
                            self.mark_attendance(student_id)
                        else:
                            self.draw_unknown_face(frame, x, y, x1 - x, y1 - y)
                    except Exception as e:
                        logging.error(f"Error recognizing face: {e}")
                        self.draw_unknown_face(frame, x, y, x1 - x, y1 - y)

            # Display the frame with detected faces
            cv2.imshow("Face Detector", frame)
            if cv2.waitKey(1) == 13:  # Exit on Enter key
                break

        self.is_running = False
        self.video_cap.release()
        cv2.destroyAllWindows()

    def draw_face_info(self, frame, x, y, w, h, student_id):
        """Draw recognized face information on the frame"""
        name, _ = self.student_cache.get(student_id, ("Unknown", "N/A"))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(frame, f"ID: {student_id}, Name: {name}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def draw_unknown_face(self, frame, x, y, w, h):
        """Draw unknown face information"""
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    def mark_attendance(self, student_id):
        """Mark attendance for the recognized student."""
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        current_date = now.strftime("%d/%m/%Y")
        name, _ = self.student_cache.get(student_id, ("Unknown", "N/A"))

        # Check if the student is already marked within the cooldown period
        if student_id not in self.attendance_tracker or now - self.attendance_tracker[student_id] > self.attendance_cooldown:
            self.attendance_tracker[student_id] = now
            with open("attendance.csv", "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([student_id, student_id, name, current_time, current_date, "Present"])
                
    def on_closing(self):
        """Handle window closing event"""
        self.is_running = False
        if self.video_cap:
            self.video_cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()

if __name__ == "__main__":
    root = Tk()
    app = Face_Recognition(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()