![Screenshot 2025-03-02 151419](https://github.com/user-attachments/assets/e4ae88a0-986b-4b6c-be65-67eeef6ba19f)
# Face Recognition Attendance System 👤📸

![Banner](https://img.shields.io/badge/Version-1.0.0-blue.svg) ![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg) ![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A robust and efficient **Face Recognition Attendance System** built with Python, OpenCV, and Tkinter. This project automates attendance tracking by detecting and recognizing faces using advanced computer vision techniques, including HAAR cascades and DNN-based face detection. It features a user-friendly GUI, database integration with MySQL, and optimized training pipelines for real-time performance.

---

## ✨ Features

- **Face Detection & Recognition**: Uses HAAR cascades and DNN models for accurate face detection and LBPH for recognition.
- **Automated Attendance Tracking**: Marks attendance with a cooldown mechanism to prevent duplicate entries.
- **User-Friendly GUI**: Built with Tkinter for easy student management and real-time feedback.
- **Database Integration**: Stores student details and attendance records in MySQL.
- **Optimized Performance**: Includes parallel processing, frame skipping, and efficient data augmentation.
- **Cross-Validation Training**: Implements K-fold cross-validation for robust model evaluation.
- **Real-Time Feedback**: Provides visual feedback during face capture and recognition.

---

## 📷 Screenshots

## 📷 Screenshots

| Student Management Panel | Face Recognition in Action |
|--------------------------|----------------------------|
| ![Screenshot 2025-05-02 071212](https://github.com/user-attachments/assets/4ddabbe2-d311-4f5b-a795-29969c814376)| ![Screenshot 2025-05-02 070038](https://github.com/user-attachments/assets/ca1343da-7c24-44b2-b5c5-2e328fd10024)|
 
---

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+**
- **MySQL Server** (for storing student and attendance data)
- A webcam for capturing face images
- Required Python libraries (listed in `requirements.txt`)

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/face-recognition-attendance-system.git
   cd face-recognition-attendance-system
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up MySQL Database**:
   - Create a database named `face_recognition`.
   - Run the following SQL to create the `student` table:
     ```sql
     CREATE TABLE student (
         Student_ID VARCHAR(50) PRIMARY KEY,
         Name VARCHAR(100),
         Department VARCHAR(50),
         Course VARCHAR(100),
         Year VARCHAR(50),
         Semester VARCHAR(50),
         Division VARCHAR(50),
         Gender VARCHAR(20),
         DOB VARCHAR(50),
         Mobile_No VARCHAR(15),
         Address VARCHAR(255),
         Roll_No VARCHAR(50),
         Email VARCHAR(100),
         Teacher_Name VARCHAR(100),
         PhotoSample VARCHAR(10)
     );
     ```
   - Update the database credentials in `student.py` and `face_recognition.py` (default: `root`/`admin`).

4. **Download DNN Model Files** (if using DNN-based detection):
   - Download `deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel` from [OpenCV's GitHub](https://github.com/opencv/opencv_3rdparty).
   - Place them in the `models/` directory.

5. **Directory Setup**:
   - Ensure the following directories exist: `data_img/`, `models/`, and `temp/`.
   - Place GUI images in `Images_GUI/` (e.g., `banner.jpg`, `bg3.jpg`, etc.).

---

## 🛠️ Usage

1. **Run the Application**:
   - Start with the student management panel:
     ```bash
     python student.py
     ```
   - Add student details and capture face images using the "Take Pic" button.

2. **Train the Model**:
   - Use the training panel to train the face recognition model:
     ```bash
     python train.py
     ```
   - This will process captured images and save the trained model as `models/recognizer.xml`.

3. **Start Face Recognition**:
   - Launch the face recognition module to mark attendance:
     ```bash
     python face_recognition.py
     ```
   - The system will detect and recognize faces, logging attendance in `attendance.csv`.

4. **Use the Face Processor** (Advanced):
   - For command-line training and detection:
     ```bash
     python face_processor.py train '{"algorithm": "DNN", "parallel": true}'
     ```
   - For processing a frame:
     ```bash
     python face_processor.py detect "<base64_frame>" '{"algorithm": "DNN"}'
     ```

---

## 📋 Project Structure

```
face-recognition-attendance-system/
│
├── data_img/                # Stores captured face images
├── models/                  # Stores trained models and DNN files
├── temp/                    # Temporary files
├── Images_GUI/              # GUI images (banner.jpg, bg3.jpg, etc.)
│
├── student.py               # Student management panel (GUI)
├── train.py                 # Training module (GUI)
├── face_recognition.py      # Face recognition and attendance tracking (GUI)
├── face_processor.py        # Core face processing logic (CLI)
├── requirements.txt         # Dependencies
├── README.md                # Project documentation
└── face_processing.log      # Logs for debugging
```

---

## 🔧 Configuration

- **Face Detection Algorithm**: Choose between `HAAR` (faster) or `DNN` (more accurate) in `face_processor.py`.
- **Training Options**: Adjust `kFold`, `testSize`, and `parallel` in the training options JSON.
- **Capture Settings**: Modify `minFaceSize`, `scaleFactor`, and `confidenceThreshold` in `face_processor.py` for better detection.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🌟 Acknowledgments

- [OpenCV](https://opencv.org/) for face detection and recognition libraries.
- [Tkinter](https://docs.python.org/3/library/tkinter.html) for GUI development.
- [MySQL Connector](https://dev.mysql.com/doc/connector-python/en/) for database integration.

---

## 📧 Contact

For questions or suggestions, feel free to reach out:

- **Email**: shaikhanique07gmail.com
- **GitHub Issues**: [Open an Issue](/issues)

---

⭐ **Star this repository if you find it useful!** ⭐
