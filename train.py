from tkinter import *
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging
import threading

# Configure logging
logging.basicConfig(
    filename="training_log.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class Train:
    def __init__(self, root):
        self.root = root
        self.root.state('zoomed')
        self.root.title("Train Panel")

        # Load images
        self.load_images()

        # Title section
        title_lb1 = Label(
            self.bg_img,
            text="Welcome to Training Panel",
            font=("verdana", 34, "bold"),
            bg="white",
            fg="navyblue"
        )
        title_lb1.place(x=0, y=0, width=1566, height=55)

        # Create buttons
        self.create_buttons()

        # Progress bar
        self.progress = ttk.Progressbar(self.bg_img, orient=HORIZONTAL, length=400, mode="determinate")
        self.progress.place(x=600, y=460, width=400, height=30)

        # Training status label
        self.status_label = Label(
            self.bg_img,
            text="",
            font=("tahoma", 12),
            bg="white",
            fg="navyblue"
        )
        self.status_label.place(x=600, y=500, width=400, height=30)

    def load_images(self):
        """Load and set up UI images."""
        try:
            # First header image
            img = Image.open(r"Images_GUI/banner.jpg")
            img = img.resize((1566, 150), Image.Resampling.LANCZOS)
            self.photoimg = ImageTk.PhotoImage(img)

            # Set image as label
            f_lb1 = Label(self.root, image=self.photoimg)
            f_lb1.place(x=0, y=0, width=1566, height=150)

            # Background image
            bg1 = Image.open(r"Images_GUI/t_bg1.jpg")
            bg1 = bg1.resize((1566, 799), Image.Resampling.LANCZOS)
            self.photobg1 = ImageTk.PhotoImage(bg1)

            # Set image as label
            self.bg_img = Label(self.root, image=self.photobg1)
            self.bg_img.place(x=0, y=130, width=1566, height=799)
        except Exception as e:
            logging.error(f"Error loading images: {e}")
            messagebox.showerror("Error", f"Failed to load UI images: {e}")

    def create_buttons(self):
        """Create buttons for the UI."""
        try:
            # Training button
            std_img_btn = Image.open(r"Images_GUI/t_btn1.png")
            std_img_btn = std_img_btn.resize((200, 200), Image.Resampling.LANCZOS)
            self.std_img1 = ImageTk.PhotoImage(std_img_btn)

            self.train_button = Button(
                self.bg_img,
                command=self.start_training_thread,
                image=self.std_img1,
                cursor="hand2"
            )
            self.train_button.place(x=700, y=200, width=200, height=200)

            self.train_button_text = Button(
                self.bg_img,
                command=self.start_training_thread,
                text="Train Dataset",
                cursor="hand2",
                font=("tahoma", 15, "bold"),
                bg="white",
                fg="navyblue"
            )
            self.train_button_text.place(x=700, y=400, width=200, height=50)
        except Exception as e:
            logging.error(f"Error creating buttons: {e}")
            messagebox.showerror("Error", f"Failed to create buttons: {e}")

    def start_training_thread(self):
        """Start the training process in a separate thread."""
        self.train_button.config(state=DISABLED)  # Disable the button
        self.train_button_text.config(state=DISABLED)  # Disable the text button
        self.status_label.config(text="Training started...")
        self.progress["value"] = 0
        training_thread = threading.Thread(target=self.train_classifier, daemon=True)
        training_thread.start()

    def preprocess_image(self, image_path):
        """Preprocess the image for better face detection."""
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                logging.error(f"Failed to load image: {image_path}")
                return None

            # Equalize histogram to improve contrast
            img = cv2.equalizeHist(img)

            # Resize image to a smaller resolution
            img = cv2.resize(img, (100, 100))  # Reduced resolution to save memory

            return img
        except Exception as e:
            logging.error(f"Error preprocessing image {image_path}: {e}")
            return None

    def augment_data(self, face):
        """Apply data augmentation techniques."""
        augmented_faces = []

        # Original face
        augmented_faces.append(face)

        # Flip horizontally
        flipped_face = cv2.flip(face, 1)
        augmented_faces.append(flipped_face)

        # Rotate by small angles
        for angle in [-10, 10]:
            rows, cols = face.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            rotated_face = cv2.warpAffine(face, M, (cols, rows))
            augmented_faces.append(rotated_face)

        return augmented_faces

    def train_classifier(self):
        """Train the face recognition model."""
        try:
            data_dir = "data_img"
            if not os.path.exists(data_dir):
                messagebox.showerror("Error", f"Dataset directory '{data_dir}' not found!", parent=self.root)
                self.status_label.config(text="Error: Dataset not found")
                return

            paths = [os.path.join(data_dir, file) for file in os.listdir(data_dir)]
            if not paths:
                messagebox.showerror("Error", "No images found in the dataset!", parent=self.root)
                self.status_label.config(text="Error: No images found")
                return

            faces = []
            ids = []

            # Load Haar Cascade for face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            logging.info("Starting training process...")

            total_images = len(paths)
            processed_images = 0

            for image_path in paths:
                try:
                    img = self.preprocess_image(image_path)
                    if img is None:
                        continue

                    # Detect faces in the image
                    detected_faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                    if len(detected_faces) == 0:
                        logging.warning(f"No faces detected in image: {image_path}")
                        continue

                    for (x, y, w, h) in detected_faces:
                        face = img[y:y+h, x:x+w]

                        # Augment data
                        augmented_faces = self.augment_data(face)

                        # Get ID from the filename
                        id = int(os.path.split(image_path)[1].split('.')[1])

                        for augmented_face in augmented_faces:
                            faces.append(augmented_face)
                            ids.append(id)

                    processed_images += 1
                    progress_value = int((processed_images / total_images) * 100)
                    self.progress["value"] = progress_value
                    self.status_label.config(text=f"Processing {processed_images}/{total_images} images...")
                    self.root.update_idletasks()  # Update the UI

                except Exception as e:
                    logging.error(f"Error processing image {image_path}: {e}")

            if len(faces) == 0:
                messagebox.showerror("Error", "No faces found in the dataset.", parent=self.root)
                logging.error("No faces found in the dataset.")
                self.status_label.config(text="Error: No faces found")
                return

            ids = np.array(ids)

            # Split data into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(faces, ids, test_size=0.2, random_state=42)

            # Train the classifier
            clf = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)  # Reduced parameters
            clf.train(X_train, y_train)

            # Validate the model
            y_pred = []
            for face in X_val:
                pred_id, _ = clf.predict(face)
                y_pred.append(pred_id)

            accuracy = accuracy_score(y_val, y_pred)
            logging.info(f"Validation Accuracy: {accuracy * 100:.2f}%")

            # Save the model
            model_path = "clf.xml"
            clf.write(model_path)
            logging.info(f"Model saved to {model_path}")

            self.progress["value"] = 100
            self.status_label.config(text=f"Training completed! Accuracy: {accuracy * 100:.2f}%")
            messagebox.showinfo(
                "Result",
                f"Training Dataset Completed!\nValidation Accuracy: {accuracy * 100:.2f}%",
                parent=self.root
            )

            # Automatically close the window after training
            self.root.after(2000, self.root.destroy)  # Close the window after 2 seconds

        except Exception as e:
            logging.error(f"Error during training: {e}")
            messagebox.showerror("Error", f"Training failed: {e}", parent=self.root)
        finally:
            self.train_button.config(state=NORMAL)  # Re-enable the button
            self.train_button_text.config(state=NORMAL)  # Re-enable the text button

if __name__ == "__main__":
    root = Tk()
    obj = Train(root)
    root.mainloop()