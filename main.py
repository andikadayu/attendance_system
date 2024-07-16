import sys
import cv2
import face_recognition
import os
import pickle
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QMessageBox
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
from PyQt5.QtCore import QTimer, Qt

class FaceRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.registered_users = self.load_registered_users()
        self.video_capture = cv2.VideoCapture(0)
        if not self.video_capture.isOpened():
            QMessageBox.critical(self, "Camera Error", "Unable to access the camera.")
            sys.exit(1)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def initUI(self):
        self.setWindowTitle('Sistem Absensi Menggunakan Face Recognition')
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        self.name_input = QLineEdit(self)
        self.name_input.setPlaceholderText('Masukkan nama pengguna')
        layout.addWidget(self.name_input)

        self.register_button = QPushButton('Daftar Pengguna', self)
        self.register_button.clicked.connect(self.register_user)
        layout.addWidget(self.register_button)

        self.attendance_button = QPushButton('Mulai Absensi', self)
        self.attendance_button.clicked.connect(self.start_attendance)
        layout.addWidget(self.attendance_button)

        self.image_label = QLabel(self)
        layout.addWidget(self.image_label)

        self.setLayout(layout)

    def load_registered_users(self):
        registered_users = {}
        if not os.path.exists('registered_users'):
            os.makedirs('registered_users')
        for file_name in os.listdir('registered_users'):
            if file_name.endswith('.pkl'):
                name = file_name[:-4]
                with open(f'registered_users/{file_name}', 'rb') as f:
                    registered_users[name] = pickle.load(f)
        return registered_users

    def register_user(self):
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, 'Error', 'Nama pengguna tidak boleh kosong.')
            return

        ret, frame = self.video_capture.read()
        if not ret or frame is None:
            QMessageBox.warning(self, 'Error', 'Tidak dapat menangkap gambar dari kamera.')
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_frame)
        if face_encodings:
            with open(f'registered_users/{name}.pkl', 'wb') as f:
                pickle.dump(face_encodings[0], f)
            self.registered_users[name] = face_encodings[0]
            QMessageBox.information(self, 'Sukses', f'Wajah {name} berhasil disimpan.')
        else:
            QMessageBox.warning(self, 'Error', 'Tidak ada wajah terdeteksi. Coba lagi.')

    def start_attendance(self):
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.video_capture.read()
        if not ret or frame is None:
            print("Warning: Unable to capture frame from camera.")
            return

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.display_image(frame, rgb_frame)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(list(self.registered_users.values()), face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(list(self.registered_users.values()), face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = list(self.registered_users.keys())[best_match_index]

                if name != "Unknown":
                    self.mark_attendance(name)
                    # Draw a rectangle around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                    # Draw text overlay with the name
                    cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            self.display_image(frame, rgb_frame)

        except Exception as e:
            print(f"Error during frame processing: {e}")

    def mark_attendance(self, name):
        temp = []
        if name not in temp :
            with open('attendance.csv', 'a') as f:
                now = datetime.now()
                justDate = now.strftime('%Y-%m-%d')
                dt_string = now.strftime('%Y-%m-%d %H:%M:%S')
                f.write(f'{name},{dt_string}\n')
                temp.append(name+":"+justDate)
                print(f'Absensi tercatat untuk {name} pada {dt_string}')

    def display_image(self, frame, rgb_frame):
        try:
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)

            # Scale pixmap to fit the size of the QLabel
            pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)

            # Display the image
            self.image_label.setPixmap(pixmap)
        except Exception as e:
            print(f"Error displaying image: {e}")

    def closeEvent(self, event):
        self.timer.stop()
        self.video_capture.release()
        cv2.destroyAllWindows()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FaceRecognitionApp()
    ex.show()
    sys.exit(app.exec_())