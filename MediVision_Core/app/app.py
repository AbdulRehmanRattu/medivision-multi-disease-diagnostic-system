import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox, QTabWidget, QStackedWidget
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtCore import Qt
import tb_inference
import covid_inference
import lung_cancer_inference
import pneumonia_inference

# Function to predict using the loaded model
def predict_image_class(script, img_path):
    if script == "tb":
        return tb_inference.predict_image_class(img_path)
    elif script == "covid":
        return covid_inference.predict_image_class(img_path)
    elif script == "lung":
        return lung_cancer_inference.predict_image_class(img_path)
    elif script == "pneumonia":
        return pneumonia_inference.predict_image_class(img_path)
    return "Unknown script"

# Main application class
class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'MediVision'
        self.left = 100
        self.top = 100
        self.width = 600
        self.height = 600
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setStyleSheet("background-color: #E3F2FD;")  # Light blue background
        
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)
        
        self.home_tab = QWidget()
        self.inference_tabs = QStackedWidget()
        
        self.tabs.addTab(self.home_tab, "Home")
        self.tabs.addTab(self.inference_tabs, "Inference")
        
        self.initHomeTab()
        self.initInferenceTabs()
        
    def initHomeTab(self):
        home_layout = QVBoxLayout()
        
        # App logo and name
        logo_label = QLabel()
        pixmap = QPixmap("logo.png")  # Use the new logo file
        pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio)
        logo_label.setPixmap(pixmap)
        logo_label.setAlignment(Qt.AlignCenter)
        home_layout.addWidget(logo_label)
        
        app_name_label = QLabel("MediVision: Medical Image Inference")
        app_name_label.setFont(QFont('Helvetica', 24, QFont.Bold))
        app_name_label.setAlignment(Qt.AlignCenter)
        home_layout.addWidget(app_name_label)
        
        app_details_label = QLabel("This application allows you to upload medical images and get predictions for Tuberculosis, COVID-19, Lung Cancer, and Pneumonia.")
        app_details_label.setFont(QFont('Helvetica', 14))
        app_details_label.setWordWrap(True)
        app_details_label.setAlignment(Qt.AlignCenter)
        home_layout.addWidget(app_details_label)
        
        self.home_tab.setLayout(home_layout)
        
    def initInferenceTabs(self):
        self.tb_tab = self.create_inference_tab("Tuberculosis Detection", "tb")
        self.covid_tab = self.create_inference_tab("COVID-19 Detection", "covid")
        self.lung_tab = self.create_inference_tab("Lung Cancer Detection", "lung")
        self.pneumonia_tab = self.create_inference_tab("Pneumonia Detection", "pneumonia")
        
        self.inference_tabs.addWidget(self.tb_tab)
        self.inference_tabs.addWidget(self.covid_tab)
        self.inference_tabs.addWidget(self.lung_tab)
        self.inference_tabs.addWidget(self.pneumonia_tab)
        
        self.initModelButtons()
        
    def initModelButtons(self):
        home_layout = self.home_tab.layout()
        
        button_layout = QHBoxLayout()
        
        self.tb_button = self.create_button('Tuberculosis Detection', self.tb_tab, button_layout)
        self.covid_button = self.create_button('COVID-19 Detection', self.covid_tab, button_layout)
        self.lung_button = self.create_button('Lung Cancer Detection', self.lung_tab, button_layout)
        self.pneumonia_button = self.create_button('Pneumonia Detection', self.pneumonia_tab, button_layout)
        
        home_layout.addLayout(button_layout)
        
    def create_button(self, text, tab, layout):
        button = QPushButton(text, self)
        button.setFont(QFont('Helvetica', 14))
        button.setStyleSheet("background-color: #2196F3; color: white; padding: 10px; border-radius: 5px;")
        button.clicked.connect(lambda: self.show_inference_tab(tab))
        layout.addWidget(button)
        return button
        
    def create_inference_tab(self, model_name, script):
        tab = QWidget()
        tab_layout = QVBoxLayout()
        
        back_button = QPushButton('Back to Dashboard', self)
        back_button.setFont(QFont('Helvetica', 12))
        back_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px;")
        back_button.clicked.connect(self.go_back_to_dashboard)
        tab_layout.addWidget(back_button)
        
        model_label = QLabel(f'Using Model: {model_name}', self)
        model_label.setFont(QFont('Helvetica', 18, QFont.Bold))
        model_label.setAlignment(Qt.AlignCenter)
        tab_layout.addWidget(model_label)
        
        upload_label = QLabel('Upload Image for Inference', self)
        upload_label.setFont(QFont('Helvetica', 18, QFont.Bold))
        tab_layout.addWidget(upload_label)
        
        upload_button = QPushButton('Upload and Predict', self)
        upload_button.setFont(QFont('Helvetica', 14))
        upload_button.setStyleSheet("background-color: #FF5722; color: white; padding: 10px; border-radius: 5px;")
        upload_button.clicked.connect(lambda: self.upload_and_predict(script, tab))
        tab_layout.addWidget(upload_button)

        self.result_label = QLabel('', self)
        self.result_label.setFont(QFont('Helvetica', 16))
        self.result_label.setAlignment(Qt.AlignCenter)
        tab_layout.addWidget(self.result_label)
        
        tab.setLayout(tab_layout)
        return tab
    
    def show_inference_tab(self, tab):
        index = self.inference_tabs.indexOf(tab)
        self.inference_tabs.setCurrentIndex(index)
        self.tabs.setCurrentIndex(1)
        
    def go_back_to_dashboard(self):
        self.tabs.setCurrentIndex(0)
        
    def upload_and_predict(self, script, tab):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "All Files (*);;Image Files (*.jpg *.jpeg *.png)", options=options)
        if file_path:
            result, probability = predict_image_class(script, file_path)
            tab.layout().itemAt(4).widget().setText(f'Prediction: {result}\nConfidence Probability: {probability:.4f}')
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())