# 🥔 Potato Leaf Health Check

##  About the Project
Potato Leaf Health Check is a Deep Learning project that uses a Convolutional Neural Network (CNN) to identify potato leaf diseases from images.

The project uses TensorFlow and Keras for model training and Streamlit to create a simple user interface.

##  Features
- Upload a potato leaf image.
- Predict the health condition of the leaf.
- CNN-based image classification.
- Simple and interactive Streamlit interface.

##  Technologies Used
- Python
- TensorFlow
- Keras
- CNN (Convolutional Neural Network)
- NumPy
- Streamlit
- Pillow

##  Project Structure

```text
cnn/
│
├── data/
├── app.py
├── cnn_code.ipynb
├── potato_cnn_model.keras
├── classification_report.txt
├── leaf.JPG
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project folder

```bash
cd Potato-Leaf-Health-Check
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment (Windows)

```bash
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
streamlit run app.py
```

##  Model
The project uses a trained CNN model saved as `potato_cnn_model.keras`.

##  Author
Ankit Jat