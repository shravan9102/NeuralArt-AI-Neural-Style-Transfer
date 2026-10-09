# 🎨 NeuralArt AI — Neural Style Transfer

Transform ordinary photographs into artistic masterpieces using Deep Learning and Computer Vision.

NeuralArt AI is an AI-powered image stylization application built using **Python, PyTorch, Flask, and Adaptive Instance Normalization (AdaIN)**. The project combines the content of one image with the artistic style of another to generate creative, stylized images.

## ✨ Features

* 🎨 **AI-Powered Image Stylization** — Transform photographs into artistic creations.
* 🧠 **Deep Learning** — Uses neural network features for image processing.
* ⚡ **Adaptive Instance Normalization (AdaIN)** — Transfers artistic style characteristics to content images.
* 🌐 **Flask Web Application** — Provides a web interface for interacting with the application.
* 🖼️ **Image Upload and Processing** — Work with content images and artistic styles.
* 🔧 **Modular Architecture** — Organized Python utilities, templates, datasets, and model files.

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Deep Learning Framework:** PyTorch
* **Web Framework:** Flask
* **Neural Style Transfer:** Adaptive Instance Normalization (AdaIN)
* **Feature Extraction:** VGG
* **Frontend:** HTML, CSS
* **Version Control:** Git and GitHub

## 🧠 How It Works

Neural style transfer combines two types of images:

* **Content Image:** The image whose main objects and structure should be preserved.
* **Style Image:** The artwork whose colors, textures, and patterns should influence the output.

The application uses deep learning features and AdaIN to combine content information with artistic style characteristics.

### Workflow

1. Upload a content image.
2. Select or provide an artistic style image.
3. Process the images through the neural style transfer model.
4. Generate and view the stylized image.

## 📁 Project Structure

```text
NeuralArt-AI-Neural-Style-Transfer/
│
├── content_data/        # Content image data
├── style_data/          # Artistic style image data
├── examples/            # Example images and outputs
├── experiment/
│   └── experiment1/     # Experiment files
├── static/
│   └── uploads/         # Uploaded images and static assets
├── templates/           # Flask HTML templates
├── utils/               # Helper modules
│
├── app.py               # Flask web application
├── train.py             # Training/model-related script
├── requirements.txt     # Python dependencies
├── vgg_normalised.pth   # Pretrained model weights
└── README.md            # Project documentation
```

## 🚀 Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/shravan9102/NeuralArt-AI-Neural-Style-Transfer.git
```

### 2. Navigate to the Project Directory

```bash
cd NeuralArt-AI-Neural-Style-Transfer
```

### 3. Create a Virtual Environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Make sure your Python version and PyTorch installation are compatible with the dependencies specified in the project.

### 5. Run the Application

```bash
python app.py
```

Open the local address displayed in your terminal. A typical Flask development address is:

```text
http://127.0.0.1:5000/
```

The exact URL and port depend on the Flask application's configuration.

## 💻 Requirements

* Python compatible with the project dependencies
* pip package manager
* PyTorch
* Flask
* Dependencies listed in `requirements.txt`
* Required pretrained model weights

A CPU may be sufficient for basic experimentation. Processing speed depends on image resolution, model configuration, and available hardware.

## 🎯 Applications

* Artistic photo transformation
* Creative image editing
* Computer vision experiments
* Deep learning demonstrations
* AI-powered image processing
* Neural network research and learning

## 🔮 Future Enhancements

* Add more artistic styles and presets.
* Introduce adjustable style intensity.
* Support batch image processing.
* Add image comparison and download options.
* Optimize inference performance.
* Improve the interface for mobile devices.
* Deploy the application to a cloud platform.

## 👨‍💻 Author

**Shravan Kumar**

AI/ML Engineer | Data Science | Deep Learning | Computer Vision

* **GitHub:** [shravan9102](https://github.com/shravan9102)
* **LinkedIn:** [Shravan Kumar](https://www.linkedin.com/in/shravan91/)
* **Project Repository:** [NeuralArt AI — Neural Style Transfer](https://github.com/shravan9102/NeuralArt-AI-Neural-Style-Transfer)

## 📄 License

Add an appropriate open-source license if you intend to allow others to use, modify, and distribute this project. Ensure that the pretrained model weights and third-party assets permit your intended usage.

---

⭐ If you find this project interesting, consider starring the repository!
