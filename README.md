🌸 Flower Analyzer
Flower Analyzer is a FastAPI-based web application that:

Predicts the type and color of flowers from an uploaded image.

Estimates concentrations of essential oils: Linalool, Geraniol, and Citronellol.

Includes a trained multi-output PyTorch model and a lightweight frontend.

🔧 Project Structure
Flower_Analyser/
├── Common/                  # Static frontend files (HTML)
├── datasets/
│   └── flower_dataset.py    # Dataset & transformations
├── models/
│   └── model.py             # PyTorch multi-output model
├── logs/                    # Runtime logs
├── best_model.pth           # Trained PyTorch model weights
├── main.py                  # FastAPI app
├── train.py                 # Model training script
└── README.md

🚀 Features
Image Upload via UI or API

Deep Learning Model with multi-output classification + regression:

Flower Type: daisy, dandelion, rose, sunflower, tulip

Flower Color: white, yellow, red, pink

Oil Concentrations: predicted in float values

HTML UI for demo interface

Log Tracking with timestamped logs for every prediction

🖥️ Installation

# Clone the repo
git clone https://github.com/laxparihar/Flower_Analyser.git
cd Flower_Analyser

# Install dependencies
pip install -r requirements.txt

🧠 Model Training

python train.py --data_dir data --epochs 10 --batch_size 32

🌐 Running the API
uvicorn main:app --host 0.0.0.0 --port 8080
