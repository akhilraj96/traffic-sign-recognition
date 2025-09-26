# 🚦 Traffic Sign Recognition

A full-stack project for **Traffic Sign Recognition** using **Deep Learning (TensorFlow/Keras)**, a **FastAPI backend**, and a **React frontend**. The model is trained on the GTSRB dataset and deployed via a REST API.

---

## 📂 Project Structure

```
├── backend
│   ├── app.py                  # FastAPI application
│   ├── Dockerfile              # Backend Dockerfile
│   ├── requirements.txt        # Backend dependencies
│   ├── setup.py                # Packaging
│   ├── view_dataset.py         # Dataset visualization
│   ├── artifacts/              # Preprocessed data & saved models
│   │   ├── train.p
│   │   ├── test.p
│   │   └── Saved_Models/
│   │       ├── LeNet.h5
│   │       └── VGGNet.h5
│   └── src/                    # Source code
│       ├── components/         # Data processing & model training
│       ├── pipeline/           # Training & prediction pipelines
│       ├── exception.py
│       ├── logger.py
│       └── utils.py
│
└── frontend
    ├── App.jsx                 # React main app
    ├── components/             # UI components
    ├── public/                 # Static assets
    └── package.json            # Frontend dependencies
```

---

## ⚙️ Setup (with Conda)

```bash
# Create and activate conda environment
conda create -n tsr_env python=3.10 -y
conda activate tsr_env

# Install backend dependencies
cd backend
pip install -e .

# Install frontend dependencies
cd ../frontend
npm install
```

---

## 🧠 Training the Model

Make sure your dataset is in the `data/` directory, then run:

```bash
cd backend
python src/pipeline/train_pipeline.py
```

The trained models will be saved under `backend/artifacts/Saved_Models/`.

---

## 🚀 Running the Backend (FastAPI)

Start the API server:

```bash
cd backend
uvicorn app:app --reload --port 8000
```

API will be available at: [http://localhost:8000](http://localhost:8000)

📌 Example request using `cURL`:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test_image.png"
```

deployed backend: [https://traffic-sign-recognition-frby.onrender.com](https://traffic-sign-recognition-frby.onrender.com)

---

## 🎨 Running the Frontend (React + Vite)

```bash
cd frontend
npm run dev
```

Frontend will be available at: [http://localhost:5173](http://localhost:5173)

deployed frontend: [https://traffic-sign-recognition-six.vercel.app](https://traffic-sign-recognition-six.vercel.app)

---

## 🐳 Docker Deployment

### Build and Run Backend

```bash
cd backend
docker build -t traffic-sign-backend .
docker run -p 8000:8000 traffic-sign-backend
```

### Build and Run Frontend

```bash
cd frontend
docker build -t traffic-sign-frontend .
docker run -p 5173:5173 traffic-sign-frontend
```

---

## 📦 Deployment Steps

1. **Backend (Render)**

   * Push backend to GitHub.
   * Create a new Web Service on [Render](https://render.com/).
   * Connect your GitHub repository.
   * Set the start command: `uvicorn app:app --host 0.0.0.0 --$PORT`.
   * Render will automatically build and deploy the backend.

2. **Frontend (Vercel)**

   * Push frontend to GitHub.
   * Create a new project on [Vercel](https://vercel.com/).
   * Connect your GitHub repository.
   * Set the framework to React.
   * Vercel will automatically build and deploy the frontend.

---

## ✅ Features

* Preprocessing & data augmentation
* Model training (LeNet, VGGNet)
* REST API for inference
* React frontend for interactive predictions
* Dockerized deployment

---

## 👨‍💻 Author

**Akhil Raj**
📧 [akhil\_raj@outlook.com](mailto:akhil_raj@outlook.com)
