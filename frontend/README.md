# Traffic Sign Recognition App

This repository contains a **traffic sign recognition application** with a FastAPI backend and a React frontend. The app supports both **image uploads** and **live camera recognition** for predicting traffic signs using a trained deep learning model.

---

## Features

* Predict traffic signs from uploaded images.
* Live camera recognition with periodic frame prediction.
* Dark theme frontend with drag-and-drop image support.
* Confidence display for predictions.
* Easy deployment using Vercel (frontend) and any FastAPI-supported host (backend).

---

## Tech Stack

* **Backend:** FastAPI, TensorFlow, Pillow (PIL), Python 3.9+
* **Frontend:** React, TailwindCSS, Vite
* **Model:** Trained TensorFlow Keras model (`.h5`) with class mapping (`classes.json`)

---

## Setup

### Backend

1. **Install dependencies:**

```bash
pip install fastapi uvicorn tensorflow pillow python-multipart
```

2. **Set environment variables (optional):**

```bash
export MODEL_PATH=traffic_sign_model.h5
export CLASSES_PATH=classes.json
```

3. **Run backend:**

```bash
uvicorn view_dataset:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

1. **Install dependencies:**

```bash
npm install
```

2. **Set API URL:**

Create `.env` in frontend:

```
VITE_API_URL=http://localhost:8000
```

3. **Run frontend:**

```bash
npm run dev
```

---

## Usage

* **Image Upload:** Click or drag an image into the upload section and click **Predict**.
* **Live Camera:** Click **Start Live Recognition** to enable your webcam. The app will automatically capture frames every 2 seconds and predict traffic signs.
* **Results:** The predicted class and confidence percentage are displayed below the input.

---

## Deployment

### Frontend (Vercel)

1. Push the frontend code to a GitHub repository.
2. Sign in to [Vercel](https://vercel.com/) and import your repository.
3. Set the environment variable `VITE_API_URL` to your deployed backend URL.
4. Deploy.

### Backend

* Can be deployed on any server supporting FastAPI (e.g., **Render**, **AWS EC2**, **Heroku**, **Google Cloud Run**).
* Ensure the backend URL is accessible from the frontend.

---

## Notes

* The app currently supports `.png`, `.jpg`, `.jpeg` images.
* Live camera prediction interval is set to **2 seconds**, which can be adjusted.
* The backend model must be trained to recognize traffic signs corresponding to your class mapping.

---

## License

This project is licensed under the [MIT License](LICENSE)
