# signlang

## Setup

### Backend
- `cd backend`
- `pip install -r requirements.txt`
- `python app.py`

Backend serves:
- `GET /health` - checks model availability
- `POST /predict` - predicts ASL alphabet letter (A–Z) using Hugging Face model

### Frontend
- `cd frontend`
- `npm install`
- `npm run dev`

## What it does

- **Recognizes ASL alphabets (A–Z)** in real time using the model from:
  - `https://huggingface.co/huzaifanasirrr/realtime-sign-language-translator`
- Builds **full words** from detected letters in the UI.
- Lets you add **punctuation** (`. , ? ! :`) directly in the UI.