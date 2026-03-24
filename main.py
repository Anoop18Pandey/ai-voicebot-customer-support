from fastapi import FastAPI, UploadFile
import whisper
from gtts import gTTS
import uuid
import tempfile

app = FastAPI()
model = whisper.load_model("base")

def predict_intent(text):
    text = text.lower()
    if "order" in text:
        return "order_status"
    elif "refund" in text:
        return "refund"
    elif "cancel" in text:
        return "cancel_order"
    else:
        return "general_query"

def generate_response(intent):
    responses = {
        "order_status": "Your order is currently in transit.",
        "refund": "Your refund will be processed within 5 days.",
        "cancel_order": "Your order has been cancelled successfully.",
        "general_query": "Please provide more details."
    }
    return responses.get(intent, "Sorry, I didn't understand.")

@app.post("/voicebot")
async def voicebot(file: UploadFile):
    audio = await file.read()

    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(audio)
        result = model.transcribe(f.name)

    text = result["text"]
    intent = predict_intent(text)
    response = generate_response(intent)

    filename = f"{uuid.uuid4()}.mp3"
    tts = gTTS(response)
    tts.save(filename)

    return {
        "text": text,
        "intent": intent,
        "response": response,
        "audio_file": filename
    }
