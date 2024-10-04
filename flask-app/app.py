# Define the FLask app
from flask import Flask, jsonify, request,render_template
from flask_cors import CORS

from pydub import AudioSegment

from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForSequenceClassification
from IPython.display import Audio, display
import librosa  # Library to process audio files
from langdetect import detect  # Language detection library

import torch
import os

app = Flask(__name__)
# CORS(app)

# Enable CORS for all routes and restrict to specific origins
CORS(app, resources={r"/*": {"origins": "*"}})


# Specify the upload folder and allowed file extensions
UPLOAD_FOLDER = 'uploads'  # Change this to your desired upload folder
CONVERTED_FOLDER = 'converted'  # Folder to save converted files
# ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'm4a'}

# Create the upload and converted folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CONVERTED_FOLDER, exist_ok=True)


speech2TextPipeline = None

translateEnglish2ArabicPipeline = None
translateArabic2EnglishPipeline = None

sentimentArabicPipeline = None
sentimentEnglishPipeline = None


# Set the device to 'cuda' if a GPU is available, otherwise use 'cpu'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


DEBUG= True


label2word = {
    "NEG": "negative",
    "NEU": "neutral",
    "POS": "positive"
}

def init_models():
    # Speech2Text
    audio2text_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-medium", device=DEVICE)


    # Transaltion            
    checkpoint = 'facebook/nllb-200-distilled-600M'
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)



    # English --> Arabic
    english2arabic_translation_pipe = pipeline('translation',
                                model=model,
                                tokenizer=tokenizer,
                                src_lang="eng_Latn",
                                tgt_lang="arb_Arab",
                                max_length=400,
                                device=DEVICE)
        
    # Arabic --> English

    arabic2english_translation_pipe = pipeline('translation',
                            model=model,
                            tokenizer=tokenizer,
                            src_lang="arb_Arab",
                            tgt_lang="eng_Latn",
                            max_length=400,
                            device=DEVICE)
    



    # Sentiment
    # Arabic 
    arabic_model_name = "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
    arabic_tokenizer = AutoTokenizer.from_pretrained(arabic_model_name)
    arabic_model = AutoModelForSequenceClassification.from_pretrained(arabic_model_name)
    arabic_sentiment_pipe = pipeline('sentiment-analysis', model=arabic_model, tokenizer=arabic_tokenizer, device=DEVICE)


    # English
    english_model_name = "finiteautomata/bertweet-base-sentiment-analysis"
    english_tokenizer = AutoTokenizer.from_pretrained(english_model_name)
    english_model = AutoModelForSequenceClassification.from_pretrained(english_model_name)
    english_sentiment_pipe = pipeline('sentiment-analysis', model=english_model, tokenizer=english_tokenizer, device=DEVICE)


    return   audio2text_pipe,  english2arabic_translation_pipe, arabic2english_translation_pipe, arabic_sentiment_pipe,  english_sentiment_pipe


# Function to load and preprocess the audio file from a specific path
def preprocess_audio(audio_path):
    audio_sample_array, sampling_rate = librosa.load(audio_path, sr=None)  # Load the audio file with original sampling rate
    return audio_sample_array, sampling_rate



# Function 1: Speech to Text using speech2TextPipeline
def speech_to_text(audio_sample_array, sampling_rate):
    result = speech2TextPipeline({"array": audio_sample_array, "sampling_rate": sampling_rate}, max_new_tokens=256)
    return result['text']



# Function 2: to detect language
def detect_language(text):
    return detect(text)



# Function 3: Translate Text (English to Arabic) using translateEnglish2ArabicPipeline
def translate_text_to_arabic(text):
    translation = translateEnglish2ArabicPipeline(text)
    return translation[0]['translation_text']

# Function 4: Translate Text (Arabic to English) using translateArabic2EnglishPipeline
def translate_text_to_english(text):
    translation = translateArabic2EnglishPipeline(text)
    return translation[0]['translation_text']



# Function 5: Sentiment Analysis (Arabic) using sentimentArabicPipeline
def sentiment_analysis_arabic(text):
    sentiment = sentimentArabicPipeline(text)
    return sentiment

# Function 6: Sentiment Analysis (English) using pre-trained sentimentEnglishPipeline
def sentiment_analysis_english(text):
    sentiment = sentimentEnglishPipeline(text)
    return sentiment



@app.route("/")
def frontCode():
    return render_template('front/ui.html')

@app.route("/hello")
def hello():
    # response.headers.add('Access-Control-Allow-Origin', '*')
    return "Hello world , from Flask"


@app.route('/run', methods=['POST'])
def Pipline():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file part in the request'}), 400

    file = request.files['audio']


    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    

    # Save the original file
    original_filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(original_filepath)

    transcript = None
    translation = None
    sentiment = None


    # Convert the audio file to .wav format
    try:
        # Step 1 Load Audio File Sent
        audio = AudioSegment.from_file(original_filepath)  # Load the file
        converted_filename = f"{os.path.splitext(file.filename)[0]}.wav"
        converted_filepath = os.path.join(CONVERTED_FOLDER, converted_filename)
        audio.export(converted_filepath, format='wav')  # Export as .wav


        # Step 2: Preprocess the audio
        audio_sample_array, sampling_rate = preprocess_audio(converted_filepath)


        # Step 3: Convert the speech to text
        transcript = speech_to_text(audio_sample_array, sampling_rate)
        if DEBUG:
            print("Speech to Text Output:", transcript, "\n\n")

        # Step 4: Detect language of the transcribed text
        detected_language = detect_language(transcript)
        if DEBUG:
            print("Detected Language:", detected_language, "\n\n")


        # Step 5: Route based on detected language
        if detected_language == 'en':  # English language detected
            if DEBUG:
                print("Processing English pipeline...")

            # Step 6: Translate Text (English to Arabic)
            translation = translate_text_to_arabic(transcript)
            if DEBUG:
                print("Translated Text (English to Arabic):", translation, "\n\n")

            # Step 7: Sentiment Analysis (Arabic)
            sentiment = sentiment_analysis_arabic(translation)
            sentiment = f"{sentiment[0]['label']} with score {sentiment[0]['score']}"  # CHECK  NEU With score 0.8
            if DEBUG:
                print("Sentiment Analysis (Arabic):", sentiment)

        elif detected_language == 'ar':  # Arabic language detected
            print("Processing Arabic pipeline...")
        
            # Step 6: Translate Text (Arabic to English)
            translation = translate_text_to_english(transcript)
            if DEBUG:
                print("Translated Text (Arabic to English):", translation ,"\n\n")

            # Step 7: Sentiment Analysis (English)
            sentiment = sentiment_analysis_english(translation)
            sentiment = f"{label2word[sentiment[0]['label']]} with score {sentiment[0]['score']}"  #  HECK  NEU With score 0.8
            if DEBUG:
                print("Sentiment Analysis (English):", sentiment)


        else:
            print("Unsupported language detected.")
    
        # # return jsonify({
        # #     'message': 'File uploaded and converted successfully',
        # #     'original_filename': file.filename,
        # #     'converted_filename': converted_filename
        # # }), 200

    except Exception as e:
        print("Error:",str(e))
        return jsonify({'error': f'Conversion failed: {str(e)}'}), 500

    return jsonify({"transcript":transcript,"translation":translation, "sentiment":sentiment}) , 200




if __name__ == '__main__':
    print("Flask App Satrted :D")

    print("Running on ", DEVICE)

    # Load Models
    print("Loading Piplines ....")
    speech2TextPipeline, translateEnglish2ArabicPipeline, translateArabic2EnglishPipeline, sentimentArabicPipeline, sentimentEnglishPipeline =init_models()

    # Run
    app.run(host='0.0.0.0', port=8080, debug=True)



# python3 -m venv env
#  .\env\Scripts\activate
# deactivate
