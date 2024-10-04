
https://github.com/user-attachments/assets/594a7122-250e-4291-9af5-dcc6d7af66fd
 # Speech-to-Text and Sentiment Analysis Application

![image](https://github.com/user-attachments/assets/f8fa4ac8-c18c-4900-8ca9-603e6aeffe92)



## Overview

This project is a web application that takes an audio file as input (in either Arabic or English) and processes it to perform speech-to-text conversion. It includes a language detection module to determine the spoken language, and based on the detected language, it translates the text and performs sentiment analysis. The application is built with a front-end interface and deployed on Azure.

## Features

- **Speech-to-Text Conversion**: Convert audio files into text format.
- **Language Detection**: Automatically detect the language of the spoken content (Arabic or English).
- **Translation**: Translate the detected text to the specified language.
- **Sentiment Analysis**: Analyze the sentiment of the translated text.
- **User-Friendly Frontend**: A simple interface for users to upload audio files and receive results.


## Project Structure
```
TELEK-APP/
├── env/                # Virtual environment for dependencies
├── templates/          # HTML templates for the web interface
├── app.py              # Main application file
└── requirements.txt    # Required Python packages
```

## UI
![image](https://github.com/user-attachments/assets/64c2d670-e490-4ed9-bd2e-3c765568ae99)



## DEMO

https://github.com/user-attachments/assets/458edfc6-833e-423f-b8d6-debbb0b6f8b6



https://github.com/user-attachments/assets/05e25e76-47fe-4c03-8b1b-a942893e418f





## Requirements

To install the necessary packages, run:

```bash
pip install -r requirements.txt
```

Install ffm (if not installed on the machine):

```bash
sudo apt update
sudo apt install ffmpeg
ffmpeg -version
```

Activate the virtual environment:

```bash
source env/bin/activate
```



Then run the application:

```bash
python app.py
```

## Deployment
This application is deployed on Azure. For details on deployment, refer to Azure documentation or check the deployment settings in the app. http://48.217.82.28:8080


## Usage
Upload an audio file (in Arabic or English) through the web interface.
The application will convert the audio to text, detect the language, translate the text, and perform sentiment analysis.
The results will be displayed on the interface.



<!-- Contributors -->
## <img  align= center width=50px height=50px src="https://media1.giphy.com/media/WFZvB7VIXBgiz3oDXE/giphy.gif?cid=6c09b952tmewuarqtlyfot8t8i0kh6ov6vrypnwdrihlsshb&rid=giphy.gif&ct=s"> Contributors <a id = "contributors"></a>

<!-- Contributors list -->
<table align="center" >
  <tr>
        <td align="center"><a href="https://github.com/MightyMaya"><img src="https://avatars.githubusercontent.com/u/130902434?v=4" width="150px;" alt=""/><br /><sub><b>Maya</b></sub></a><br /></td>
        <td align="center"><a href="https://github.com/Reem463"><img src="https://avatars.githubusercontent.com/u/181993417?v=4" width="150px;" alt=""/><br /><sub><b>Reem Al Ghazali </b></sub></a><br /></td>
        <td align="center"><a href="https://github.com/BasmaElhoseny01"><img src="https://avatars.githubusercontent.com/u/72309546?v=4" width="150px;" alt=""/><br /><sub><b>Basma Elhoseny</b></sub></a><br /></td>
        <td align="center"><a href="https://github.com/MrJouH4"><img src="https://avatars.githubusercontent.com/u/75612905?v=4" width="150px;" alt=""/><br /><sub><b>Youssef Hisham</b></sub></a><br /></td>
        <td align="center"><a href="https://github.com/ZahyElgendy"><img src="https://avatars.githubusercontent.com/u/145224435?v=4" width="150px;" alt=""/><br /><sub><b>Zahy Elgendy</b></sub></a><br /></td>
        <td align="center"><a href="https://github.com/Usama-Mohammed"><img src="https://avatars.githubusercontent.com/u/181756088?v=4" width="150px;" alt=""/><br /><sub><b>Usama Mohammed</b></sub></a><br /></td>
  </tr>
</table>
