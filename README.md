# Dark - AI Voice Assistant with Object Detection

This project develops an advanced AI voice assistant capable of real-time object detection, voice interaction, knowledge base management, and web search integration. Designed to be a versatile personal assistant, learning tool, and accessibility aid, it leverages deep learning for computer vision and speech recognition for natural user interaction.

## Features

* **Real-Time Object Detection**: Utilizes a webcam to capture video frames and detect objects in real-time using a deep learning model. [cite_start]It identifies various classes of objects based on predefined labels[cite: 1, 2].
* [cite_start]**Voice Interaction**: Listens for voice commands and responds accordingly, processing audio input with the Vosk speech recognition library[cite: 3].
* [cite_start]**Categorization of Detected Objects**: Detected objects are categorized into predefined categories (like Vehicles, Animals, Food, etc.) and the system can respond to user queries related to these categories[cite: 5]. [cite_start]It can provide singular or plural responses based on the number of objects detected[cite: 6].
* [cite_start]**Knowledge Base Interaction**: Maintains a knowledge base that allows it to respond to user queries by retrieving information from a text file[cite: 7]. [cite_start]Users can also add new entries to the knowledge base through voice input, enhancing the system's ability to learn and remember[cite: 8].
* [cite_start]**Web Search Integration**: Allows users to search Wikipedia through voice commands and summarizes the information retrieved[cite: 9]. [cite_start]It uses natural language processing to parse and respond to user queries effectively[cite: 10].
* [cite_start]**Time and Date Information**: Can provide the current time and date upon request[cite: 11].
* [cite_start]**Noise Reduction**: Implements noise reduction on the audio input to improve the accuracy of speech recognition, especially useful in noisy environments[cite: 12].
* [cite_start]**Threading**: Runs video processing in a separate thread, allowing for simultaneous audio processing and object detection without significant lag[cite: 13].

## Use Cases

* [cite_start]**Personal Assistant**: Can act as a personal assistant that interacts with users by identifying objects around them and answering questions based on those objects[cite: 14].
* [cite_start]**Learning Tool**: Can be used as an educational tool to teach users about different objects and categories[cite: 15].
* [cite_start]**Accessibility Aid**: Useful for visually impaired individuals by providing audio feedback on the environment[cite: 16].
* [cite_start]**Smart Home Integration**: Can potentially be integrated into smart home systems to help users identify objects or get information about their surroundings[cite: 17].

## Technologies Used

* **Python**: Primary programming language.
* **Vosk**: Offline speech recognition library.
* **OpenCV (`cv2`)**: For real-time video capture and object detection.
* **`pyttsx3`**: Python text-to-speech conversion library.
* **`pyaudio`**: For audio input from the microphone.
* **`wikipediaapi`**: Python wrapper for the Wikipedia API.
* **`numpy`**: Numerical computing library.
* **`noisereduce`**: Noise reduction for audio processing.
* **`sumy`**: Library for automatic text summarization.
* **`fuzzywuzzy`**: For fuzzy string matching in the knowledge base.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Vosk Speech Recognition Model:**
    * Download the `vosk-model-en-us-0.22` model from [https://alphacephei.com/vosk/models](https://alphacephei.com/vosk/models).
    * Extract the downloaded archive.
    * **Place the extracted `vosk-model-en-us-0.22` folder directly into the `datadrk/` directory.** Your path should look like `datadrk/vosk-model-en-us-0.22/`.

## Usage

1.  **Ensure your webcam and microphone are connected and accessible.**
2.  **Run the main script:**
    ```bash
    python main.py
    ```
3.  **To interact with the AI, start your command by saying "Dark".**
    * **Examples:**
        * "Dark, what time is it?"
        * "Dark, what is the date?"
        * "Dark, look at the camera for animals." (or "vehicles", "food", etc.)
        * "Dark, search online for artificial intelligence."
        * "Dark, who is the president of the United States?" (This will trigger a knowledge base search, or you can teach it new facts if not found.)

## Project Structure

├── main.py
├── README.md
├── requirements.txt
├── LICENSE
└── datadrk/
├── darkmodel/
│   └── dark_knowledge.mind
├── genfacemodel/
│   ├── age_deploy.prototxt
│   ├── age_net.caffemodel
│   ├── gender_deploy.prototxt
│   ├── gender_net.caffemodel
│   ├── opencv_face_detector.pbtxt
│   ├── opencv_face_detector_uint8.pb
│   └── thknw.mind
├── localvision/
│   ├── k1.pb
│   ├── k2.pbtxt
│   ├── k3.mind
│   └── k4.mindg
└── vosk-model-en-us-0.22/ (This directory should be placed here after download)
└── ... (Vosk model files)


## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
