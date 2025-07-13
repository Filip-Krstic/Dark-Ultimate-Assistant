import os
import json
import vosk
import cv2
import argparse
import threading
import pyttsx3
import pyaudio
import time
import wikipediaapi
import numpy as np
import noisereduce as nr
from datetime import datetime
from time import sleep
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from fuzzywuzzy import process
from datetime import datetime
from playsound import playsound


# Initialize the backup text-to-speech engine
class _TTS:
    engine = None
    rate = None
    def __init__(self):
        self.engine = pyttsx3.init()
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[1].id)
    def start(self,text_):
        self.engine.say(text_)
        self.engine.runAndWait()



def speak_text(command):
    tts = _TTS()
    tts.start(command)
    del(tts)


def process_video(image=None):
    # Initialize variables for face tracking
    previous_face_ids = set()

    def get_response(age_group, gender):
        key = f"({age_group}){gender}"
        try:
            with open('datadrk/genfacemodel/thknw.mind', 'r') as file:
                lines = [line.strip() for line in file if line.startswith(key)]
            
            if lines:
                response = random.choice(lines).split(';', 1)[1].strip()
                return response
            else:
                return "No responses found for this age group and gender."
        except FileNotFoundError:
            return "The responses file was not found."
        except Exception as e:
            return f"An error occurred: {e}"

    
    def process_faces(frame, faceBoxes, genderList, ageList, padding):
        nonlocal previous_face_ids

        current_face_ids = set()  # To store current frame's face IDs

        for faceBox in faceBoxes:
            face_id = faceBox[4]  # Get the face ID

            # Extract face ROI
            face = frame[max(0, faceBox[1] - padding):
                         min(faceBox[3] + padding, frame.shape[0] - 1), max(0, faceBox[0] - padding)
                             :min(faceBox[2] + padding, frame.shape[1] - 1)]

            # Process gender
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]

            # Process age
            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]

            # Store the face ID in current IDs set
            current_face_ids.add(face_id)

            # Only process if it's a new face ID
            if face_id not in previous_face_ids:
                age_group = f"{age[1:-1]}"
                if gender == "male" or gender == "Male":
                    gender = "M"
                elif gender == "female" or gender == "Female":
                    gender = "F"
                else:
                    gender = "F" # all humans start off as females
                # Speak text in a separate thread
                if age_group and gender:
                    response = get_response(age_group, gender)
                    speak_text(response)
                #speak_text(f'Hello {gender} human, how are you? Is your age around {age[1:-1]} years')

        # Update previous face IDs with current ones
        previous_face_ids.update(current_face_ids)

    def highlightFace(net, frame, conf_threshold=0.7):
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

        net.setInput(blob)
        detections = net.forward()
        faceBoxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                face_id = i  # Assign a unique ID to each detected face
                faceBoxes.append([x1, y1, x2, y2, face_id])
        return frameOpencvDnn, faceBoxes

    parser = argparse.ArgumentParser()
    parser.add_argument('--image')
    args = parser.parse_args()

    faceProto = "datadrk/genfacemodel/opencv_face_detector.pbtxt"
    faceModel = "datadrk/genfacemodel/opencv_face_detector_uint8.pb"
    ageProto = "datadrk/genfacemodel/age_deploy.prototxt"
    ageModel = "datadrk/genfacemodel/age_net.caffemodel"
    genderProto = "datadrk/genfacemodel/gender_deploy.prototxt"
    genderModel = "datadrk/genfacemodel/gender_net.caffemodel"

    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)

    video = cv2.VideoCapture(args.image if args.image else 0)
    padding = 20

    while True:
        hasFrame, frame = video.read()
        if not hasFrame:
            #print("No frame available")
            break

        resultImg, faceBoxes = highlightFace(faceNet, frame)
        if not faceBoxes:
            #print("No face detected")
            continue

        # Process faces
        process_faces(resultImg, faceBoxes, genderList, ageList, padding)
        sleep(0.5)

    video.release()


def local_objects(ask):
    thres = 0.55  # Threshold to detect object
    nms_threshold = 0.2  # Non-max suppression threshold

    # Set up the webcam
    try:
        cap = cv2.VideoCapture(0)  # Use 0 for the built-in webcam, 1 for an external webcam
    except:
        cap = cv2.VideoCapture(1)
    # Set camera properties (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

    classNames = []
    with open('datadrk/localvision/k3.mind', 'r') as f:
        classNames = f.read().splitlines()

    font = cv2.FONT_HERSHEY_PLAIN
    Colors = np.random.uniform(0, 255, size=(len(classNames), 3))

    weightsPath = "datadrk/localvision/k1.pb"
    configPath = "datadrk/localvision/k2.pbtxt"
    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    # Capture a single frame from the webcam
    ret, img = cap.read()
    if not ret:
        print("Error: Could not read frame from camera.")
        cap.release()
        exit()

    # Detect objects in the captured image
    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1, -1)[0])
    confs = list(map(float, confs))

    # Apply Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

    current_obj = []
    if len(classIds) != 0:
        for i in indices:
            box = bbox[i]
            color = Colors[classIds[i] - 1]
            confidence = str(round(confs[i], 2))
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness=2)
            cv2.putText(img, classNames[classIds[i] - 1] + " " + confidence, (x + 10, y + 20),
                        font, 1, color, 2)
            current_obj.append(f"{classNames[classIds[i] - 1]}")

    cap.release()
    cv2.destroyAllWindows()

    user_input = ""
    for i in current_obj:
        #print(i)
        user_input = user_input + " " + i

    # Define categories
    categories = {}
    file_path = 'datadrk/localvision/k4.mindg'
    with open(file_path, 'r') as file:
        contents = file.read()
        try:
            categories = eval(contents)
        except Exception as e:
            print(f"Error parsing the file: {e}")



    # Function to categorize input
    def categorize_input(input_string):
        found_items = {category: [] for category in categories}
        words = input_string.lower().split()

        for category, items in categories.items():
            for item in items:
                if item in words:
                    found_items[category].append("".join(item))

        return found_items

    # Categorize and output results
    result = categorize_input(user_input)

    ve = []
    an = []
    cl = []
    sp = []
    fo = []
    fu = []
    el = []
    mi = []
    le = []
    hy = []
    we = []
    for category, items in result.items():
        if items:
            if category == "Vehicles":#
                ve.append(f"{items}")
            elif category == "Animals":#
                an.append(f"{items}")
            elif category == "Clothes":#
                cl.append(f"{items}")
            elif category == "Sports":#
                sp.append(f"{items}")
            elif category == "Food": #
                fo.append(f"{items}")
            elif category == "Furniture":#
                fu.append(f"{items}")
            elif category == "Electronics":#
                el.append(f"{items}")
            elif category == "Miscellaneous":#
                mi.append(f"{items}")
            elif category == "Leisure":#
                le.append(f"{items}")
            elif category == "Hygiene":#
                hy.append(f"{items}")
            elif category == "Weapons":#
                we.append(f"{items}")

    sve = str(ve).replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    san = str(an).replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    scl = str(cl).replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    ssp = str(sp).replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    sfo = str(fo).replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    sfu = str(fu).replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    sel = str(el).replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    smi = str(mi).replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    sle = str(le).replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    shy = str(hy).replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    swe = str(we).replace("[", "").replace("]", "").replace("'", "").replace('"', "")


    vowels = ["a","e","i","o","u"]
    found = 0

    if "transport" in ask:
        if sve == "":
            speak_text("i dont see any modes of transportation")
        elif "," in sve:
            speak_text("the modes of transportation i see are " + sve)
        else:
            found = any(i in sve[0] for i in vowels)
            if found:
                speak_text("the mode of transportation i see is an " + sve)
            else:
                speak_text("the mode of transportation i see is a " + sve)

    elif "vehicle" in ask:
        if sve == "":
            speak_text("i dont see any vehicles")
        elif "," in sve:
            speak_text("the vehicles i see are " + sve)
        else:
            found = any(i in sve[0] for i in vowels)
            if found:
                speak_text("the vehicle i see is an " + sve)
            else:
                speak_text("the vehicle i see is a " + sve)

    elif "pet" in ask:
        if san == "":
            speak_text("i dont see any pets")
        elif "," in san:
            speak_text("the pets that i see are " + san)
        else:
            found = any(i in san[0] for i in vowels)
            if found:
                speak_text("the pet i see is an " + san)
            else:
                speak_text("the pet i see is a " + san)

    elif "animal" in ask:
        if san == "":
            speak_text("i dont see any animals")
        elif "," in san:
            speak_text("the animals that i see are " + san)
        else:
            found = any(i in san[0] for i in vowels)
            if found:
                speak_text("the animal i see is an " + san)
            else:
                speak_text("the animal i see is a " + san)

    elif "wear" in ask:
        if scl == "":
            speak_text("i dont see anything you can wear")
        elif "," in scl:
            speak_text("the things that you could wear that i see are " + scl)
        else:
            found = any(i in scl[0] for i in vowels)
            if found:
                speak_text("the item you can wear is an " + scl)
            else:
                speak_text("the item you can wear is a " + scl)

    elif "cloth" in ask:
        if scl == "":
            speak_text("i dont see any clothes")
        elif "," in scl:
            speak_text("the clothes that i see are " + scl)
        else:
            found = any(i in scl[0] for i in vowels)
            if found:
                speak_text("the cloth i see is an " + scl)
            else:
                speak_text("the cloth i see is a " + scl)

    elif "sport" in ask:
        if ssp == "":
            speak_text("i dont see anything that you can use for sports")
        elif "," in ssp:
            speak_text("the things that are used for sports and that i see are " + ssp)
        else:
            found = any(i in ssp[0] for i in vowels)
            if found:
                speak_text("the item used for sports is an " + ssp)
            else:
                speak_text("the item used for sports is a " + ssp)

    elif "food" in ask:
        if sfo == "":
            speak_text("i dont see any food")
        elif "," in sfo:
            speak_text("the food that i see are " + sfo)
        else:
            found = any(i in sfo[0] for i in vowels)
            if found:
                speak_text("the food i see is an " + sfo)
            else:
                speak_text("the food i see is a " + sfo)

    elif "nutrient" in ask:
        if sfo == "":
            speak_text("i dont see any items with nutritional value")
        elif "," in sfo:
            speak_text("the items with nutritional content that i see are " + sfo)
        else:
            found = any(i in sfo[0] for i in vowels)
            if found:
                speak_text("the nutrient i see is an " + sfo)
            else:
                speak_text("the nutrient i see is a " + sfo)

    elif "furniture" in ask:
        if sfu == "":
            speak_text("i dont see any furniture")
        elif "," in sfu:
            speak_text("the furniture that i see are " + sfu)
        else:
            found = any(i in sfu[0] for i in vowels)
            if found:
                speak_text("the furniture i see is an " + sfu)
            else:
                speak_text("the furniture i see is a " + sfu)

    elif "electronic" in ask:
        if sel == "":
            speak_text("i dont see any electronics")
        elif "," in sel:
            speak_text("the electronics that i see are " + sel)
        else:
            found = any(i in sel[0] for i in vowels)
            if found:
                speak_text("the electronic i see is an " + sel)
            else:
                speak_text("the electronic i see is a " + sel)

    elif "uncategorised" in ask:
        if smi == "":
            speak_text("i dont see any uncategorised items")
        elif "," in smi:
            speak_text("the uncategorised items that i see are " + smi)
        else:
            found = any(i in smi[0] for i in vowels)
            if found:
                speak_text("the uncategorised item i see is an " + smi)
            else:
                speak_text("the uncategorised item i see is a " + smi)

    elif "relaxing" in ask:
        if sle == "":
            speak_text("i dont see any items that can be used to relax")
        elif "," in sle:
            speak_text("the items for relaxing that i see are " + sle)
        else:
            found = any(i in sle[0] for i in vowels)
            if found:
                speak_text("the relaxing item i see is an " + sle)
            else:
                speak_text("the relaxing item i see is a " + sle)

    elif "leisure" in ask:
        if sle == "":
            speak_text("i dont see any items that can be used for leisure")
        elif "," in sle:
            speak_text("the items that are used for leisure that i see are " + sle)
        else:
            found = any(i in sle[0] for i in vowels)
            if found:
                speak_text("the leisure item i see is an " + sle)
            else:
                speak_text("the leisure item i see is a " + sle)

    elif "hygiene" in ask:
        if shy == "":
            speak_text("i dont see any items that can be used for hygiene")
        elif "," in shy:
            speak_text("the items that are used for hygiene that i see are " + shy)
        else:
            found = any(i in shy[0] for i in vowels)
            if found:
                speak_text("the hygiene item i see is an " + shy)
            else:
                speak_text("the hygiene item i see is a " + shy)

    elif "health" in ask:
        if shy == "":
            speak_text("i dont see any items that can be used for health")
        elif "," in shy:
            speak_text("the items that are used for health that i see are " + shy)
        else:
            found = any(i in shy[0] for i in vowels)
            if found:
                speak_text("the health item i see is an " + shy)
            else:
                speak_text("the health item i see is a " + shy)

    elif "danger" in ask:
        if swe == "":
            speak_text("i dont see any items that can endanger you")
        elif "," in swe:
            speak_text("the items that you can be harmed with that i see are " + swe)
        else:
            found = any(i in swe[0] for i in vowels)
            if found:
                speak_text("the dangerous item i see is an " + swe)
            else:
                speak_text("the dangerous item i see is a " + swe)

    elif "weapon" in ask:
        if swe == "":
            speak_text("i dont see any items that can be used as weapons")
        elif "," in swe:
            speak_text("the items that can be used as weapons that i see are " + swe)
        else:
            found = any(i in swe[0] for i in vowels)
            if found:
                speak_text("the weapon i see is an " + swe)
            else:
                speak_text("the weapon i see is a " + swe)

    else:
        speak_text("i didnt find what you wanted sorry")

    

def search_knowledge_base(usr_in):
    set_emotion = ""
    found = False
    with open('datadrk/darkmodel/dark_knowledge.mind', 'r') as file:
        knowledge_base = [line.strip() for line in file if line.strip()]
        
    user_inputs = [line.split(';', 1)[0] for line in knowledge_base]
    best_match = process.extractOne(usr_in, user_inputs)
    
    if best_match and best_match[1] > 85:  # Threshold for similarity
        matched_output = knowledge_base[user_inputs.index(best_match[0])].split(';', 1)[1]
        speak_text(matched_output)
        found = True
        

    return found

def add_to_knowledge_base(usr_in):
    speak_text("um what should i reply to this")
    while True:
        data = stream.read(4000, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)
        reduced_noise = nr.reduce_noise(y=audio_data, sr=16000, prop_decrease=0.5)
        processed_data = reduced_noise.astype(np.int16).tobytes()

        if recognizer.AcceptWaveform(processed_data):
            result = recognizer.Result()
            output = json.loads(result).get("text", "").lower()

            # Remove "the " and " the" from output
            output = output.replace("the ", "").replace(" the", "").strip()
            output = output.replace("said ", "").strip()

            if output.strip():  # Check for non-empty output
                if "nothing" in output:
                    speak_text("ok i wont learn about it then") # e is kinda good
                    break
                else:
                    with open('datadrk/darkmodel/dark_knowledge.mind', 'a') as file:
                        file.write(f"\n{usr_in};{output}\n")
                    
                    print(f"Added: {usr_in};{output}")
                    break
        else:
            partial_result = recognizer.PartialResult()
            print(json.loads(partial_result).get("partial", ""))

def summarize_text(text, sentence_count=5):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    
    summary = summarizer(parser.document, sentence_count)
    return ' '.join(str(sentence) for sentence in summary)


def wikisrch(title):
    wiki_wiki = wikipediaapi.Wikipedia(
        language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI,
        user_agent="FriendlyBoi/1.0 (donotreply@gmail.com)"
    )
    
    page = wiki_wiki.page(title)

    if page.exists():
        summary = page.summary
        return summary
    else:
        return "im sorry but i cant search now"

# Initialize Vosk model
model_path = 'datadrk/vosk-model-en-us-0.22'
if not os.path.exists(model_path):
    print("Model not found, please download from https://alphacephei.com/vosk/models, use vosk-model-en-us-0.22")
    sleep(3)
    exit(1)

model = vosk.Model(model_path)

# Set up audio input
recognizer = vosk.KaldiRecognizer(model, 16000)

# Start streaming from the microphone
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
stream.start_stream()

video_thread = threading.Thread(target=process_video)
video_thread.start()

print("-<[STARTING SEQUENCE STARTED]>-")

# Loop indefinitely for user to speak
while True:
    data = stream.read(4000, exception_on_overflow=False)

    # Convert byte data to numpy array for noise reduction
    audio_data = np.frombuffer(data, dtype=np.int16)

    # Perform noise reduction
    reduced_noise = nr.reduce_noise(y=audio_data, sr=16000, prop_decrease=0.5)

    # Convert back to byte data
    processed_data = reduced_noise.astype(np.int16).tobytes()

    if recognizer.AcceptWaveform(processed_data):
        result = recognizer.Result()
        my_text = json.loads(result).get("text", "").lower()

        # Remove "the " and " the" from input
        my_text = my_text.replace("the ", "").replace(" the", "").strip()

        # Check for the wake word
        if "dark" in my_text:
            my_text = my_text.replace("dark ", "").strip()

            if "search online " in my_text:
                # Search Wikipedia
                my_text = my_text.replace("search online ", "").strip()
                c = summarize_text(wikisrch(my_text), sentence_count=5)
                if c != "im sorry but i cant search now":
                    speak_text(c)
                else:
                    speak_text(c)
            else:
                if "time" and "what" in my_text:
                    # check the time
                    speak_text("the time is ")
                    sleep(0.1)
                    now = time.time()
                    hour, minute = time.strftime('%H'), time.strftime('%M')
                    spcom = f"{hour}:{minute}"
                    tts = _TTS()
                    tts.start(spcom)
                    del(tts)
                else:
                    if "date" and "what" in my_text:
                        # check the date
                        speak_text("today the date is ")
                        sleep(0.1)
                        spcom = datetime.today().strftime('%Y-%m-%d')
                        tts = _TTS()
                        tts.start(spcom)
                        del(tts)
                    else:
                        if "look" and "camera"  in my_text:
                            # look for certain objects in the camera
                            local_objects(my_text)
                        else:
                            # Standard memory
                            if not search_knowledge_base(my_text):
                                add_to_knowledge_base(my_text)
        else:
            pass
            #print("Wake word not detected.")

    else:
        partial_result = recognizer.PartialResult()
        partial_text = json.loads(partial_result).get("partial", "")
        
        # Remove "the " and " the" from partial result
        partial_text = partial_text.replace("the ", "").replace(" the", "").strip()
        partial_text = partial_text.replace("said ", "").strip()
        
        # Check for the wake word in the partial result
        if "dark" in partial_text:
            print("Wake word detected:", partial_text)
        else:
            pass
            #print("Partial result:", partial_text)
