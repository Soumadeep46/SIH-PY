import streamlit as st
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime, date
import pandas as pd
import requests
from PIL import Image
import io
from dotenv import load_dotenv
import time
import queue
import json
import threading

# Load environment variables
load_dotenv()
PINATA_API_KEY = os.getenv("PINATA_API_KEY")
PINATA_API_SECRET = os.getenv("PINATA_API_SECRET")
PINATA_GROUP_ID = "c3f85477-d28f-4c60-aafe-0729afe405ef"  # SIH group ID

# Streamlit page config
st.set_page_config(page_title="MP Police Attendance System", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main { padding-top: 2rem; }
    .stButton>button { width: 100%; margin-bottom: 1rem; }
    .upload-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        background-color: #e8f0fe;
    }
    .current-time {
        font-size: 1.2rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background-color: #1e1e1e;
        color: white;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .recognition-time {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
    .uploaded-photos {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 1rem;
    }
    .uploaded-photo {
        width: 100px;
        height: 100px;
        object-fit: cover;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialization_done' not in st.session_state:
    st.session_state.initialization_done = False
if 'encode_list_known' not in st.session_state:
    st.session_state.encode_list_known = []
if 'class_names' not in st.session_state:
    st.session_state.class_names = []
if 'marked_names' not in st.session_state:
    st.session_state.marked_names = set()
if 'last_pinata_check' not in st.session_state:
    st.session_state.last_pinata_check = None
if 'pinata_hashes' not in st.session_state:
    st.session_state.pinata_hashes = set()
if 'encoding_queue' not in st.session_state:
    st.session_state.encoding_queue = queue.Queue()
if 'recognition_times' not in st.session_state:
    st.session_state.recognition_times = {}
if 'uploaded_photos' not in st.session_state:
    st.session_state.uploaded_photos = []

def upload_to_pinata(file_content, filename):
    try:
        url = "https://api.pinata.cloud/pinning/pinFileToIPFS"
        headers = {
            'pinata_api_key': PINATA_API_KEY,
            'pinata_secret_api_key': PINATA_API_SECRET
        }
        
        # Convert StreamlitUploadedFile to bytes
        if hasattr(file_content, 'getvalue'):
            file_data = file_content.getvalue()
        else:
            file_data = file_content
            
        files = {
            'file': (filename, file_data)
        }
        
        metadata = {
            'name': filename,
            'keyvalues': {
                'group': PINATA_GROUP_ID
            }
        }
        
        data = {
            'pinataMetadata': json.dumps(metadata),
            'pinataOptions': json.dumps({
                'cidVersion': 0,
                'customPinPolicy': {
                    'regions': [
                        {
                            'id': 'FRA1',
                            'desiredReplicationCount': 1
                        }
                    ]
                }
            })
        }
        
        response = requests.post(url, files=files, headers=headers, data=data)
        return response.json()
    except Exception as e:
        st.error(f"Error uploading to Pinata: {str(e)}")
        return None

def fetch_pinata_files():
    try:
        url = "https://api.pinata.cloud/data/pinList"
        headers = {
            'pinata_api_key': PINATA_API_KEY,
            'pinata_secret_api_key': PINATA_API_SECRET
        }
        params = {
            'metadata[keyvalues]': json.dumps({
                'group': {'value': PINATA_GROUP_ID, 'op': 'eq'}
            })
        }
        response = requests.get(url, headers=headers, params=params)
        return response.json()
    except Exception as e:
        st.error(f"Error fetching from Pinata: {str(e)}")
        return None

def process_ipfs_image(ipfs_hash):
    try:
        url = f"https://gateway.pinata.cloud/ipfs/{ipfs_hash}"
        response = requests.get(url)
        if response.status_code == 200:
            # Open image with PIL and convert to RGB
            image = Image.open(io.BytesIO(response.content)).convert('RGB')
            # Convert to numpy array
            img_array = np.array(image)
            return img_array, url
        return None, None
    except Exception as e:
        st.error(f"Error processing IPFS image: {str(e)}")
        return None, None

def encode_face(image):
    try:
        if image is None:
            return None
            
        # Ensure image is RGB
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] == 3 and not isinstance(image, np.ndarray):
            image = np.array(image)
            
        encodings = face_recognition.face_encodings(image)
        if encodings:
            return encodings[0]
        return None
    except Exception as e:
        st.error(f"Error encoding face: {str(e)}")
        return None

def check_new_pinata_files():
    try:
        pinata_data = fetch_pinata_files()
        if not pinata_data:
            return

        for item in pinata_data.get("rows", []):
            ipfs_hash = item["ipfs_pin_hash"]
            if ipfs_hash not in st.session_state.pinata_hashes:
                img, url = process_ipfs_image(ipfs_hash)
                if img is not None:
                    encoding = encode_face(img)
                    if encoding is not None:
                        name = item.get("metadata", {}).get("name", "Unknown").split('.')[0]
                        st.session_state.encoding_queue.put({
                            'encoding': encoding,
                            'name': name
                        })
                        st.session_state.pinata_hashes.add(ipfs_hash)
                        st.session_state.uploaded_photos.append({
                            'name': name,
                            'url': url
                        })

        st.session_state.last_pinata_check = datetime.now()
        
    except Exception as e:
        st.error(f"Error checking Pinata files: {str(e)}")

def process_encoding_queue():
    while not st.session_state.encoding_queue.empty():
        data = st.session_state.encoding_queue.get()
        st.session_state.encode_list_known.append(data['encoding'])
        st.session_state.class_names.append(data['name'])
        st.success(f"Added new face encoding for: {data['name']}")

def mark_attendance(name):
    try:
        with open('Attendance.csv', 'a') as f:
            now = datetime.now()
            date_string = now.strftime('%B %d, %Y')
            time_string = now.strftime('%H:%M:%S')
            f.write(f'{name},{date_string},{time_string}\n')
            
        # Store recognition time
        st.session_state.recognition_times[name] = now
    except Exception as e:
        st.error(f"Error marking attendance: {str(e)}")

def upload_thread(file_content, filename):
    response = upload_to_pinata(file_content, filename)
    if response and 'IpfsHash' in response:
        st.success(f"Image uploaded successfully for {filename}")
        check_new_pinata_files()
    else:
        st.error("Failed to upload image")

def main():
    st.title("Madhya Pradesh Police Attendance System")

    # Initialize system
    if not st.session_state.initialization_done:
        with st.spinner("Initializing system..."):
            check_new_pinata_files()
            st.session_state.initialization_done = True

    # Current Date and Time Display
    current_time = datetime.now()
    st.markdown(f"""
        <div class="current-time">
            Current Date: {current_time.strftime('%B %d, %Y')}<br>
            Current Time: {current_time.strftime('%H:%M:%S')}
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])

    with col2:
        st.subheader("Upload New Image")
        with st.container():
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            person_name = st.text_input("Person's Name")
            if st.button("Upload"):
                if uploaded_file is not None and person_name:
                    file_name = f"{person_name}.{uploaded_file.name.split('.')[-1]}"
                    # Start upload in a separate thread
                    thread = threading.Thread(
                        target=upload_thread,
                        args=(uploaded_file, file_name)
                    )
                    thread.start()

        st.subheader("System Status")
        status_placeholder = st.empty()
        
        if st.session_state.last_pinata_check:
            status_placeholder.info(
                f"Last IPFS check: {st.session_state.last_pinata_check.strftime('%H:%M:%S')}\n"
                f"Known faces: {len(st.session_state.encode_list_known)}"
            )

        # Recognition Times Display
        st.subheader("Recognition Times")
        for name, time in st.session_state.recognition_times.items():
            st.markdown(f"""
                <div class="recognition-time">
                    {name}: {time.strftime('%H:%M:%S')}
                </div>
            """, unsafe_allow_html=True)

        st.subheader("Attendance Log")
        if os.path.exists('Attendance.csv'):
            df = pd.read_csv('Attendance.csv', names=['Name', 'Date', 'Time'])
            st.dataframe(df)

        # Display uploaded photos
        st.subheader("Uploaded Photos")
        uploaded_photos_html = '<div class="uploaded-photos">'
        for photo in st.session_state.uploaded_photos:
            uploaded_photos_html += f'<img src="{photo["url"]}" alt="{photo["name"]}" class="uploaded-photo">'
        uploaded_photos_html += '</div>'
        st.markdown(uploaded_photos_html, unsafe_allow_html=True)

    with col1:
        st.subheader("Camera Feed")
        frame_placeholder = st.empty()

        cap = cv2.VideoCapture(0)
        
        try:
            while cap.isOpened():
                if (st.session_state.last_pinata_check is None or 
                    (datetime.now() - st.session_state.last_pinata_check).seconds >= 5):
                    check_new_pinata_files()
                
                process_encoding_queue()

                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to grab frame from camera")
                    continue

                # Ensure frame is in RGB format
                if len(frame.shape) == 2:  # Grayscale
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                elif frame.shape[2] == 3:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                elif frame.shape[2] == 4:  # RGBA
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                else:
                    st.error(f"Unsupported image format: {frame.shape}")
                    continue

                small_frame = cv2.resize(frame_rgb, (0, 0), fx=0.25, fy=0.25)

                face_locations = face_recognition.face_locations(small_frame)
                face_encodings = face_recognition.face_encodings(small_frame, face_locations)

                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    if len(st.session_state.encode_list_known) > 0:
                        matches = face_recognition.compare_faces(
                            st.session_state.encode_list_known, 
                            face_encoding,
                            tolerance=0.6
                        )
                        
                        if True in matches:
                            match_index = matches.index(True)
                            name = st.session_state.class_names[match_index]
                            
                            top *= 4
                            right *= 4
                            bottom *= 4
                            left *= 4

                            cv2.rectangle(frame_rgb, (left, top), (right, bottom), (0, 255, 0), 2)
                            cv2.rectangle(frame_rgb, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                            cv2.putText(frame_rgb, name, (left + 6, bottom - 6), 
                                      cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

                            if name not in st.session_state.marked_names:
                                mark_attendance(name)
                                st.session_state.marked_names.add(name)
                                st.success(f"{name} marked present!")

                frame_placeholder.image(frame_rgb, channels="RGB")

        except Exception as e:
            st.error(f"Camera error: {str(e)}")
        finally:
            cap.release()

    st.markdown("""
        <div style='position: fixed; bottom: 0; width: 100%; background-color: #0e1117; 
                    color: #fafafa; text-align: center; padding: 10px; font-size: 14px;'>
            © 2024 Madhya Pradesh Police | Powered by Aditya Bhattacharya
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

