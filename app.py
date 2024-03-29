import torch
import torchvision
import numpy as np
import cv2
from matplotlib import pyplot as plt
import uuid
import os
import queue
import datetime
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import plotly.express as px

def capture_camera():
    #Load fine-tuned YOLO detection model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp5/weights/last.pt', force_reload=True)

    log = []
    #Open webcam to capture images
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    while cap.isOpened():
        ret, frame = cap.read()
        # Make detections using the model
        results = model(frame)

        #Extracting detected labels and detected time
        predictions = results.pred[0]
        detected_labels = predictions[:,-1].cpu().numpy()
        current_time = datetime.datetime.now()
        hour_minute = current_time.strftime("%H:%M:%S")
        #Taking logs of events
        class_names = results.names
        detected_class_names = [class_names[label] for label in detected_labels]
        if detected_class_names != []:
            for word in detected_class_names:
                log.append([word,hour_minute])

        cv2.imshow('YOLO', np.squeeze(results.render()))

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    #Show the log of your pet.
    #combined_data = np.array([[event, time] for event, times in log.items() for time in times])
    log = np.array(log)
    return log

def show_figure(log):
    fig = px.scatter(x=log[:,1],y=log[:,0])
    fig.show()

#webrtc_streamer(key='sample')
# col1, col2 = st.columns([2.8,0.3,3])
# col1.write(" ")
# with col1:
#     st.subheader("Video Input")
#     #Camera Image
#     with st.form("image_input"):
#         captured_image =
result_queue: "queue.Queue[List[Detection]]" = queue.Queue() #type:ignore
second_queue: "queue.Queue[List[Detection]]" = queue.Queue() #type:ignore

camera_load_state = st.text('Preparing camera...')
ctx = webrtc_streamer(key='example',
                      video_frame_callback=capture_camera,
                      rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                      media_stream_constraints={"video": True, "audio": False},
                      async_processing=True)

if ctx.state.playing:
        frame_count = 0
        frame_counter_placeholder = st.empty()
        frame_counter_placeholder.text("Recording... 0%")

        result = ""
        prediction_placeholder = st.empty()

        while True:
            frame_count = (second_queue.get() + 1) * 100 / 20
            frame_counter_placeholder.text(f"📽️ Recording... {frame_count: .0f}%")

            if frame_count == 100:
                frame_counter_placeholder.text("🛠️ AI at work... 🦾")
                result += result_queue.get() + " -> "
                prediction_placeholder.markdown(f"<h1>{result}</h1>", unsafe_allow_html=True)



# if st.checkbox('Show the log'):
#     st.subheader('Here is the log of today.')
#     figure = show_figure(camera_output)
#     st.write(figure)
