import torch
import torchvision
import numpy as np
import cv2
from matplotlib import pyplot as plt
import uuid
import os
import datetime
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import plotly.express as px

def capture_camera():
    #Load fine-tuned YOLO detection model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp5/weights/last.pt', force_reload=True)

    log = {'Dogs_run':[],
       'Dogs_sleep':[],
       'Dogs_trash':[],
       'Dogs_poop':[],
       'Dogs_eat':[]}
    #Open webcam to capture images
    cap = cv2.VideoCapture(0)
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
                log[word].append(hour_minute)

        cv2.imshow('YOLO', np.squeeze(results.render()))

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return log

def show_log(log):
    combined_data = np.array([[event, time] for event, times in log.items() for time in times])
    fig = px.scatter(x=combined_data[:,1],y=combined_data[:,0])
    fig.show()

#webrtc_streamer(key='sample')
# col1, col2 = st.columns([2.8,0.3,3])
# col1.write(" ")
# with col1:
#     st.subheader("Video Input")
#     #Camera Image
#     with st.form("image_input"):
#         captured_image =
