import torch
import numpy as np
import cv2
import datetime
import streamlit as st
import plotly.express as px

@st.cache_resource
def load_model():
    #Load fine-tuned YOLO detection model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp5/weights/last.pt', force_reload=True)
    return model

cap = cv2.VideoCapture(0)
model = load_model()
log = []

#Basic structure of the webpage
st.title("Archangel")
frame_placeholder = st.empty()
log_placeholder = st.empty()

#Add "Stop" button and store its state in a variable
stop_button_pressed = st.button("Stop")

#Livestream webcam using OpenCV
while cap.isOpened() and not stop_button_pressed:
    ret, frame = cap.read()
    if not ret:
        st.write("The video capture has ended.")
        break
    #Applying finetuned YOLO model to webcam
    results = model(frame)

    #Extracting detected labels and detected time
    predictions = results.pred[0]
    detected_labels = predictions[:,-1].cpu().numpy() #list of numbers
    current_time = datetime.datetime.now()
    hour_minute = current_time.strftime("%H:%M:%S")
    #Taking logs of events
    class_names = results.names #dictionary
    detected_class_names = [class_names[label] for label in detected_labels]
    if detected_class_names != []:
        for word in detected_class_names:
            log.append([word,hour_minute])
            #Storing log as session state
            if 'log' not in st.session_state:
                st.session_state.log = log
    #Displaying the webcam on the webpage
    results_array = np.squeeze(results.render())
    stream = cv2.cvtColor(results_array,cv2.COLOR_BGR2RGB)
    frame_placeholder.image(stream,channels="RGB")

    # Break the loop if the 'q' key is pressed or the user clicks the "Stop" button
    if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
        break

cap.release()
cv2.destroyAllWindows()

#Showing the log as a graph after the webcam is closed
log_array = np.array(st.session_state.log)
fig = px.scatter(x=log_array[:,1],y=log_array[:,0],
                 labels={"x":"Time","y":"Action"},title="Activity Log")
log_placeholder.plotly_chart(figure_or_data=fig,use_container_width=False,
                             sharing="streamlit",theme="streamlit")

#if __name__ = '__main__':
