import torch
import numpy as np
import cv2
import datetime
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import time
import pandas as pd

@st.cache_resource
def load_model():
    with st.spinner('Processing...'):
        time.sleep(3)
        #Load fine-tuned YOLO detection model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp5/weights/last.pt', force_reload=True)
    return model

def store_log(log,date):
    '''This function stores the activity logs to your local storage.'''
    df = pd.DataFrame(log,columns=['Action','Time'])
    df_title = log[0][1]
    try:
        df.to_csv(f'log_storage/{date}-{df_title}.csv')
        st.write("Successfully stored the log!")
        time.sleep(10)
    except OSError:
        st.write("The directory does not exist. Failed to store the log.")
        time.sleep(10)
    st.session_state.log = None

def reset_page():
    with st.spinner("Rerunning the app..."):
        time.sleep(3)
    st.session_state.clear()
    st.rerun()

def main():
    cap = cv2.VideoCapture(0)
    model = load_model()
    log = []

    #Basic structure of the webpage
    st.markdown("""
    <h1 style="font-size: 80px; color: #E9FBFF; text-align: center; font-family:
    times new roman">Archangel</h1>""", unsafe_allow_html=True)
    frame_placeholder = st.empty()
    log_placeholder = st.empty()
    col1,col2 = st.columns(2)

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
        date = current_time.strftime("%m.%d.%Y")
        hour_minute = current_time.strftime("%H:%M:%S")
        #Passing date infromation to session_state
        st.session_state.date = date

        #Taking logs of events
        class_names = results.names #dictionary
        detected_class_names = [class_names[label] for label in detected_labels]
        if detected_class_names != []:
            for word in detected_class_names:
                log.append([word,hour_minute])
                #Storing log as session state
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
    if 'log' not in st.session_state or st.session_state.log == None:
        st.write("No specific action detected...")
    else:
        log_array = np.array(st.session_state.log)

        print(log_array)

        fig = px.scatter(x=log_array[:,1],y=log_array[:,0],
                        labels={"x":"Time","y":"Action"},title="Activity Log")
        fig.update_layout(xaxis = go.layout.XAxis(tickangle = 45))
        log_placeholder.plotly_chart(figure_or_data=fig,use_container_width=False,
                                    sharing="streamlit",theme="streamlit")

        #Show button for storing the log
        with col1:
            st.button("Store Log",on_click=store_log(log=log_array,date=st.session_state.date))

    #Show refresh button
    with col2:
        st.button("Reset",on_click=reset_page)

if __name__ == '__main__':
    main()
