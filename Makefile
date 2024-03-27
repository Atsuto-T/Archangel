#Start webcam. Press 'q' for quit. Automatically shows the log.
start:
	python -c "from app import capture_camera; capture_camera()"

install_requirements:
	@pip install -r requirements.txt

#HEROKU COMMANDS
streamlit_main_ui:
	-@streamlit run app.py --server.port 8503
