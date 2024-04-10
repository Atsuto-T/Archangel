.DEFAULT_GOAL := default

#Start webcam. Press 'q' for quit. Automatically shows the log.
start:
	python -c "from app import capture_camera; capture_camera()"

install_requirements:
	@pip install -r requirements.txt

#Run Streamlit Cloud
streamlit_main_ui:
	-@streamlit run app.py --server.port 8503

#Package Actions
save_model:
	python -c "from archangel.registry import save_model; save_model()"
