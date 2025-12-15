.PHONY: all clean test install download

VIDEO_FILE = VID20251205101021.mp4
VIDEO_URL = https://link.storjshare.io/raw/jvq7udetowr4oj2h6kp3kxjll3ya/cloudmemory/VID20251205101021.mp4

all: install download

install:
	@echo "Installing dependencies..."
	@pip install -r requirements.txt

download:
	@if [ ! -f $(VIDEO_FILE) ]; then \
		echo "Downloading video file..."; \
		wget $(VIDEO_URL); \
	else \
		echo "Video file already exists"; \
	fi

test: download
	@echo "Running tracker..."
	@python main.py

clean:
	@echo "Cleaning up..."
	#@rm -f $(VIDEO_FILE)
	@rm -rf __pycache__
	@rm -f *.pyc
	@echo "Cleanup complete"