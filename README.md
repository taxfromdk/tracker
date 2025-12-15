# Video Object Tracker

A real-time multi-object tracker with correlation-based template matching, built with OpenCV and Pygame.

## Features

- **Multi-object tracking**: Click to add multiple trackers simultaneously
- **Correlation-based tracking**: Uses normalized cross-correlation for robust tracking
- **Multithreaded processing**: Parallel tracking updates for better performance
- **Interactive UI**: Real-time visualization with adjustable playback speed
- **Auto-removal**: Trackers automatically removed when correlation drops below threshold

## Visual Indicators

- **Green rectangle**: Original patch location (65x65 pixels)
- **Blue rectangle**: Search area (201x201 pixels)
- **Red dot**: Current tracked position
- **Yellow text**: Correlation score (0.000 to 1.000)

## Installation

```bash
# Clone the repository
git clone https://github.com/taxfromdk/tracker.git
cd tracker

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
make install

# Download sample video
make download
```

## Usage

### Quick Start

```bash
# Run the tracker
make test

# Or directly with Python
python main.py
```

### Controls

- **Mouse Click**: Add a new tracker at the clicked position
- **Spacebar**: Play/pause video
- **+/=**: Increase playback speed (2x, 4x, 8x, 16x)
- **-**: Decrease playback speed (0.5x, 0.25x, 0.125x)
- **Left/Right Arrows**: Step frame by frame
- **ESC/Q**: Exit application
- **Mouse Drag on Slider**: Seek to specific frame

## Technical Details

### Architecture

- **Tracker Class**: Thread-safe tracker with configurable patch and search sizes
- **Template Matching**: OpenCV's `cv2.matchTemplate` with `TM_CCOEFF_NORMED`
- **Threading**: `ThreadPoolExecutor` with 8 worker threads for parallel processing
- **Rendering**: Pygame/SDL2 for hardware-accelerated display

### Parameters

- **Patch Size**: 32 pixels (65x65 total area)
- **Search Size**: 100 pixels (201x201 total area)
- **Correlation Threshold**: 0.6 (trackers removed below this value)
- **Max Workers**: 8 parallel threads

## Project Structure

```
tracker/
├── main.py           # Main application with Tracker class
├── requirements.txt  # Python dependencies
├── Makefile         # Build and run commands
├── README.md        # This file
└── .gitignore      # Git ignore rules
```

## Requirements

- Python 3.7+
- OpenCV 4.x
- Pygame 2.x
- NumPy

## Makefile Commands

- `make` or `make all`: Install dependencies and download video
- `make install`: Install Python packages
- `make download`: Download sample video
- `make test`: Run the tracker application
- `make clean`: Clean temporary files

## Performance Notes

- Multithreading provides significant speedup with multiple trackers
- Correlation computation is the main bottleneck
- Search area size directly impacts performance
- Consider GPU acceleration for many trackers (>10)

## Future Improvements

- [ ] GPU acceleration with CUDA
- [ ] Advanced tracking algorithms (KCF, CSRT)
- [ ] Track prediction and smoothing
- [ ] Export tracking data to CSV/JSON
- [ ] Batch processing for multiple videos
- [ ] Configurable UI themes

## License

MIT License - feel free to use and modify as needed.

## Author

Created by Jesper ([@taxfromdk](https://github.com/taxfromdk))