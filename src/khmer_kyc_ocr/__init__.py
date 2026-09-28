import os

# Refuse decompression bombs: a 5 MB PNG can otherwise expand to gigabytes of pixels.
# OpenCV reads this on first decode, so it must be set before any imdecode call.
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(40_000_000))
