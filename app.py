FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# System deps: cairo/pango (WeasyPrint), ffmpeg, tesseract for OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libglib2.0-0 \
    libffi8 \
    fonts-dejavu-core \
    ffmpeg \
    tesseract-ocr \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Runtime env
ENV OUT_DIR=/tmp/out \
    PYTHONUNBUFFERED=1 \
    PORT=8080
RUN mkdir -p $OUT_DIR

EXPOSE 8080

CMD ["bash","-lc","gunicorn -w 2 -b 0.0.0.0:$PORT app:app --timeout 300"]
