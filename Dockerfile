FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libglib2.0-0 \
    libffi8 \
    fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*


WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV PORT=8080 PYTHONUNBUFFERED=1 OUT_DIR=/tmp/out
RUN mkdir -p $OUT_DIR
EXPOSE 8080

# If your app file is App.py, change app:app -> App:app
CMD ["bash","-lc","gunicorn -w 2 -b 0.0.0.0:$PORT app:app --timeout 300"]
