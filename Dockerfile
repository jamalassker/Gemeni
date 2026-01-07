FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .

RUN python -m venv --copies /opt/venv \
    && . /opt/venv/bin/activate \
    && pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .
CMD ["python", "bot.py"]
