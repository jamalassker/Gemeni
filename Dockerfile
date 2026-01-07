FROM python:3.12-slim-bookworm

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv --copies /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Make virtual environment available
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY . .

# Default command (replace with your app entrypoint)
CMD ["python", "bot.py"]
