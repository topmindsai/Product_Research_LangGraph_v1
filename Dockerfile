FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install the package itself
RUN pip install --no-cache-dir -e .

# Use shell form to expand $PORT variable
CMD uvicorn product_research.api.routes:app --host 0.0.0.0 --port ${PORT:-8000}
