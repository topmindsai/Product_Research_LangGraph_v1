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

# Use main.py which reads PORT from environment via os.getenv()
CMD ["python", "main.py"]
