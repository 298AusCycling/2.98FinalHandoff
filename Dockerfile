# Use a minimal Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app code, including the .streamlit config
COPY . .


# Expose the port for documentation (Cloud Run sets PORT env var)
EXPOSE 8080

# Run Streamlit with Cloud Run's dynamic port
ENTRYPOINT ["sh", "-c", "streamlit run app.py --server.port=$PORT --server.enableCORS=false"]
