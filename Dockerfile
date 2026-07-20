# Use the official lightweight Python image as the base.
FROM python:3.11-slim

# Set the container working directory.
WORKDIR /app

# Copy the dependency file.
COPY requirements.txt .

# Install dependencies. --no-cache-dir reduces the image size.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project into the working directory.
COPY . .

# Install gosu, a lightweight su/sudo alternative used to switch users.
# Clean package metadata in the same layer to reduce image size.
RUN apt-get update && \
    apt-get install -y gosu && \
    rm -rf /var/lib/apt/lists/*

# Install the container entrypoint.
COPY entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["entrypoint.sh"]

# Create a non-root user for the application.
RUN useradd -m -u 1001 appuser

# Document the port exposed by the container.
# This should match the internal CODEBUDDY_PORT configuration.
EXPOSE 8001

# Start the production ASGI server with Hypercorn.
CMD ["hypercorn", "web:app", "--bind", "0.0.0.0:8001"]
