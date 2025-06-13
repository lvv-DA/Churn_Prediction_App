# Use a slim Python base image for smaller size
FROM python:3.9-slim-buster

# Set the working directory inside the container
WORKDIR /app

# --- Dependencies Installation ---
# Copy only requirements.txt first to leverage Docker cache
# This helps if your requirements change less often than your code
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir reduces image size by not storing pip's cache
RUN pip install --no-cache-dir -r requirements.txt

# --- Copy Streamlit Configuration (config.toml, secrets.toml) ---
# Create the .streamlit directory if it doesn't exist
RUN mkdir -p .streamlit/
# Copy the config.toml and secrets.toml files into the .streamlit directory
COPY .streamlit/config.toml .streamlit/
COPY .streamlit/secrets.toml .streamlit/

# --- Copy Application Code and Data ---
# Copy your main Streamlit application files (e.g., your_app.py, other .py files)
# and any subdirectories containing your data, images, etc.
# The '.' means copy everything from the current host directory (where Dockerfile is)
# to the WORKDIR /app in the container.
# Make sure your app data and source code are relative to the Dockerfile.
COPY . .

# --- Container Configuration ---
# Expose the port Streamlit runs on. Default is 8501.
EXPOSE 8501

# --- Command to run your Streamlit application ---
# `streamlit run` is the command to start a Streamlit app.
# `your_main_app_script.py` should be the name of your primary Streamlit script.
# `--server.port=8501` ensures it listens on the exposed port.
# `--server.address=0.0.0.0` makes the app accessible from outside the container.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]