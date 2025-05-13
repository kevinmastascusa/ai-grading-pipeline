# Use an official Miniconda3 image as a parent image
FROM continuumio/miniconda3:latest

# Set the working directory in the container
WORKDIR /app

# Copy the Conda environment file
COPY environment.yml .

# Create the Conda environment from the environment.yml file
# This also installs all specified dependencies including Python
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "ai-grading-pipeline", "/bin/bash", "-c"]

# Download NLTK data
RUN python -m nltk.downloader punkt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy the rest of the application code into the container
# This includes your main.py, src directory, etc.
COPY . .

# Expose the port the Flask app runs on
EXPOSE 5000

# Define the command to run the Flask application within the Conda environment
# This assumes your Flask app is in src/api/app.py
CMD ["conda", "run", "-n", "ai-grading-pipeline", "python", "src/api/app.py"]