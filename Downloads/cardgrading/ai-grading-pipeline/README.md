# AI Grading Pipeline

[![CI](https://github.com/kevinmastascusa/ai-grading-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/kevinmastascusa/ai-grading-pipeline/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python Version](https://img.shields.io/badge/python-3.8-blue.svg)
[![Docker](https://img.shields.io/badge/Docker-blue?logo=docker&logoColor=white)](Dockerfile)
[![Conda](https://img.shields.io/badge/Conda-env-green.svg)](environment.yml)

The AI Grading Pipeline is an end-to-end, scalable, and reproducible system designed to revolutionize trading card grading using advanced image processing and machine learning. This project leverages a modern MLOps stack—including Python, Conda, Docker, Flask, and potentially PySpark—to automate the grading process, streamline the ML lifecycle, and deliver consistent, data-driven insights into card quality. With built-in CI/CD using GitHub Actions and readiness for Kubernetes deployment, this pipeline is engineered for robustness and continuous improvement.

## Table of Contents

- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Development Environment Setup](#development-environment-setup)
- [Building and Running with Docker](#building-and-running-with-docker)
- [CI/CD with GitHub Actions](#cicd-with-github-actions)
- [MLOps Considerations](#mlops-considerations)
- [Usage](#usage)
- [Running the Flask API (Locally without Docker)](#running-the-flask-api-locally-without-docker)
- [Contributing](#contributing)
- [License](#license)

## Key Features

- ✨ **Automated Grading:** Leverages computer vision and machine learning to predict card grades.
- 📈 **Scalable Architecture:** Designed with Docker and Kubernetes in mind for handling varying loads.
- 🔄 **Reproducible Environments:** Conda and Docker ensure consistent setups across development and production.
- 🤖 **MLOps Focused:** Incorporates practices for versioning, experiment tracking, and model deployment.
- 🚀 **CI/CD Pipeline:** Automated testing and deployment workflows using GitHub Actions.
- 📊 **Data-Driven Insights:** Potential for rich analytics from grading data.
- modular **Modular Design:** Clearly separated components for data ingestion, feature extraction, model training, and API serving.

## System Architecture

(Placeholder for a system architecture diagram. This could illustrate the flow from data ingestion through model training to API deployment and interaction.)

Example components:

`Data Sources -> Data Ingestion -> Preprocessing/Feature Extraction -> Model Training -> Model Registry -> Prediction API -> User Interface/Client`

## Technologies Used

- **Programming Language:** Python 3.8
- **Environment Management:** Conda
- **Containerization:** Docker
- **Web Framework (API):** Flask
- **CI/CD:** GitHub Actions
- **Machine Learning Libraries:** NLTK, spaCy, Scikit-learn, TensorFlow (as per `environment.yml`)
- **Data Handling:** Pandas, NumPy, SQLAlchemy
- **Big Data Processing (Optional):** PySpark
- **Deployment (Target):** Kubernetes
- **Development Tools:** pytest, Flake8
- **Data/Model Versioning (Optional):** DVC

## Project Structure

```text
ai-grading-pipeline/
├── .github/
│   └── workflows/
│       ├── ci.yml            # GitHub Actions CI workflow
│       └── cd.yml            # GitHub Actions CD workflow (template)
├── data/
│   ├── raw/                # Raw data (e.g., images, initial CSVs)
│   └── processed/          # Processed data for training/analysis
├── models/                 # (Optional) For storing trained model artifacts (consider DVC)
├── notebooks/              # Jupyter notebooks for EDA, experimentation
│   └── cardgod.ipynb
├── src/                    # Core source code
│   ├── api/                # Flask API for serving predictions
│   │   └── app.py
│   ├── data_ingestion/     # Data loading and database interaction
│   │   ├── database_setup.py
│   │   └── ingest_data.py
│   ├── feature_extraction/ # Feature engineering and processing
│   │   ├── extract_features.py
│   │   └── spark_pipeline.py # Example for Spark-based features
│   ├── model_training/     # ML model training and evaluation scripts
│   │   └── train_model.py
│   ├── visualization/      # (Optional) Scripts for generating visualizations
│   │   └── visualize.py
│   └── utils/              # Utility functions and helpers
│       └── helpers.py
├── .dockerignore           # Specifies files to exclude from Docker image
├── .gitignore              # Specifies intentionally untracked files for Git
├── Dockerfile              # Defines the Docker image build process
├── environment.yml         # Conda environment specification
├── main.py                 # Main application script (if applicable, e.g., for CLI)
├── README.md               # This file!
├── requirements.txt        # pip dependencies (consider consolidating into environment.yml)
├── deployment.yaml         # Example Kubernetes deployment manifest
├── service.yaml            # Example Kubernetes service manifest
└── LICENSE                 # Project license file (e.g., MIT License)
```

## Prerequisites

Before you begin, ensure you have the following installed:

- [Git](https://git-scm.com/)
- [Conda (Miniconda or Anaconda)](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or Docker Engine on Linux)

## Development Environment Setup

This project uses Conda for managing Python environments and dependencies.

1. **Clone the repository:**

   ```bash
   git clone https://github.com/kevinmastascusa/ai-grading-pipeline.git
   cd ai-grading-pipeline
   ```

2. **Create and activate the Conda environment:**

   The `environment.yml` file specifies all necessary dependencies.

   ```bash
   conda env create -f environment.yml
   conda activate ai-grading-pipeline
   ```

   *Note: NLTK (`punkt`) and spaCy (`en_core_web_sm`) data are downloaded during the Docker build. For local development without Docker, you might need to download them manually after activating the environment:*

   ```bash
   python -m nltk.downloader punkt
   python -m spacy download en_core_web_sm
   ```

## Building and Running with Docker

The `Dockerfile` encapsulates the application and its environment.

1. **Build the Docker image:**

   From the project root directory:

   ```bash
   docker build -t ai-grading-pipeline .
   ```

2. **Run the Docker container:**

   This example runs the Flask API, exposing port 5000.

   ```bash
   docker run -p 5000:5000 ai-grading-pipeline
   ```

   The API should then be accessible at `http://localhost:5000`.

## CI/CD with GitHub Actions

Automated workflows ensure code quality and streamline deployments.

- **Continuous Integration (`.github/workflows/ci.yml`):**
  - Triggered on every `push` and `pull_request` to main branches.
  - Sets up the Conda environment.
  - Performs linting (e.g., Flake8) and runs unit tests (e.g., pytest).
  - (Optionally) Builds the Docker image to verify build success.

- **Continuous Deployment (`.github/workflows/cd.yml`):**
  - A template workflow, typically triggered on merges/pushes to a `main` or `release` branch.
  - Builds the Docker image.
  - Pushes the image to a container registry (e.g., Docker Hub, GitHub Container Registry).
  - Includes placeholder steps for deploying to an orchestration platform like Kubernetes.
  - **Important:** Requires configuring secrets (e.g., `DOCKER_USERNAME`, `DOCKER_PASSWORD`, `KUBE_CONFIG_DATA`) in your GitHub repository settings (`Settings > Secrets and variables > Actions`).

## MLOps Considerations

Embracing MLOps for a robust and maintainable machine learning system.

- **Experiment Tracking:** Utilize tools like [MLflow](https://mlflow.org/) to log experiments, parameters, metrics, and artifacts from your model training runs.
- **Data and Model Versioning:** For large datasets and model files that are not suitable for Git, [DVC (Data Version Control)](https://dvc.org/) is recommended. `dvc` is included as an optional dependency in `environment.yml`.
- **Model Registry:** Store and manage versions of your trained models (MLflow can provide this).
- **Monitoring:** Implement comprehensive logging in your API and set up monitoring for model performance, data drift, and system health in production.

## Usage

Detailed instructions on using different components of the pipeline.

- **Data Ingestion (`src/data_ingestion/`):**
  - Run `ingest_data.py` (potentially with arguments) to load data from CSVs or other sources into your database.
  - Ensure `database_setup.py` has been configured and run if it creates schema.
- **Feature Extraction (`src/feature_extraction/`):**
  - Execute scripts like `extract_features.py` to process raw data and generate features for training.
- **Model Training (`src/model_training/train_model.py`):**
  - Run `train_model.py` to train your AI models using the processed features. This script should handle data loading, model selection, training, evaluation, and saving the trained model.
- **API (`src/api/app.py`):**
  - The Flask application serves predictions. Once a model is trained and available, the API loads it and exposes prediction endpoints.
- **Notebooks (`notebooks/`):**
  - Use Jupyter notebooks for exploratory data analysis (EDA), initial model prototyping, and result visualization.

## Running the Flask API (Locally without Docker)

1. Ensure your Conda environment (`ai-grading-pipeline`) is activated.
2. Navigate to the project root.
3. Run the application:

   ```bash
   python src/api/app.py
   ```

   Access the API at `http://localhost:5000` (or as configured).

## Contributing

Contributions are highly welcome! If you'd like to contribute, please follow these guidelines:

1. Fork the repository.
2. Create a new branch for your feature or bug fix (`git checkout -b feature/your-feature-name` or `bugfix/issue-number`).
3. Make your changes and commit them with clear, descriptive messages.
4. Ensure your code adheres to the project's style guidelines (run linters).
5. Write unit tests for new functionality and ensure all tests pass.
6. Push your changes to your fork.
7. Open a pull request to the `main` branch of this repository.

Please ensure your contributions pass all CI checks.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.