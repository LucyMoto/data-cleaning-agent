# Makefile for Streamlit Data Cleaning Agent

IMAGE_NAME=data-cleaning-agent
CONTAINER_NAME=data-cleaning-agent-container

.PHONY: help run test lint format install docker stop logs

help:
	@echo "Available commands:"
	@echo "  make install  - Install dependencies"
	@echo "  make run      - Start the Streamlit app locally"
	@echo "  make test     - Run tests"
	@echo "  make lint     - Check code for style issues"
	@echo "  make format   - Auto-fix formatting"
	@echo "  make docker   - Build and run the Docker container"
	@echo "  make stop     - Stop the Docker container"
	@echo "  make logs     - View Docker container logs"

install:
	@echo "Installing dependencies..."
	@poetry install

run:
	@echo "Starting Streamlit app..."
	@poetry run streamlit run app.py

test:
	@echo "Running tests..."
	@poetry run pytest -q

lint:
	@echo "Linting code..."
	@poetry run ruff check .

format:
	@echo "Formatting code..."
	@poetry run ruff format .

docker:
	@echo "Building Docker image..."
	@docker build -t $(IMAGE_NAME) .
	@echo "Stopping and removing existing container..."
	-@docker stop $(CONTAINER_NAME) && docker rm $(CONTAINER_NAME)
	@echo "Running Docker container..."
	@docker run --name $(CONTAINER_NAME) -d -p 8501:8501 $(IMAGE_NAME)

stop:
	@echo "Stopping Docker container..."
	-@docker stop $(CONTAINER_NAME) && docker rm $(CONTAINER_NAME)

logs:
	@docker logs $(CONTAINER_NAME)
