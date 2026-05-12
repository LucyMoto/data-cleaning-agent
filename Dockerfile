# Use an exact Python version for reproducible builds
FROM python:3.11.9-slim

# Stops Python writing .pyc cache files into the image (wasted space)
# Makes Python print logs immediately so they show up in `docker logs`
# Suppresses Streamlit's "enter your email" prompt on first run
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true

# All following instructions run relative to this directory inside the container
WORKDIR /app

# Copy dependency files first so Docker can cache the pip install layer —
# changes to app.py won't re-trigger a full reinstall
COPY pyproject.toml .
COPY data_cleaning_agent/ ./data_cleaning_agent/
RUN pip install --no-cache-dir .

# Copy the app after installing dependencies
COPY app.py .

# Create a non-root user — running as root is a security risk
# --create-home creates /home/appuser/ which Streamlit needs for its config cache
RUN useradd --create-home appuser
USER appuser

# Document that the container listens on this port (use -p 8501:8501 when running)
EXPOSE 8501

# Verify the app is responding, not just that the process is alive
# Uses Python's built-in HTTP library since curl isn't installed in slim images
# --start-period gives Streamlit 30s to boot before failed checks count against it
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# --server.address=0.0.0.0 makes Streamlit reachable from outside the container
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
