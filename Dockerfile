FROM: chooses the base image (a lightweight Python in this case).
ENV: sets flags that reduce clutter and keep logs visible.
WORKDIR: defines the working directory inside the image.
COPY requirements.txt + RUN pip install: installs dependencies in a cacheable layer.
COPY . .: copies your project files.
CMD: tells Docker what to run when the container starts.