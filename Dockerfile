# Use the official Python base image
FROM python:3.9-slim

#update pip
RUN pip install --upgrade pip

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Copy the packages file
COPY packages.txt .

# Install pachakges
#RUN apt update && apt install -y libsm6 libxext6 libfontconfig1 libxrender1 libgl1-mesa-glx libgtk2.0-dev

RUN apt update && apt install -y python3-opencv

# Install the app dependencies
RUN pip install --no-cache-dir -r requirements.txt


EXPOSE 8501

# Copy the app code to the container
COPY . .

# Set the command to run the app
CMD ["streamlit", "run", "app.py", "--server.port", "8501"]