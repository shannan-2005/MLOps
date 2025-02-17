# Use an official TensorFlow image as base
FROM tensorflow/tensorflow:latest

# Set the working directory inside the container
WORKDIR /app

# Copy the local files into the container
COPY app.py train.csv test.csv .  

# Install required packages
RUN pip install pandas matplotlib numpy

# Run the script when the container starts
CMD ["python", "app.py"]
