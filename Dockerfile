# Use an official Python 3.11 runtime as a parent image
FROM python:3.11.4

# Set the working directory in the container
WORKDIR /usr/src/app

# Install Pipenv
RUN pip install pipenv

# Copy the Pipfile and Pipfile.lock from the src directory
COPY src/Pipfile src/Pipfile.lock ./

# Install the dependencies from Pipfile
RUN pipenv install --deploy --ignore-pipfile

# Use Pipenv to run the script
# Adjust the path to your main Python script if needed
CMD ["pipenv", "run", "python", "./flood.py"]