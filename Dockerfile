# Use an official Python 3.11 runtime as a parent image
FROM python:3.11.4

# Set the working directory in the container
WORKDIR /usr/src/app

# Install Pipenv
RUN pip install pipenv

# Set environment variables to increase pip timeout
ENV PIP_DEFAULT_TIMEOUT=100

# Copy the Pipfile and Pipfile.lock from the root directory
COPY Pipfile Pipfile.lock ./

# Install the dependencies from Pipfile with verbose output
# Use a more reliable PyPI mirror
RUN pipenv install --deploy --ignore-pipfile --verbose --pypi-mirror https://pypi.python.org/simple

# Copy the rest of the application code
COPY src/ ./

# Copy the credentials directory
COPY credentials/ ./credentials/

# Use Pipenv to run the script
CMD ["pipenv", "run", "python", "./main.py"]
