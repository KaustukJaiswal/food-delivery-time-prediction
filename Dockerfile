# set the base image
FROM python:3.12-slim

# install dependency
RUN apt-get update 

# set up the working directory
WORKDIR /app

# copy the requirements file
COPY requirements-dockers.txt ./

# install the packages
RUN pip install -r requirements-dockers.txt

# copy the app contents
COPY app.py ./
COPY ./models/preprocessor.joblib ./models/preprocessor.joblib
COPY ./notebooks/clean_data.py ./notebooks/clean_data.py
COPY ./run_information.json ./

# expose the port
EXPOSE 8000

# Run the file using command
CMD [ "python","./app.py" ]