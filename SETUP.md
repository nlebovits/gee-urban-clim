## Setup Instructions

This codebase runs one or more scripts to generate climate hazards data for a given country based on random forest models trained in Google Earth Engine using historic data. Various models can be created by updating the input settings in the `config.py`, and the script can be run for each climate hazard separately (currently heat and flood are available), or all together. 

I am in the process of containerizing the workflow with Docker to make it maximally portable.

### Installation and Setup

This project requires a Google Cloud storage account and acess to the Google Earth Engine API. To be run outside of Docker, it requires `pipenv` and potentially `pyenv`. 

Once those are installed, the steps for setup are as follows:

1) Create `credentials/` subdirectory in the rooot dir. Add it to the `.gitignore` with `credentials/`.
2) Download a service key from GCS and put it in `credentials/`. 
3) Create .env file in `src/`. It should look like this:
```
GOOGLE_CLOUD_PROJECT=your-project-name
GOOGLE_CLOUD_BUCKET=your-bucket-name
GOOGLE_APPLICATION_CREDENTIALS=credentials/service-account-key.json
```

4) make sure that, for your chosen project, GEE API is enabled and the project is registered for GEE usage
5) In the root directory, run `earthengine authenciate`. You should only have to do this once.
6) In order to make the flood script run properly, download the full [EMDAT disasters database](https://public.emdat.be/) as an Excel sheet and upload it to your GCP storage bucket. Update the `EMDAT_DATA_PATH` variable in the config file to match this path. 

### Python Development
You can set up your local Python environment so you can develop outside of Docker. Build your local environment to match what is defined in the Dockerfile. Install the same python version as is in the Dockerfile, using `pyenv` to manage multiple distributions if needed. Use `pipenv` to create a virtual environment. Install the pip dependencies that are defined in the Pipfile into your virtual environment. Install the executables with `apt-get`. Now you can develop in Python in your terminal and IDE and run unit tests with `pytest`.