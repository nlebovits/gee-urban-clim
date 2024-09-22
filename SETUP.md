# Bare bones setup instructions

1) Create `credentials/` subdirectory in the rooot dir. Add it to the `.gitignore` with `credentials/`.
2) Download a service key from GCS and put it in `credentials/`. 
3) Create .env file in `src/`. It should look like this:
```
GOOGLE_CLOUD_PROJECT=your-project-name
GOOGLE_CLOUD_BUCKET=your-bucket-name
GOOGLE_APPLICATION_CREDENTIALS=credentials/service-account-key.json
```

4) make sure that, for your chosen project, GEE API is enabled and the project is registered for GEE usage
5) run `earthengine authenciate`

May have to do this:
In order to use Earth Engine scripts, you must authenticate. From the home page open a local terminal and enter

earthengine authenticate

For the flood module, download the full [EMDAT disasters database](https://public.emdat.be/) as an Excel sheet and upload it to your GCP storage bucket. Update the `EMDAT_DATA_PATH` variable in the config file to match this path. 

6) To run, simply use `pipenv shell` and then either `python -m src.main [Country Name]` to run the full module or `python -m src.heat.heat [Country Name]` or `python -m src.flood.flood [Country Name]` for an individual module.

## Python Development
You can set up your local Python environment so you can develop outside of Docker. Build your local environment to match what is defined in the Dockerfile. Install the same python version as is in the Dockerfile, using `pyenv` to manage multiple distributions if needed. Use `pipenv` to create a virtual environment. Install the pip dependencies that are defined in the Pipfile into your virtual environment. Install the executables with `apt-get`. Now you can develop in Python in your terminal and IDE and run unit tests with `pytest`.