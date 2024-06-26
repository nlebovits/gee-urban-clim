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