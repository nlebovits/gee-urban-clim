# Bare bones setup instructions

1) Create `credentials/` subdirectory in the rooot dir. Add it to the `.gitignore` with `credentials/`.
2) Download a service key from GCS and put it in `credentials/`. 
3) Create .env file in `src/`. It should look like this:
```
GOOGLE_CLOUD_PROJECT=your-project-name
GOOGLE_CLOUD_BUCKET=your-bucket-name
GOOGLE_APPLICATION_CREDENTIALS=credentials/service-account-key.json
```