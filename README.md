# TRIBE v2 Inference App

Minimal FastAPI + vanilla-JS web app that runs Meta's [TRIBE v2](https://github.com/facebookresearch/tribev2) brain encoding model on an uploaded video and returns predicted neural activations.

---

## Local run

```bash
# 1. Install system deps (macOS/Linux)
brew install ffmpeg          # macOS
# sudo apt install ffmpeg libsndfile1   # Ubuntu

# 2. Install Python deps
cd tribev2-app
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/tribev2.git

# 3. Start server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Open http://localhost:8000
```

---

## Docker

```bash
# Build
docker build -t tribev2-app .

# Run (model cache persisted via volume)
docker run -p 8000:8000 -v $(pwd)/cache:/app/cache tribev2-app
```

---

## Azure deployment (ACR + ACI)

### 1. Push image to Azure Container Registry

```bash
ACR=<your-acr-name>          # e.g. myregistry
RG=<your-resource-group>
IMAGE=tribev2-app

az acr login --name $ACR
docker tag tribev2-app $ACR.azurecr.io/$IMAGE:latest
docker push $ACR.azurecr.io/$IMAGE:latest
```

### 2. Deploy to Azure Container Instances

```bash
az container create \
  --resource-group $RG \
  --name tribev2-app \
  --image $ACR.azurecr.io/$IMAGE:latest \
  --registry-login-server $ACR.azurecr.io \
  --registry-username $(az acr credential show -n $ACR --query username -o tsv) \
  --registry-password $(az acr credential show -n $ACR --query "passwords[0].value" -o tsv) \
  --cpu 4 \
  --memory 8 \
  --ports 8000 \
  --dns-name-label tribev2-app \
  --environment-variables PYTHONUNBUFFERED=1 \
  --restart-policy OnFailure

# Get public URL
az container show --resource-group $RG --name tribev2-app \
  --query ipAddress.fqdn -o tsv
```

Health check endpoint for Azure: `GET /health` → `{"status":"ok"}`

---

## API

| Method | Path | Description |
|--------|------|-------------|
| POST | `/predict` | Upload video → run inference → return shape + preview |
| GET | `/download/brain_predictions.npy` | Download last prediction |
| GET | `/health` | Health check |
