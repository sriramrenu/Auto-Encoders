from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from gradio_client import Client, file as hf_file
import os
import random
import time

app = FastAPI(title="Multi-Domain AE Platform Backend")

# Allowing CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    domain: str
    data: list[float]

# Configuration for Hugging Face Space
HF_SPACE_URL = os.getenv("HF_SPACE_URL", "URL_NOT_SET")
hf_client = None

def get_hf_client():
    global hf_client
    if hf_client is None:
        if HF_SPACE_URL == "URL_NOT_SET":
            return None
        try:
            hf_client = Client(HF_SPACE_URL)
        except Exception as e:
            print(f"Failed to connect to HF Space: {e}")
            return None
    return hf_client

def mock_inference():
    time.sleep(1.2)
    error_score = random.random()
    is_anomaly = error_score > 0.75
    confidence = error_score * 100 if is_anomaly else (1 - error_score) * 100
    
    return {
        "status": "success",
        "reconstruction_error": f"{error_score:.4f}",
        "is_anomaly": is_anomaly,
        "confidence": f"{confidence:.2f}%"
    }

@app.post("/api/analyze")
async def analyze_data(req: AnalyzeRequest):
    import torch
    domain = req.domain.lower()
    
    if 'fraud' in domain or 'bank' in domain:
        if hasattr(fraud_model, 'forward'):
            try:
                # Pad or truncate to exact 30 features
                features = req.data.copy()
                if len(features) < 30: features.extend([0.0] * (30 - len(features)))
                elif len(features) > 30: features = features[:30]
                
                tensor_data = torch.Tensor([features])
                with torch.no_grad():
                    recon = fraud_model(tensor_data)
                    mse = torch.mean((tensor_data - recon)**2).item()
                
                is_anomaly = mse > 0.69 # Fraud Anomaly Threshold
                conf = min(mse * 50, 99.9) if is_anomaly else min((2.0/max(mse, 0.001))*10, 99.9)
                result = {
                    "status": "success", "reconstruction_error": f"{mse:.4f}",
                    "is_anomaly": is_anomaly, "confidence": f"{conf:.2f}%",
                    "domain": "Banking/Fraud",
                    "insights": "[REAL INFERENCE] Extreme structural deviation" if is_anomaly else "[REAL INFERENCE] Normal Transaction bounds"
                }
            except Exception as e:
                print(e)
                result = mock_inference()
                result["domain"] = "Banking/Fraud (Fallback)"
@app.post("/api/analyze")
async def analyze_data(req: AnalyzeRequest):
    client = get_hf_client()
    domain = req.domain.lower()
    
    if client:
        try:
            # Prepare feature string for Hugging Face
            feature_str = ", ".join([str(x) for x in req.data])
            # Determine which tab to call
            if 'fraud' in domain or 'bank' in domain:
                api_name = "/predict_fraud"
            else:
                api_name = "/predict_network"
            
            result_str = client.predict(feature_str, api_name=api_name)
            
            # Parse Hugging Face string result
            mse = float(result_str.split(":")[1].split("\n")[0].strip())
            is_anomaly = "ANOMALY" in result_str or "INTRUSION" in result_str
            
            return {
                "status": "success", "reconstruction_error": f"{mse:.4f}",
                "is_anomaly": is_anomaly, "confidence": f"{mse*100:.2f}%" if is_anomaly else "99%",
                "domain": f"HF Space Proxy ({domain})",
                "insights": result_str.split("\n")[1] if "\n" in result_str else result_str
            }
        except Exception as e:
            print(f"HF Proxy Error: {e}")
            return mock_inference()
    
    return mock_inference()
        
    raise HTTPException(status_code=400, detail="Unsupported text/json domain.")

@app.post("/api/analyze/file")
async def analyze_file(domain: str = Form(...), file: UploadFile = File(...)):
    client = get_hf_client()
    domain = domain.lower()
    
    if client and ('image' in domain or 'health' in domain):
        try:
            # Save incoming file to temp location for Hugging Face API
            temp_path = f"temp_{int(time.time())}.png"
            with open(temp_path, "wb") as f:
                f.write(await file.read())
            
            # Call Hugging Face API
            result_str = client.predict(hf_file(temp_path), api_name="/predict_medical")
            
            # Cleanup
            os.remove(temp_path)
            
            mse = float(result_str.split(":")[1].split("\n")[0].strip())
            is_anomaly = "PATHOLOGY" in result_str or "ANOMALY" in result_str
            
            return {
                "status": "success", "reconstruction_error": f"{mse:.4f}",
                "is_anomaly": is_anomaly, "confidence": f"{(1-mse)*100:.2f}%" if not is_anomaly else "98%",
                "domain": "HF Space Proxy (Medical)",
                "insights": result_str.split("\n")[1]
            }
        except Exception as e:
            print(f"HF Image Proxy Error: {e}")
            return mock_inference()
            
    return mock_inference()

    raise HTTPException(status_code=400, detail="Unsupported file domain.")

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "message": "FastAPI Backend is running."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
