# AskBuddyX AWS S3 Deployment Guide

## ⚠️ Important Limitation

**The current Web UI requires a backend server (Flask) that loads the ML model.** S3 can only host static files (HTML/CSS/JS), not Python applications.

## Two Deployment Options

### Option 1: Static Frontend (S3) + Your Mac as Backend (Recommended for Testing)
- Frontend: S3 + CloudFront + Your Domain
- Backend: Your Mac (running Flask server)
- Expose Mac via ngrok or Cloudflare Tunnel

### Option 2: Full AWS Deployment (Production)
- Frontend: S3 + CloudFront + Your Domain  
- Backend: EC2 instance (but MLX won't work - need to convert model)

---

## Option 1: S3 Frontend + Mac Backend (Quick Setup)

### Step 1: Create Static HTML File

Create `webui/static/chat.html`:

```html
<!DOCTYPE html>
<html>
<head>
    <title>AskBuddyX</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; }
        #chat { border: 1px solid #ccc; height: 400px; overflow-y: scroll; padding: 10px; margin-bottom: 10px; }
        .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
        .user { background: #e3f2fd; text-align: right; }
        .assistant { background: #f5f5f5; }
        input { width: 80%; padding: 10px; }
        button { padding: 10px 20px; }
    </style>
</head>
<body>
    <h1>🤖 AskBuddyX</h1>
    <div id="chat"></div>
    <input id="input" type="text" placeholder="Ask me anything...">
    <button onclick="send()">Send</button>
    
    <script>
        // UPDATE THIS WITH YOUR BACKEND URL
        const API_URL = 'https://YOUR_NGROK_URL.ngrok-free.app/chat';
        
        async function send() {
            const input = document.getElementById('input');
            const msg = input.value;
            if (!msg) return;
            
            addMsg(msg, 'user');
            input.value = '';
            
            try {
                const res = await fetch(API_URL, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: msg})
                });
                const data = await res.json();
                addMsg(data.response, 'assistant');
            } catch(e) {
                addMsg('Error: ' + e.message, 'assistant');
            }
        }
        
        function addMsg(text, role) {
            const div = document.createElement('div');
            div.className = 'message ' + role;
            div.textContent = text;
            document.getElementById('chat').appendChild(div);
        }
        
        document.getElementById('input').onkeypress = (e) => {
            if (e.key === 'Enter') send();
        };
    </script>
</body>
</html>
```

### Step 2: Expose Your Mac Backend

**Install ngrok:**
```bash
brew install ngrok
```

**Start your Flask server:**
```bash
cd /Users/kashifsalahuddin/AskBuddyX
source .venv/bin/activate
python webui/app.py
```

**In another terminal, expose it:**
```bash
ngrok http 5001
```

**Copy the HTTPS URL** (e.g., `https://abc123.ngrok-free.app`)

**Update `chat.html`** with your ngrok URL:
```javascript
const API_URL = 'https://abc123.ngrok-free.app/chat';
```

### Step 3: Deploy to S3

**Create S3 bucket:**
```bash
# Replace with your domain
DOMAIN="askbuddyx.yourdomain.com"

aws s3 mb s3://$DOMAIN --region us-east-1

# Enable static website hosting
aws s3 website s3://$DOMAIN \
    --index-document chat.html
```

**Make bucket public:**
```bash
cat > bucket-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [{
    "Sid": "PublicRead",
    "Effect": "Allow",
    "Principal": "*",
    "Action": "s3:GetObject",
    "Resource": "arn:aws:s3:::askbuddyx.yourdomain.com/*"
  }]
}
EOF

aws s3api put-bucket-policy \
    --bucket $DOMAIN \
    --policy file://bucket-policy.json
```

**Upload HTML:**
```bash
aws s3 cp webui/static/chat.html s3://$DOMAIN/chat.html \
    --content-type "text/html"
```

### Step 4: Set Up CloudFront + SSL

**Request SSL certificate:**
```bash
aws acm request-certificate \
    --domain-name $DOMAIN \
    --validation-method DNS \
    --region us-east-1
```

**Validate certificate:**
1. Go to AWS Certificate Manager console
2. Add CNAME records to your domain DNS
3. Wait for validation (5-30 minutes)

**Create CloudFront distribution:**
```bash
aws cloudfront create-distribution \
    --origin-domain-name $DOMAIN.s3-website-us-east-1.amazonaws.com \
    --default-root-object chat.html
```

Note the CloudFront domain (e.g., `d123abc.cloudfront.net`)

### Step 5: Configure Your Domain

Add CNAME record to your domain DNS:
```
Type: CNAME
Name: askbuddyx
Value: d123abc.cloudfront.net
TTL: 300
```

### Step 6: Test

Visit: `https://askbuddyx.yourdomain.com`

---

## Option 2: Full AWS Deployment (Production)

### Problem: MLX Requires Apple Silicon

AWS EC2 doesn't have Apple Silicon instances. You have 3 options:

**A) Convert Model to ONNX/TensorFlow**
- Export your trained adapter
- Convert to ONNX format
- Deploy on EC2 with CPU/GPU

**B) Use AWS Outposts with Mac mini**
- AWS offers Mac mini instances
- Very expensive (~$1/hour)
- Not practical for small projects

**C) Keep Mac as Backend, Use VPN**
- Set up VPN between Mac and AWS
- Use private IP for backend
- More complex but works

### Recommended: Convert to ONNX

**Step 1: Export Model**
```python
# On your Mac
from mlx_lm import load, generate
import onnx

model, tokenizer = load("mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit")
# Export to ONNX (requires additional tools)
# This is complex - consider using Hugging Face Transformers instead
```

**Step 2: Deploy on EC2**
```bash
# Launch EC2 instance
aws ec2 run-instances \
    --image-id ami-0c55b159cbfafe1f0 \
    --instance-type t3.medium \
    --key-name your-key

# SSH and install
ssh ubuntu@ec2-ip
sudo apt update
sudo apt install python3-pip nginx
pip3 install flask onnxruntime transformers

# Deploy your app
# Configure nginx as reverse proxy
# Set up SSL with certbot
```

---

## Quick Start (Simplest Path)

### For Testing (5 minutes):

1. **Start Flask on your Mac:**
   ```bash
   cd /Users/kashifsalahuddin/AskBuddyX
   source .venv/bin/activate
   python webui/app.py
   ```

2. **Expose with ngrok:**
   ```bash
   ngrok http 5001
   ```

3. **Update chat.html with ngrok URL**

4. **Upload to S3:**
   ```bash
   aws s3 cp webui/static/chat.html s3://your-bucket/
   ```

5. **Access via S3 URL:**
   ```
   http://your-bucket.s3-website-us-east-1.amazonaws.com/chat.html
   ```

### For Production:

1. Convert model to ONNX or use Hugging Face Transformers
2. Deploy backend on EC2
3. Set up proper SSL and domain
4. Use CloudFront for caching

---

## Cost Estimates

**Option 1 (S3 + ngrok + Your Mac):**
- S3: $0.50/month
- CloudFront: $1-5/month
- ngrok: Free (or $8/month for custom domain)
- **Total: ~$2-15/month**

**Option 2 (Full AWS):**
- EC2 t3.medium: $30/month
- S3 + CloudFront: $2-5/month
- **Total: ~$32-35/month**

---

## Troubleshooting

**CORS Error:**
Add to your Flask app:
```python
from flask_cors import CORS
CORS(app)
```

**ngrok URL changes:**
- Use ngrok paid plan for static URL
- Or update S3 file each time

**Model too slow on EC2:**
- Use GPU instance (g4dn.xlarge)
- Or optimize model (quantization)

---

## Next Steps

1. Choose deployment option
2. Set up S3 bucket
3. Configure backend (ngrok or EC2)
4. Set up domain and SSL
5. Test end-to-end
6. Monitor and optimize

**Need help?** Check the main documentation or create an issue on GitHub.