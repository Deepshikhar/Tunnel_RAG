# Ollama Inference Server on Google Colab (via Cloudflare Tunnel)

This project demonstrates how to run an **Ollama LLM** on **Google Colab** and securely expose it to the public internet using a **Cloudflare Tunnel**, allowing it to be consumed remotely (e.g., by a local **RAG application**).

This setup is useful when:

* You want to use **free/temporary GPU/CPU resources** on Colab
* You want to **keep your RAG pipeline local**
* You need a **remote LLM inference backend** accessible via HTTP

---

## Architecture Overview

```
Local RAG Application
        |
        |  (HTTP requests)
        v
Cloudflare Tunnel (Public HTTPS URL)
        |
        v
Google Colab Notebook
        |
        v
Ollama Server (Mistral/ LLama Model)
```

---

## Requirements

### On Google Colab

* Google Colab notebook
* Internet access enabled
* Python runtime (CPU is sufficient for Mistral small models)

### On Local Machine

* Python 3.9+
* RAG framework (e.g., LangChain)
* HTTP access to the Cloudflare tunnel URL

---

## Step-by-Step Setup (Colab)

### 1. Start a New Colab Notebook

Open a new notebook at:
[https://colab.research.google.com/](https://colab.research.google.com/)

---

### 2. Install Ollama

```bash
!curl -fsSL https://ollama.com/install.sh | sh
```

Verify installation:

```bash
!ollama --version
```

---

### 3. Pull the Mistral Model

```bash
!ollama pull mistral
```

This downloads the model into the Colab environment.

---

### 4. Start the Ollama Server

```bash
!ollama serve
```

Ollama will start listening on:

```
http://localhost:11434
```

⚠️ **Keep this cell running**.

---

### 5. Download Cloudflare Tunnel (cloudflared)

Open a **new cell** and run:

```bash
# Download the latest Cloudflare Tunnel binary for Linux
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64

# Make the binary executable
!chmod +x cloudflared-linux-amd64
```

---

### 6. Expose Ollama via Cloudflare Tunnel

```bash
# Create a public HTTPS tunnel to the local Ollama server
!./cloudflared-linux-amd64 tunnel --url http://localhost:11434
```

After running this command, you will see output similar to:

```
https://random-name.trycloudflare.com
```

✅ **This is your public Ollama endpoint**

---

## Using the Tunnel URL

You can now send requests to Ollama from your local machine.

### Example: Test with curl

```bash
curl https://<your-cloudflare-url>/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral",
    "prompt": "Explain RAG in simple terms"
  }'
```

---

## Using with a Local RAG Application

In your local RAG code (LangChain example):

```python
from langchain_community.llms import Ollama

llm = Ollama(
    base_url="https://<your-cloudflare-url>",
    model="mistral"
)
```

Your retriever, embeddings, and vector store can remain fully local.

---

## Important Notes

### ⚠️ Colab Limitations

* Sessions are **temporary**
* The tunnel URL changes every restart
* Not suitable for production use

### 🔐 Security

* The tunnel URL is **public**
* Do not expose sensitive data
* Use only for testing, demos, or development

---

## Use Cases

* Remote LLM inference for local RAG
* Lightweight LLM experimentation
* Portfolio or proof-of-concept projects
* Testing distributed LLM architectures

---

## License

This project is for educational and experimental purposes.
