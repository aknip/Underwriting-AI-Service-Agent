# Underwriting-AI-Service-Agent

## Virtual environment using uv
- Install: curl -LsSf https://astral.sh/uv/install.sh | sh
- Create a virtual environment in current dir in folder .venv.
	- `uv venv`
- Activate environment
	- `source .venv/bin/activate`
- Deactivate environment
	- `deactivate`

## Install dependencies
uv pip install -r requirements.txt

# Set keys for Reddit and OpenAI
export REDDIT_CLIENT_ID=
export REDDIT_CLIENT_SECRET=
export OPENAI_API_KEY=


## Run
- python main.py
or 
- fastapi dev --port 8080

Open:
http://localhost:8080
http://localhost:8080/docs



## Reddit Setup
see Video 
- https://www.youtube.com/watch?v=eDNI7-HLWcM

https://www.reddit.com/prefs/apps
App ID: UFpsIsEXDdypT7mZYQpNyQ
App Secret: JRKNKYHRiUVc1q-5HzJzQYGGe8KXIg


## Pydantic AI agents
- https://ai.pydantic.dev/agents/
- https://skolo-online.medium.com/create-ai-agent-crud-application-with-pydanticai-step-by-step-524f36aba381

## Sources:
- https://medium.com/google-cloud/how-i-built-an-agent-with-pydantic-ai-and-google-gemini-4887e5dd041d
- https://github.com/GoogleCloudPlatform/generative-ai/tree/main/gemini/sample-apps/swot-agent




## Google Vertex setup (instead of OpenAI)

### Set environment variables for Google and Reddit
export GOOGLE_CLOUD_PROJECT=swot-ai-agent

### Google Setup
see Docs: 
- https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/sample-apps/SETUP.md
- For GOOGLE_APPLICATION_CREDENTIALS:
	- Install gcloud CLI, then:
	- gcloud init
	- gcloud auth application-default login
	- see https://cloud.google.com/docs/authentication/external/set-up-adc
