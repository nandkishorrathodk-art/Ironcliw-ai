# Backend - Ironcliw API Server

This directory contains the FastAPI backend server that powers Ironcliw with Claude AI integration.

## Structure

```
backend/
├── chatbots/
│   └── claude_chatbot.py    # Claude AI integration
├── main.py                  # FastAPI application & endpoints
├── run_server.py           # Server runner with proper paths
├── logs/                   # Application logs
└── static/                 # Static files and demos
```

## Key Components

### main.py
- FastAPI application setup
- Chat endpoints (`/chat`, `/chat/stream`, `/chat/history`)
- Health check endpoint (`/health`)
- Static file serving

### claude_chatbot.py
- Anthropic Claude API integration
- Conversation history management
- Streaming response support
- Token usage tracking

## Running the Backend

```bash
# From project root
python3 backend/run_server.py

# Or use the main launcher
python3 start_system.py
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/chat` | POST | Send message and get response |
| `/chat/stream` | POST | Get streaming response |
| `/chat/history` | GET | Get conversation history |
| `/chat/history` | DELETE | Clear conversation history |
| `/chat/mode` | GET | Get current mode (always "claude") |
| `/health` | GET | Health check |
| `/docs` | GET | Interactive API documentation |

### Voice Unlock API (Biometric Authentication)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/voice-unlock/status` | GET | Voice unlock system status |
| `/api/voice-unlock/health` | GET | Health check for voice services |
| `/api/voice-unlock/users` | GET | List enrolled voice profiles |
| `/api/voice-unlock/users/{name}` | GET | Get specific user profile |
| `/api/voice-unlock/stats` | GET | Voice unlock statistics |
| `/api/voice-unlock/authenticate` | POST | Authenticate with voice (audio file) |
| `/api/voice-unlock/verify-speaker` | POST | Verify speaker identity |
| `/api/voice-unlock/unlock` | POST | Voice-authenticated screen unlock |
| `/api/voice-unlock/profiles/reload` | POST | Reload speaker profiles |
| `/api/voice-unlock/ws/authenticate` | WS | Real-time voice authentication |

## Environment Variables

Required in `.env` file at project root:
```env
ANTHROPIC_API_KEY=your-api-key-here
CLAUDE_MODEL=claude-3-haiku-20240307
CLAUDE_MAX_TOKENS=1024
CLAUDE_TEMPERATURE=0.7
```

## Dependencies

Core requirements:
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `anthropic` - Claude API client
- `python-dotenv` - Environment variables
- `pydantic` - Data validation

## Development

### Adding New Endpoints
1. Add route in `main.py` ChatbotAPI class
2. Add to router: `self.router.add_api_route(...)`
3. Implement method with proper type hints

### Testing
```bash
# Test chat endpoint
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_input": "Hello Ironcliw!"}'

# Check health
curl http://localhost:8000/health
```

### Testing Voice Unlock API with Postman

The Voice Unlock API provides biometric authentication via voice recognition.

**Important:** The server runs on **port 8000** by default (not 8010).

**Quick curl tests:**
```bash
# Voice Unlock Status
curl http://localhost:8000/api/voice-unlock/status

# Voice Unlock Health
curl http://localhost:8000/api/voice-unlock/health

# List Enrolled Users
curl http://localhost:8000/api/voice-unlock/users

# Get Voice Unlock Stats
curl http://localhost:8000/api/voice-unlock/stats
```

**Using Postman:**
1. Import the collection from `postman/collections/Ironcliw_API_Collection.postman_collection.json`
2. Import the environment from `postman/environments/Ironcliw_Environment.postman_environment.json`
3. Select "Ironcliw Local Development" environment
4. Verify `base_url` is set to `http://localhost:8000`
5. Start the server: `python backend/main.py`
6. Wait ~60-90 seconds for full initialization (ML models loading)
7. Test Voice Unlock endpoints - all should return HTTP 200

**Sample Responses:**

Status endpoint (`/api/voice-unlock/status`):
```json
{
  "enabled": true,
  "ready": true,
  "models_loaded": true,
  "initialized": true,
  "owner_name": "Derek J. Russell",
  "enrolled_users": 1
}
```

Users endpoint (`/api/voice-unlock/users`):
```json
{
  "success": true,
  "users": [
    {
      "speaker_name": "Derek J. Russell",
      "is_primary_user": true,
      "total_samples": 59
    }
  ],
  "count": 1
}
```

**Troubleshooting 404 Errors:**
- Verify server is running on the correct port (default: 8000)
- Ensure server has fully started (~60-90 seconds for ML model loading)
- Check the Postman environment `base_url` matches the server port

## Notes
- All AI processing happens via Claude API (no local models)
- Conversation history is maintained in memory
- CORS is enabled for frontend integration
- Static files served from `/static` directory