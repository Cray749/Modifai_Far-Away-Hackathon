import logging
import json
import uuid
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from modifai.agents.schemas import AgentPackage
from modifai.core.llm_provider import get_llm_provider

logger = logging.getLogger(__name__)

app = FastAPI(title="Modifai Agent Runtime")
# In-memory store of registered agents
registered_agents = {}

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str

@app.post("/agents/{agent_id}")
def chat_with_agent(agent_id: str, req: ChatRequest):
    if agent_id not in registered_agents:
        raise HTTPException(status_code=404, detail="Agent not found")
        
    package: AgentPackage = registered_agents[agent_id]
    
    provider = get_llm_provider()
    
    # Construct the runtime system prompt combining the package details
    system_prompt = package["system_prompt"]
    system_prompt += "\n\nAdditional Instructions:\n"
    for inst in package.get("instructions", []):
        system_prompt += f"- {inst}\n"
        
    logger.info("Agent %s received message: %s", package["name"], req.message)
    
    try:
        # We don't have a specific schema, just plain text response
        # wait, BaseLLMProvider.generate requires JSON return? 
        # By default our get_llm_provider setup uses safe_json_generation if response_schema is passed,
        # but if response_schema is None, BedrockProvider currently returns JSON if we force tool, else plain text.
        # Let's use a strict response_schema to ensure the response is {"answer": "..."} across all providers.
        
        response_schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"}
            },
            "required": ["answer"]
        }
        
        
        raw_output = provider.generate(
            system_prompt=system_prompt,
            user_prompt=req.message,
            response_schema=response_schema
        )
        
        answer = raw_output.get("answer", "I could not generate an answer.")
        
        event = {
            "event_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": "agent_runtime",
            "iteration": 0,
            "decision": f"Handled chat for {agent_id}",
            "reason": None,
            "data": {"agent_id": agent_id, "user_message": req.message, "answer": answer}
        }
        try:
            with open("events.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except Exception:
            pass
            
        return ChatResponse(answer=answer)
        
    except Exception as e:
        logger.error("Error generating agent response: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

def register_agent(agent_id: str, package: AgentPackage):
    """Register an agent package into the runtime."""
    registered_agents[agent_id] = package
    logger.info("Agent registered: /agents/%s", agent_id)

@app.get("/chat/{agent_id}", response_class=HTMLResponse)
def get_chat_ui(agent_id: str):
    if agent_id not in registered_agents:
        raise HTTPException(status_code=404, detail="Agent not found")
        
    package = registered_agents[agent_id]
    name = package.get("name", agent_id)
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chat with {name}</title>
        <style>
            body {{ font-family: sans-serif; max-width: 600px; margin: 40px auto; }}
            #chat-box {{ height: 400px; border: 1px solid #ccc; padding: 10px; overflow-y: auto; margin-bottom: 10px; }}
            .msg {{ margin-bottom: 10px; }}
            .user {{ color: blue; }}
            .agent {{ color: green; }}
            input[type="text"] {{ width: 80%; padding: 8px; }}
            button {{ padding: 8px 16px; cursor: pointer; }}
        </style>
    </head>
    <body>
        <h2>Chat with {name}</h2>
        <div id="chat-box"></div>
        <div>
            <input type="text" id="user-input" placeholder="Ask something..." onkeypress="handleKeyPress(event)">
            <button onclick="sendMessage()">Send</button>
        </div>
        
        <script>
            async function sendMessage() {{
                const input = document.getElementById('user-input');
                const text = input.value.trim();
                if (!text) return;
                
                const chatBox = document.getElementById('chat-box');
                chatBox.innerHTML += `<div class="msg user"><b>You:</b> ${{text}}</div>`;
                input.value = '';
                chatBox.scrollTop = chatBox.scrollHeight;
                
                try {{
                    const response = await fetch('/agents/{agent_id}', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify({{message: text}})
                    }});
                    const data = await response.json();
                    
                    if (data.answer) {{
                        chatBox.innerHTML += `<div class="msg agent"><b>{name}:</b> ${{data.answer}}</div>`;
                    }} else {{
                        chatBox.innerHTML += `<div class="msg agent"><b>Error:</b> ${{JSON.stringify(data)}}</div>`;
                    }}
                }} catch (e) {{
                    chatBox.innerHTML += `<div class="msg agent"><b>Error:</b> ${{e.message}}</div>`;
                }}
                chatBox.scrollTop = chatBox.scrollHeight;
            }}
            
            function handleKeyPress(e) {{
                if (e.key === 'Enter') sendMessage();
            }}
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

