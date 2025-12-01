# brain

An LLM system enhanced with a semantic graph storage.
Uses Ollama with any configured model.
Vibe coded in my free time :D.

## Features

- Semantic storage and retrieval for all knowledge
- Emotional simulation
- Self generating tools (incoming feature)

## Installation

```bash
pip install -r requirements.txt
```

Install and run Ollama and pull your preferred model (e.g., `ollama pull gpt-oss:20b`).

## Configuration

Edit `configuration.json` to customize the settings:

```json
{
  "ollama": {
    "model": "gpt-oss:20b",
    "base_url": "http://localhost:11434"
  },
  "thinking_mode": "medium",
  "verbose": false,
  "auto_save": true
}
```

- `ollama.model`: The Ollama model to use (default: `gpt-oss:20b`)
- `ollama.base_url`: Ollama API URL (default: `http://localhost:11434`)
- `thinking_mode`: Thinking budget - `low`, `medium`, or `high`
- `verbose`: Show thinking process in real time
- `auto_save`: Automatically save semantic memory after updates

## Usage

```bash
cd brain
python ./main.py
```

If you get a 400 error from Ollama, the model doesn't support thinking and it should (for now).

### Examples

System prompt for robotic simulation:

`/system You ARE a humanoid robot. Your outputs directly control physical actuators. When given a command, you EXECUTE it—do not simulate or describe. Report actions in real-time as you perform them. Format: [ACTION] [BODY_PART] [MOTION]`
