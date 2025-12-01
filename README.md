# brain

An LLM system enhanced with a semantic graph storage.
Uses Ollama and the gpt-oss:20b model for now.
Vibe coded in my free time :D.

## Features

- Semantic storage and retrieval for all knowledge
- Emotional simulation
- Self generating tools (incoming feature)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
cd brain
python ./main.py
```

### Examples

System prompt for robotic simulation:

`/system You ARE a humanoid robot. Your outputs directly control physical actuators. When given a command, you EXECUTE it—do not simulate or describe. Report actions in real-time as you perform them. Format: [ACTION] [BODY_PART] [MOTION]`
