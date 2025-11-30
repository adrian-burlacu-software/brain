"""
Ollama Chat Interface for gpt-oss:20b model
Supports conversation history and thinking modes
"""

import requests
import json
from typing import Generator


# ANSI color codes
class Colors:
    PINK = "\033[95m"
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"


class OllamaChat:
    """Chat interface for Ollama models with conversation support."""
    
    def __init__(
        self,
        model: str = "gpt-oss:20b",
        base_url: str = "http://localhost:11434",
        thinking_mode: str = "medium",
        verbose: bool = False
    ):
        """
        Initialize the Ollama chat interface.
        
        Args:
            model: The Ollama model to use
            base_url: Ollama API base URL
            thinking_mode: Thinking budget - "low", "medium", or "high"
            verbose: Show thinking process in real time
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.conversation_history: list[dict] = []
        self.thinking_mode = thinking_mode
        self.verbose = verbose
        
        # Thinking mode configurations (token budgets)
        self.thinking_budgets = {
            "low": 1024,
            "medium": 4096,
            "high": 16384
        }
    
    def _get_thinking_budget(self) -> int:
        """Get the token budget for the current thinking mode."""
        return self.thinking_budgets.get(self.thinking_mode, 4096)
    
    def _build_options(self) -> dict:
        """Build model options including thinking configuration."""
        return {
            "num_predict": -1,  # No limit on response length
            "temperature": 0.7,
            "top_p": 0.9,
        }
    
    def chat(self, message: str, stream: bool = True) -> str | Generator[str, None, None]:
        """
        Send a message and get a response.
        
        Args:
            message: The user message
            stream: Whether to stream the response
            
        Returns:
            The assistant's response (or generator if streaming)
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": message
        })
        
        # Build the request payload
        payload = {
            "model": self.model,
            "messages": self.conversation_history,
            "stream": stream,
            "options": self._build_options(),
            "think": True  # Enable thinking mode
        }
        
        url = f"{self.base_url}/api/chat"
        
        if stream:
            return self._stream_response(url, payload)
        else:
            return self._get_response(url, payload)
    
    def _stream_response(self, url: str, payload: dict) -> Generator[str, None, None]:
        """Stream the response from Ollama."""
        full_response = ""
        thinking_content = ""
        in_thinking = False
        
        try:
            with requests.post(url, json=payload, stream=True) as response:
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        
                        # Handle thinking content if present
                        if "message" in data:
                            msg = data["message"]
                            
                            # Check for thinking content
                            if "thinking" in msg and msg["thinking"]:
                                thinking_chunk = msg.get("thinking", "")
                                thinking_content += thinking_chunk
                                
                                if self.verbose:
                                    if not in_thinking:
                                        yield f"\n{Colors.PINK}{Colors.DIM}"
                                        in_thinking = True
                                    yield thinking_chunk
                            
                            # Regular content
                            if "content" in msg and msg["content"]:
                                # Close thinking section if we were in it
                                if self.verbose and in_thinking:
                                    yield f"{Colors.RESET}\n\n{Colors.GREEN}"
                                    in_thinking = False
                                elif not in_thinking and not full_response:
                                    # Start green for response if no thinking shown
                                    yield f"{Colors.GREEN}"
                                
                                chunk = msg["content"]
                                full_response += chunk
                                yield chunk
                        
                        # Check if done
                        if data.get("done", False):
                            break
            
            # Reset colors at end
            yield Colors.RESET
            
            # Add assistant response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": full_response
            })
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Error communicating with Ollama: {e}"
            yield error_msg
    
    def _get_response(self, url: str, payload: dict) -> str:
        """Get the complete response from Ollama."""
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            
            assistant_message = data.get("message", {}).get("content", "")
            
            # Add to history
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            return assistant_message
            
        except requests.exceptions.RequestException as e:
            return f"Error communicating with Ollama: {e}"
    
    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
        print("Conversation history cleared.")
    
    def set_thinking_mode(self, mode: str):
        """
        Set the thinking mode.
        
        Args:
            mode: "low", "medium", or "high"
        """
        if mode in self.thinking_budgets:
            self.thinking_mode = mode
            print(f"Thinking mode set to: {mode}")
        else:
            print(f"Invalid mode. Choose from: {list(self.thinking_budgets.keys())}")
    
    def set_verbose(self, enabled: bool):
        """
        Enable or disable verbose mode to see thinking in real time.
        
        Args:
            enabled: True to enable, False to disable
        """
        self.verbose = enabled
        status = "enabled" if enabled else "disabled"
        print(f"Verbose mode {status}.")
    
    def set_system_prompt(self, prompt: str):
        """Set a system prompt at the beginning of the conversation."""
        # Remove existing system prompt if any
        self.conversation_history = [
            msg for msg in self.conversation_history 
            if msg.get("role") != "system"
        ]
        # Add new system prompt at the beginning
        self.conversation_history.insert(0, {
            "role": "system",
            "content": prompt
        })
        print("System prompt set.")
    
    def get_history(self) -> list[dict]:
        """Get the conversation history."""
        return self.conversation_history.copy()


def main():
    """Interactive chat interface."""
    print(f"{Colors.CYAN}{Colors.BOLD}" + "=" * 60)
    print("Ollama Chat Interface - gpt-oss:20b")
    print("Thinking Mode: Medium")
    print("=" * 60 + f"{Colors.RESET}")
    print(f"\n{Colors.YELLOW}Commands:{Colors.RESET}")
    print("  /clear     - Clear conversation history")
    print("  /thinking  - Change thinking mode (low/medium/high)")
    print(f"  /verbose   - Toggle verbose mode ({Colors.PINK}thinking{Colors.RESET} in pink)")
    print("  /system    - Set system prompt")
    print("  /history   - Show conversation history")
    print("  /quit      - Exit the chat")
    print(f"{Colors.CYAN}" + "=" * 60 + f"{Colors.RESET}")
    
    chat = OllamaChat(
        model="gpt-oss:20b",
        thinking_mode="medium",
        verbose=False
    )
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                cmd = user_input.lower().split()[0]
                
                if cmd == "/quit":
                    print("Goodbye!")
                    break
                    
                elif cmd == "/clear":
                    chat.clear_history()
                    continue
                    
                elif cmd == "/thinking":
                    parts = user_input.split()
                    if len(parts) > 1:
                        chat.set_thinking_mode(parts[1].lower())
                    else:
                        print(f"Current thinking mode: {chat.thinking_mode}")
                        print("Usage: /thinking <low|medium|high>")
                    continue
                
                elif cmd == "/verbose":
                    parts = user_input.split()
                    if len(parts) > 1:
                        value = parts[1].lower()
                        if value in ("on", "true", "1", "yes"):
                            chat.set_verbose(True)
                        elif value in ("off", "false", "0", "no"):
                            chat.set_verbose(False)
                        else:
                            print("Usage: /verbose <on|off>")
                    else:
                        # Toggle
                        chat.set_verbose(not chat.verbose)
                    continue
                    
                elif cmd == "/system":
                    prompt = user_input[7:].strip()
                    if prompt:
                        chat.set_system_prompt(prompt)
                    else:
                        print("Usage: /system <your system prompt>")
                    continue
                    
                elif cmd == "/history":
                    history = chat.get_history()
                    if history:
                        print("\n--- Conversation History ---")
                        for i, msg in enumerate(history):
                            role = msg["role"].capitalize()
                            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                            print(f"{i+1}. [{role}]: {content}")
                        print("----------------------------")
                    else:
                        print("No conversation history yet.")
                    continue
                    
                else:
                    print(f"Unknown command: {cmd}")
                    continue
            
            # Send message and stream response
            print(f"\n{Colors.BOLD}Brain:{Colors.RESET} ", end="", flush=True)
            for chunk in chat.chat(user_input, stream=True):
                print(chunk, end="", flush=True)
            print()  # New line after response
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
