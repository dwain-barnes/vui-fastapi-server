#!/usr/bin/env python3
"""
Ultra-simple LLM chat with TTS integration
Uses Ollama for LLM and OpenAI-compatible TTS server
"""
import requests
import pygame
import tempfile
import os
import io
import threading
import queue
import re
import json

# Configuration
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:latest"  # Change to your installed model
TTS_URL = "http://localhost:8000"  # Your VUI TTS server (OpenAI-compatible)

# Initialize pygame for audio playback
pygame.mixer.init()

def chat_ollama_streaming(message, history=[]):
    """Chat with Ollama using streaming for real-time response"""
    system_prompt = {
        "role": "system", 
        "content": "You are having a natural conversation. Respond naturally and completely to questions. Use your full knowledge and provide detailed answers when appropriate."
    }
    
    if not history:
        messages = [system_prompt, {"role": "user", "content": message}]
    else:
        messages = [system_prompt] + history[-10:] + [{"role": "user", "content": message}]
    
    # Use streaming to get text as it's generated - NO LIMITS
    response = requests.post(f"{OLLAMA_URL}/api/chat", json={
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": True,  # Enable streaming!
        "options": {
            "temperature": 0.7,
            "top_p": 0.9
            # Removed max_tokens limit!
        }
    }, stream=True)
    
    if response.status_code == 200:
        full_reply = ""
        current_sentence = ""
        
        # Process streaming response
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    if 'message' in data and 'content' in data['message']:
                        chunk = data['message']['content']
                        current_sentence += chunk
                        full_reply += chunk
                        
                        # Check if we have a complete sentence
                        if any(punct in chunk for punct in '.!?'):
                            sentence = current_sentence.strip()
                            if len(sentence) > 10:  # Only speak meaningful sentences (lowered threshold)
                                print(f"🗣️  Speaking: {sentence}")
                                # Speak this sentence immediately - NO LIMITS
                                threading.Thread(target=text_to_speech, args=(sentence,), daemon=True).start()
                            current_sentence = ""
                        
                        # Show progress
                        print(f"\r🤔 {full_reply}", end="", flush=True)
                        
                except json.JSONDecodeError:
                    continue
        
        print()  # New line after streaming
        
        # Speak any remaining text
        if current_sentence.strip():
            threading.Thread(target=text_to_speech, args=(current_sentence.strip(),), daemon=True).start()
        
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": full_reply})
        return full_reply, history
    else:
        return f"Ollama Error: {response.status_code}", history

def text_to_speech(text):
    """Convert text to speech - non-blocking for real-time streaming"""
    try:
        # NO LENGTH LIMITS - send full text to TTS
        response = requests.post(f"{TTS_URL}/v1/audio/speech", 
            json={
                "model": "vui",
                "input": text,  # Full text, no truncation
                "response_format": "wav"
            },
            timeout=60  # Longer timeout for longer sentences
        )
        
        if response.status_code == 200:
            # Play immediately without blocking
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name
            
            # Play audio in a way that doesn't block
            pygame.mixer.Sound(tmp_path).play()
            
            # Clean up after a delay
            threading.Timer(10.0, lambda: safe_delete(tmp_path)).start()
            return True
        else:
            print(f"TTS Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"TTS Error: {e}")
        return False

def safe_delete(filepath):
    """Safely delete temp file"""
    try:
        if os.path.exists(filepath):
            os.unlink(filepath)
    except:
        pass

def main():
    print("🤖 Ollama Chat with VUI TTS")
    print(f"LLM: {OLLAMA_MODEL} | TTS: VUI")
    print("Type 'quit' to exit, 'mute' to toggle TTS\n")
    
    history = []
    tts_enabled = True
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'mute':
                tts_enabled = not tts_enabled
                print(f"TTS {'enabled' if tts_enabled else 'disabled'}")
                continue
            elif not user_input:
                continue
            
            # Get streaming Ollama response with real-time TTS
            print("🤔 Thinking...")
            reply, history = chat_ollama_streaming(user_input, history)
            print(f"\n✅ Complete: {reply}")
            
            # No need for additional TTS here - already handled in streaming
            
            print()  # Empty line for readability
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("Goodbye! 👋")

if __name__ == "__main__":
    main()