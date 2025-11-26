import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv(".env.local")

# Test Murf API directly
def test_murf_api():
    api_key = os.environ.get("MURF_API_KEY")
    if not api_key:
        print("MURF_API_KEY not found in environment variables")
        return False
        
    url = "https://api.murf.ai/v1/speech/synthesize"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "audio/wav"
    }
    
    payload = {
        "text": "Hello, this is a test of the Murf text to speech API.",
        "voiceId": "en-US-matthew",
        "style": "Conversation",
        "sampleRate": 24000,
        "audioEncoding": "wav"
    }
    
    try:
        print("Testing Murf API...")
        response = requests.post(url, json=payload, headers=headers)
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {response.headers}")
        
        if response.status_code == 200:
            print("Success! Received audio data.")
            print(f"Audio data length: {len(response.content)} bytes")
            # Save to file for verification
            with open("test_output.wav", "wb") as f:
                f.write(response.content)
            print("Audio saved to test_output.wav")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Exception occurred: {e}")
        return False

if __name__ == "__main__":
    test_murf_api()