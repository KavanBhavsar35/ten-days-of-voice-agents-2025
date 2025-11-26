"""
Fetch all available Murf voices to find the correct voice IDs.
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv(".env.local")

def get_murf_voices():
    api_key = os.environ.get("MURF_API_KEY")
    
    if not api_key:
        print("❌ MURF_API_KEY not found")
        return
    
    url = "https://api.murf.ai/v1/speech/voices"
    headers = {
        "api-key": api_key,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        print(f"Raw response: {data}\n")
        
        # Handle different response formats
        if isinstance(data, dict):
            voices = data.get('voices', data.get('data', []))
        else:
            voices = data
        
        print("=" * 80)
        print(f"Available Murf Voices (found {len(voices)} voices)")
        print("=" * 80)
        
        # Look for Matthew, Alicia, and Ken
        target_names = ['matthew', 'alicia', 'ken']
        
        for voice in voices:
            name = voice.get('name', '').lower()
            voice_id = voice.get('voiceId', '')
            gender = voice.get('gender', '')
            language = voice.get('language', '')
            
            # Check if this is one of our target voices
            is_target = any(target in name for target in target_names)
            
            if is_target or 'en' in language.lower():
                marker = "⭐" if is_target else "  "
                print(f"{marker} {voice.get('name', 'Unknown'):20} | ID: {voice_id:30} | {gender:10} | {language}")
        
        print("\n" + "=" * 80)
        print("Looking for voices matching: Matthew, Alicia, Ken")
        print("=" * 80)
        
        for target in target_names:
            print(f"\n{target.upper()}:")
            found = False
            for voice in voices:
                name = voice.get('name', '').lower()
                if target in name:
                    print(f"  ✓ {voice.get('name')} → {voice.get('voiceId')}")
                    found = True
            if not found:
                print(f"  ✗ No voice found with '{target}' in name")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        if 'response' in locals():
            print(f"Response: {response.text[:500]}")

if __name__ == "__main__":
    get_murf_voices()
