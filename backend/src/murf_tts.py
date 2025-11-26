import asyncio
import contextlib
import logging
import os
from typing import AsyncIterable
import base64

import requests
from livekit import rtc
from livekit.agents import tokenize, tts

logger = logging.getLogger(__name__)


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: str = "en-US-ryan",
        style: str = "Conversational",
        tokenizer: tokenize.SentenceTokenizer = tokenize.basic.SentenceTokenizer(),
    ) -> None:
        """
        Initialize Murf TTS.
        
        Args:
            voice: The voice ID to use (e.g., "en-US-ryan")
            style: The speaking style (e.g., "Conversational", "Narration")
            tokenizer: The tokenizer to use for sentence segmentation
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,
            ),
            sample_rate=24000,
            num_channels=1,
        )
        self._voice = voice
        self._style = style
        self._tokenizer = tokenizer
        self._api_key = os.environ.get("MURF_API_KEY")
        
        if not self._api_key:
            raise ValueError("MURF_API_KEY environment variable is required")

    def _synthesize_audio_sync(self, text: str) -> bytes:
        """
        Synchronous method to synthesize speech using Murf API.
        
        Args:
            text: The text to synthesize
            
        Returns:
            Audio data as bytes
        """
        url = "https://api.murf.ai/v1/speech/generate"
        
        headers = {
            "api-key": self._api_key,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        payload = {
            "voiceId": self._voice,
            "style": self._style,
            "text": text,
            "format": "WAV",
            "sampleRate": 24000,
            "channelType": "MONO",
            "encodeAsBase64": False,
            "speed": 1.0,  # Normal speed
            "pitch": 0,    # Normal pitch
        }
        
        try:
            logger.info(f"Synthesizing with Murf: voice={self._voice}, text_length={len(text)}")
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Murf API returns JSON with audio URL or base64 data
            response_data = response.json()
            
            if 'audioFile' in response_data:
                # Download the audio file
                audio_url = response_data['audioFile']
                audio_response = requests.get(audio_url, timeout=30)
                audio_response.raise_for_status()
                return audio_response.content
            elif 'audioContent' in response_data:
                # Base64 encoded audio
                return base64.b64decode(response_data['audioContent'])
            else:
                logger.error(f"Unexpected Murf API response: {response_data}")
                raise ValueError("Unexpected API response format")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error synthesizing speech with Murf: {e}")
            if 'response' in locals():
                logger.error(f"Response status: {response.status_code}")
                logger.error(f"Response body: {response.text[:500]}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in Murf TTS: {e}")
            raise

    @contextlib.asynccontextmanager
    async def synthesize(self, text: str, *, conn_options=None):
        """
        Synthesize text to speech (async context manager).
        
        Args:
            text: The text to synthesize
            conn_options: Connection options (unused but required by interface)
            
        Yields:
            Async generator of synthesized audio
        """
        
        async def _do_synthesize():
            try:
                # Run the synchronous API call in a thread pool
                loop = asyncio.get_event_loop()
                audio_data = await loop.run_in_executor(None, self._synthesize_audio_sync, text)
                
                # Skip WAV header (44 bytes) if present
                if len(audio_data) > 44 and audio_data[:4] == b'RIFF':
                    audio_data = audio_data[44:]
                
                # Create audio frame with raw PCM data
                audio_frame = rtc.AudioFrame(
                    data=audio_data,
                    sample_rate=24000,
                    num_channels=1,
                    samples_per_channel=(len(audio_data) // 2)  # 16-bit audio = 2 bytes per sample
                )
                
                yield tts.SynthesizedAudio(request_id="", frame=audio_frame)
            except Exception as e:
                logger.error(f"Error in synthesize: {e}")
                raise
        
        yield _do_synthesize()

    async def aclose(self) -> None:
        """Close the TTS instance."""
        pass


# Create a default instance
def create_tts(voice: str = "en-US-ryan", style: str = "Conversational") -> TTS:
    """
    Create a Murf TTS instance.
    
    Args:
        voice: The voice ID to use
        style: The speaking style
        
    Returns:
        Murf TTS instance
    """
    return TTS(voice=voice, style=style)
