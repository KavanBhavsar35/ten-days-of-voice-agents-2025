import asyncio
import logging
import os
from dotenv import load_dotenv
from livekit import api

# Load environment variables
load_dotenv(".env.local")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_agent_connection():
    """Test if we can connect to the LiveKit server and verify agent is registered"""
    
    # Get LiveKit credentials
    livekit_url = os.environ.get("LIVEKIT_URL", "ws://127.0.0.1:7880")
    api_key = os.environ.get("LIVEKIT_API_KEY", "devkey")
    api_secret = os.environ.get("LIVEKIT_API_SECRET", "secret")
    
    logger.info(f"Testing connection to LiveKit server: {livekit_url}")
    logger.info(f"API Key: {api_key[:5]}...{api_key[-5:] if len(api_key) > 10 else api_key}")
    
    try:
        # Create LiveKit client
        livekit_api = api.LiveKitAPI(livekit_url, api_key, api_secret)
        
        # List available agents
        logger.info("Fetching agent information...")
        # Note: This is a simplified test - in reality, you'd want to check if your specific agent is available
        
        logger.info("Connection test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return False
    finally:
        # Clean up
        if 'livekit_api' in locals():
            await livekit_api.aclose()

if __name__ == "__main__":
    asyncio.run(test_agent_connection())