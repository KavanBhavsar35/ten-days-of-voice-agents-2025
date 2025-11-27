import logging
import json
from datetime import datetime
import os
from typing import Annotated, Literal

from db import FraudDatabase
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    function_tool,
    RunContext
)
from livekit.plugins import murf, deepgram, google, silero

load_dotenv(".env.local")
logger = logging.getLogger("fraud-agent")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(SCRIPT_DIR, "fraud_db.sqlite")

# --- 1. The Fraud Agent ---
class FraudAgent(Agent):
    def __init__(self, case_data,db_path=None):
        self.db = FraudDatabase(db_path or DB_FILE)
        self.case = case_data
        
        tx = case_data.get('transaction', {})
        
        super().__init__(
            instructions=f"""
            You are 'Sarah' from the **Chase Bank Fraud Security Team**.
            You are calling {case_data.get('name')} regarding a suspicious transaction on their card ending in {case_data.get('card_last4')}.
            
            YOUR SECURITY DATA (PRIVATE):
            - Correct Security Answer: "{case_data.get('security_answer')}"
            
            TRANSACTION DETAILS:
            - {tx.get('merchant')}
            - {tx.get('amount')}
            - {tx.get('location')}
            - {tx.get('timestamp')}
            
            CALL FLOW:
            1. **Introduction:** State your name and that this is an urgent security alert.
            2. **Verification:** Ask the user to verify identity by answering: "{case_data.get('security_question')}".
               - IF WRONG: Politely apologize and end the call.
               - IF RIGHT: Proceed.
            3. **Investigation:** Read transaction details. Ask: "Did you authorize this charge?"
            4. **Resolution:**
               - YES (Authorized) -> Mark SAFE.
               - NO (Unauthorized) -> Mark FRAUD.
            5. **Action:** You MUST call `resolve_fraud_case` to save the result.
            """
        )

    @function_tool
    async def resolve_fraud_case(
        self, 
        ctx: RunContext, 
        decision: Annotated[Literal["confirmed_safe", "confirmed_fraud", "verification_failed"], "The final outcome"],
        summary_note: Annotated[str, "A brief summary of what the customer said"]
    ):
        """Call this to close the case and update the database."""
        self.db.update_case_status(self.case.get("customer_id"), decision, summary_note)
        
        if decision == "confirmed_fraud":
            return "Marked as FRAUD. Card blocked. New card mailed."
        elif decision == "confirmed_safe":
            return "Marked as SAFE. Restrictions lifted."
        else:
            return "Verification failed. Account locked."

# --- 3. Entrypoint & Prewarm ---

def prewarm(proc: JobProcess):
    # Load VAD model in the prewarm phase
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    try:
        ctx.log_context_fields = {"room": ctx.room.name}
        await ctx.connect()

        # 1. Load the mock case
        db = FraudDatabase(DB_FILE)
        case = db.get_case_by_name("Amit Sharma")
        
        if not case:
            logger.warning("Amit Sharma not found. Using fallback data.")
            case = {
                "customer_id": "UNKNOWN",
                "name": "Customer",
                "card_last4": "0000",
                "security_question": "your zip code",
                "security_answer": "00000",
                "transaction": {"merchant": "Unknown", "amount": "$0", "location": "Unknown", "timestamp": "Recently"}
            }
        
        # 2. Initialize Session with Female Voice
        session = AgentSession(
            stt=deepgram.STT(model="nova-3"),
            llm=google.LLM(model="gemini-2.5-flash"),
            tts=murf.TTS(
                voice="en-US-alicia", # FIXED: Using a known working female voice
                style="Conversation",
                text_pacing=True
            ),
            vad=ctx.proc.userdata["vad"], 
        )

        # 3. Start Agent
        agent = FraudAgent(case_data=case)
        await session.start(agent=agent, room=ctx.room)
        
        # 4. Trigger the call
        await session.say(f"Hello, am I speaking with {case['name']}?", allow_interruptions=True)

    except Exception as e:
        logger.error(f"CRITICAL ERROR in entrypoint: {e}")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))