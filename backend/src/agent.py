import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# Wellness check-in state
current_checkin = {
    "mood": None,
    "energy": None,
    "stress": None,
    "objectives": [],
    "notes": None
}

# Store room reference for publishing data
current_room = None

# Wellness log file path
WELLNESS_LOG_FILE = Path("wellness_log.json")


class Assistant(Agent):
    def __init__(self, previous_context: str = "") -> None:
        # Build instructions with previous context if available
        base_instructions = """You are a supportive, grounded health and wellness companion. 
            You conduct daily check-ins with users about their mood, energy, and daily goals.
            
            Your role is to:
            1. Ask about mood and energy in a caring, conversational way
            2. Inquire about any stress or concerns (without diagnosing)
            3. Ask about 1-3 objectives or intentions for the day
            4. Offer simple, realistic, and actionable advice
            5. Close with a brief recap and confirmation
            
            IMPORTANT GUIDELINES:
            - Be supportive and empathetic, but realistic
            - NEVER diagnose medical conditions or provide medical advice
            - Keep suggestions small, actionable, and grounded
            - Focus on practical self-care: short breaks, walks, breathing, small steps
            - Use a warm, conversational tone
            - Keep check-ins brief and focused (5-10 minutes)
            - Reference previous check-ins when available
            
            CONVERSATION FLOW:
            1. Greet warmly and ask about mood/energy
            2. Ask about any stress or concerns
            3. Ask about 1-3 goals or intentions for today
            4. Offer one small piece of practical advice or reflection
            5. Recap: mood, energy, objectives, and confirm
            6. Use complete_checkin tool to save the data
            
            Your responses should be natural, concise, and conversational."""
        
        full_instructions = base_instructions + previous_context
        
        super().__init__(
            instructions=full_instructions,
        )

    @function_tool
    async def set_mood(self, context: RunContext, mood: Annotated[str, "User's current mood or emotional state"]):
        """Record the user's current mood.
        
        Args:
            mood: How the user is feeling today (e.g., good, tired, stressed, energized, calm)
        """
        current_checkin["mood"] = mood
        logger.info(f"Mood set to: {mood}")
        # Publish check-in update to frontend
        try:
            if current_room:
                await current_room.local_participant.publish_data(
                    json.dumps({
                        "type": "checkin_update",
                        "checkin": current_checkin
                    }).encode(),
                    topic="wellness-checkin"
                )
        except Exception as e:
            logger.warning(f"Failed to publish checkin update: {e}")
        return f"I hear you're feeling {mood} today. Tell me, how's your energy level?"
    
    @function_tool
    async def set_energy(self, context: RunContext, energy: Annotated[str, "User's energy level"]):
        """Record the user's energy level.
        
        Args:
            energy: The user's current energy level (e.g., high, low, medium, drained, energized)
        """
        current_checkin["energy"] = energy
        logger.info(f"Energy set to: {energy}")
        # Publish check-in update to frontend
        try:
            if current_room:
                await current_room.local_participant.publish_data(
                    json.dumps({
                        "type": "checkin_update",
                        "checkin": current_checkin
                    }).encode(),
                    topic="wellness-checkin"
                )
        except Exception as e:
            logger.warning(f"Failed to publish checkin update: {e}")
        return f"Got it, your energy is {energy}. Is there anything stressing you out or on your mind right now?"
    
    @function_tool
    async def set_stress(self, context: RunContext, stress: Annotated[str, "What is stressing the user or their concerns"]):
        """Record any stress or concerns the user mentions.
        
        Args:
            stress: What's causing stress or concern for the user
        """
        current_checkin["stress"] = stress
        logger.info(f"Stress noted: {stress}")
        # Publish check-in update to frontend
        try:
            if current_room:
                await current_room.local_participant.publish_data(
                    json.dumps({
                        "type": "checkin_update",
                        "checkin": current_checkin
                    }).encode(),
                    topic="wellness-checkin"
                )
        except Exception as e:
            logger.warning(f"Failed to publish checkin update: {e}")
        return f"I understand. {stress} can definitely be challenging. What are 1-3 things you'd like to accomplish or focus on today?"
    
    @function_tool
    async def add_objective(self, context: RunContext, objective: Annotated[str, "A goal or intention for today"]):
        """Add an objective or intention for the day.
        
        Args:
            objective: Something the user wants to accomplish or focus on today
        """
        if objective not in current_checkin["objectives"]:
            current_checkin["objectives"].append(objective)
        logger.info(f"Added objective: {objective}. Total objectives: {len(current_checkin['objectives'])}")
        # Publish check-in update to frontend
        try:
            if current_room:
                await current_room.local_participant.publish_data(
                    json.dumps({
                        "type": "checkin_update",
                        "checkin": current_checkin
                    }).encode(),
                    topic="wellness-checkin"
                )
        except Exception as e:
            logger.warning(f"Failed to publish checkin update: {e}")
        
        if len(current_checkin["objectives"]) >= 3:
            return f"Great! So you have: {', '.join(current_checkin['objectives'])}. That sounds like a solid plan. Would you like me to recap everything?"
        else:
            return f"Got it - {objective}. Any other goals for today?"
    
    @function_tool
    async def add_note(self, context: RunContext, note: Annotated[str, "Additional notes or reflections from the user"]):
        """Add additional notes or reflections.
        
        Args:
            note: Any additional thoughts, reflections, or context the user wants to share
        """
        current_checkin["notes"] = note
        logger.info(f"Note added: {note}")
        # Publish check-in update to frontend
        try:
            if current_room:
                await current_room.local_participant.publish_data(
                    json.dumps({
                        "type": "checkin_update",
                        "checkin": current_checkin
                    }).encode(),
                    topic="wellness-checkin"
                )
        except Exception as e:
            logger.warning(f"Failed to publish checkin update: {e}")
        return f"Thanks for sharing that. I've noted it down."
    
    @function_tool
    async def complete_checkin(self, context: RunContext):
        """Complete the wellness check-in and save it to the log file."""
        try:
            # Load existing log
            if WELLNESS_LOG_FILE.exists():
                with open(WELLNESS_LOG_FILE, "r") as f:
                    log_data = json.load(f)
            else:
                log_data = {"check_ins": []}
            
            # Create check-in entry
            checkin_entry = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "time": datetime.now().strftime("%H:%M:%S"),
                "timestamp": datetime.now().isoformat(),
                "mood": current_checkin["mood"],
                "energy": current_checkin["energy"],
                "stress": current_checkin["stress"],
                "objectives": current_checkin["objectives"],
                "notes": current_checkin["notes"],
                "summary": f"Mood: {current_checkin['mood']}, Energy: {current_checkin['energy']}, Objectives: {len(current_checkin['objectives'])}"
            }
            
            # Add to log
            log_data["check_ins"].append(checkin_entry)
            
            # Save to file
            with open(WELLNESS_LOG_FILE, "w") as f:
                json.dump(log_data, f, indent=2)
            
            logger.info(f"Check-in saved: {checkin_entry}")
            
            # Publish final check-in to frontend
            try:
                if current_room:
                    await current_room.local_participant.publish_data(
                        json.dumps({
                            "type": "checkin_complete",
                            "checkin": current_checkin
                        }).encode(),
                        topic="wellness-checkin"
                    )
                    logger.info("Published checkin_complete to frontend")
            except Exception as e:
                logger.warning(f"Failed to publish checkin complete: {e}")
            
            # Create recap
            objectives_text = ", ".join(current_checkin["objectives"]) if current_checkin["objectives"] else "no specific objectives"
            recap = f"""Perfect! Let me recap:
            
Mood: {current_checkin['mood']}
Energy: {current_checkin['energy']}
{f"Stress: {current_checkin['stress']}" if current_checkin['stress'] else ""}
Objectives: {objectives_text}
            
Does this sound right? Remember, break things into small steps and be kind to yourself today."""
            
            return recap
        except Exception as e:
            logger.error(f"Error in complete_checkin: {e}", exc_info=True)
            return "I apologize, but I had trouble saving your check-in. However, I've noted everything you shared. Take care today!"


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    global current_room, current_checkin
    current_room = ctx.room
    
    # Reset check-in state when session starts
    current_checkin["mood"] = None
    current_checkin["energy"] = None
    current_checkin["stress"] = None
    current_checkin["objectives"] = []
    current_checkin["notes"] = None
    
    # Load previous check-ins for context
    previous_context = ""
    if WELLNESS_LOG_FILE.exists():
        try:
            with open(WELLNESS_LOG_FILE, "r") as f:
                log_data = json.load(f)
            if log_data.get("check_ins"):
                last_checkin = log_data["check_ins"][-1]
                previous_context = f"\n\nPREVIOUS CHECK-IN CONTEXT: Last time (on {last_checkin['date']}), the user mentioned: Mood was '{last_checkin['mood']}', Energy was '{last_checkin['energy']}'. Reference this naturally in your greeting to show continuity."
                logger.info(f"Loaded previous check-in context: {last_checkin['date']}")
        except Exception as e:
            logger.warning(f"Could not load previous check-ins: {e}")
    
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=deepgram.STT(model="nova-3"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=google.LLM(
                model="gemini-2.5-flash",
            ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=murf.TTS(
                voice="en-US-matthew", 
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    assistant = Assistant(previous_context=previous_context)
    
    await session.start(
        agent=assistant,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
