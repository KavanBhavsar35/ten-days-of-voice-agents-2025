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

# Order state structure
current_order = {
    "drinkType": None,
    "size": None,
    "milk": None,
    "extras": [],
    "name": None
}

# Store all orders during session
order_history = []

# Store room reference for publishing data
current_room = None


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a friendly barista at Murf's Coffee House, the finest coffee shop in town! 
            You are warm, enthusiastic, and passionate about coffee. The user is interacting with you via voice.
            
            Your job is to help customers place their coffee orders. For each order, you need to collect:
            1. Drink type (e.g., latte, cappuccino, americano, espresso, mocha, cold brew, frappuccino, etc.)
            2. Size (small, medium, or large)
            3. Milk preference (whole milk, skim milk, oat milk, almond milk, soy milk, or none)
            4. Extras (whipped cream, extra shot, vanilla syrup, caramel syrup, chocolate syrup, cinnamon, etc.) - optional
            5. Customer's name for the order
            
            IMPORTANT: When the customer provides multiple pieces of information in one sentence, extract ALL the details and use the appropriate tools for each piece of information in the SAME response. For example, if they say "I want a large latte with oat milk", immediately call set_drink_type, set_size, and set_milk tools.
            
            Ask friendly, clarifying questions for any MISSING information only.
            Once you have drink type, size, milk, and name, use the complete_order tool immediately.
            Extras are optional - if the customer says "no extras" or "nothing else", proceed to complete the order.
            
            Your responses should be concise, natural, and conversational without complex formatting or symbols.
            Be helpful and make suggestions if customers are unsure what to order!""",
        )

    @function_tool
    async def set_drink_type(self, context: RunContext, drink_type: Annotated[str, "The type of coffee drink (e.g., latte, cappuccino, americano, espresso, mocha, cold brew)"]):
        """Set the type of drink for the current order.
        
        Args:
            drink_type: The type of coffee drink the customer wants
        """
        current_order["drinkType"] = drink_type
        logger.info(f"Drink type set to: {drink_type}")
        # Publish order update to frontend
        try:
            if current_room:
                await current_room.local_participant.publish_data(
                    json.dumps({
                        "type": "order_update", 
                        "order": current_order,
                        "history": order_history
                    }).encode(),
                    topic="coffee-order"
                )
        except Exception as e:
            logger.warning(f"Failed to publish order update: {e}")
        return f"Got it! A {drink_type}. What size would you like - small, medium, or large?"
    
    @function_tool
    async def set_size(self, context: RunContext, size: Annotated[str, "The size of the drink: small, medium, or large"]):
        """Set the size for the current order.
        
        Args:
            size: The size of the drink (small, medium, or large)
        """
        current_order["size"] = size.lower()
        logger.info(f"Size set to: {size}")
        # Publish order update to frontend
        try:
            if current_room:
                await current_room.local_participant.publish_data(
                    json.dumps({
                        "type": "order_update", 
                        "order": current_order,
                        "history": order_history
                    }).encode(),
                    topic="coffee-order"
                )
        except Exception as e:
            logger.warning(f"Failed to publish order update: {e}")
        return f"Perfect! A {size} {current_order.get('drinkType', 'drink')}. What kind of milk would you like?"
    
    @function_tool
    async def set_milk(self, context: RunContext, milk: Annotated[str, "The type of milk: whole milk, skim milk, oat milk, almond milk, soy milk, or none"]):
        """Set the milk preference for the current order.
        
        Args:
            milk: The type of milk preference
        """
        current_order["milk"] = milk.lower()
        logger.info(f"Milk set to: {milk}")
        # Publish order update to frontend
        try:
            if current_room:
                await current_room.local_participant.publish_data(
                    json.dumps({
                        "type": "order_update", 
                        "order": current_order,
                        "history": order_history
                    }).encode(),
                    topic="coffee-order"
                )
        except Exception as e:
            logger.warning(f"Failed to publish order update: {e}")
        return f"Noted! {milk}. Would you like any extras like whipped cream, extra shot, or flavored syrups?"
    
    @function_tool
    async def add_extra(self, context: RunContext, extra: Annotated[str, "An extra item to add (e.g., whipped cream, extra shot, vanilla syrup, caramel syrup, chocolate syrup, cinnamon)"]):
        """Add an extra item to the current order.
        
        Args:
            extra: The extra item to add to the drink
        """
        if extra.lower() not in current_order["extras"]:
            current_order["extras"].append(extra.lower())
        logger.info(f"Added extra: {extra}. Current extras: {current_order['extras']}")
        # Publish order update to frontend
        try:
            if current_room:
                await current_room.local_participant.publish_data(
                    json.dumps({
                        "type": "order_update", 
                        "order": current_order,
                        "history": order_history
                    }).encode(),
                    topic="coffee-order"
                )
        except Exception as e:
            logger.warning(f"Failed to publish order update: {e}")
        return f"Added {extra}! Anything else you'd like to add?"
    
    @function_tool
    async def set_customer_name(self, context: RunContext, name: Annotated[str, "The customer's name for the order"]):
        """Set the customer's name for the current order.
        
        Args:
            name: The customer's name
        """
        current_order["name"] = name
        logger.info(f"Customer name set to: {name}")
        # Publish order update to frontend
        try:
            if current_room:
                await current_room.local_participant.publish_data(
                    json.dumps({
                        "type": "order_update", 
                        "order": current_order,
                        "history": order_history
                    }).encode(),
                    topic="coffee-order"
                )
        except Exception as e:
            logger.warning(f"Failed to publish order update: {e}")
        return f"Great! I have your name as {name}. Let me complete your order now."
    
    @function_tool
    async def complete_order(self, context: RunContext):
        """Complete and save the order to a JSON file. Use this when all order details are collected."""
        try:
            # Validate that all required fields are filled
            missing_fields = []
            if not current_order["drinkType"]:
                missing_fields.append("drink type")
            if not current_order["size"]:
                missing_fields.append("size")
            if not current_order["milk"]:
                missing_fields.append("milk preference")
            if not current_order["name"]:
                missing_fields.append("customer name")
            
            if missing_fields:
                return f"I still need to know: {', '.join(missing_fields)}. Can you provide that information?"
            
            # Create orders directory if it doesn't exist
            orders_dir = Path("orders")
            orders_dir.mkdir(exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = orders_dir / f"order_{current_order['name']}_{timestamp}.json"
            
            # Save order to JSON file
            order_data = {
                **current_order,
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }
            
            with open(filename, "w") as f:
                json.dump(order_data, f, indent=2)
            
            logger.info(f"Order saved to {filename}: {order_data}")
            
            # Save completed order to history
            order_history.append(order_data.copy())
            
            # Publish final order to frontend with history
            try:
                if current_room:
                    await current_room.local_participant.publish_data(
                        json.dumps({
                            "type": "order_complete", 
                            "order": order_data,
                            "history": order_history
                        }).encode(),
                        topic="coffee-order"
                    )
                    logger.info("Published order_complete to frontend with history")
            except Exception as e:
                logger.warning(f"Failed to publish order complete: {e}")
            
            # Create order summary
            extras_text = ", ".join(current_order["extras"]) if current_order["extras"] else "no extras"
            summary = f"""Perfect! I've placed your order, {current_order['name']}!
            
Your order: {current_order['size']} {current_order['drinkType']} with {current_order['milk']}, {extras_text}.
            
Your order has been saved and will be ready shortly. Thank you for choosing Murf's Coffee House!"""
            
            # Note: Current order state persists until session ends - no reset here
            # This allows the beverage visualization to remain visible
            # Order history is maintained throughout the session
            
            return summary
        except Exception as e:
            logger.error(f"Error in complete_order: {e}", exc_info=True)
            return f"I apologize, but I encountered an error completing your order. However, I have all your details saved: {order_state['size']} {order_state['drinkType']} with {order_state['milk']} for {order_state['name']}. Let me try again!"


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    global current_room, current_order, order_history
    current_room = ctx.room
    
    # Reset order state when session starts
    current_order["drinkType"] = None
    current_order["size"] = None
    current_order["milk"] = None
    current_order["extras"] = []
    current_order["name"] = None
    order_history = []  # Reset order history for new session
    
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
    await session.start(
        agent=Assistant(),
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
