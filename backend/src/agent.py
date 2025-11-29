import logging
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

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
from livekit.plugins import silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel
import murf_tts

logger = logging.getLogger("game_master")

load_dotenv(".env.local")

# Load game state
GAME_STATE_FILE = Path("../shared-data/game_state.json")
game_state = {}
if GAME_STATE_FILE.exists():
    with open(GAME_STATE_FILE, "r", encoding="utf-8") as f:
        game_state = json.load(f)
        logger.info(f"Loaded game state for {game_state.get('player', {}).get('name', 'Adventurer')}")
else:
    logger.warning(f"Game state file not found: {GAME_STATE_FILE}")


def save_game_state():
    """Save the current game state to JSON"""
    with open(GAME_STATE_FILE, "w") as f:
        json.dump(game_state, f, indent=2)
    logger.info("Game state saved")


def roll_dice(sides=20, modifier=0):
    """Roll a dice with optional modifier"""
    roll = random.randint(1, sides)
    total = roll + modifier
    logger.info(f"Dice roll: d{sides} = {roll} + {modifier} = {total}")
    return roll, total


def get_stat_modifier(stat_value):
    """Convert stat value to D&D-style modifier"""
    return (stat_value - 10) // 2


class GameMasterAgent(Agent):
    def __init__(self) -> None:
        player_name = game_state.get("player", {}).get("name", "Adventurer")
        player_class = game_state.get("player", {}).get("class", "Warrior")
        location = game_state.get("current_location", {}).get("name", "Unknown")
        
        super().__init__(
            instructions=f"""You are an epic Game Master running a fantasy D&D-style adventure!

SETTING: High Fantasy World
You are guiding {player_name}, a brave {player_class}, through an immersive adventure filled with danger, mystery, and glory!

CURRENT SITUATION:
- Player: {player_name} the {player_class}
- Location: {location}
- Your role: Describe vivid scenes, create tension, and guide the story

YOUR GM STYLE:
1. IMMERSIVE STORYTELLING
   - Paint vivid scenes with sensory details
   - Create atmosphere and tension
   - Use dramatic pauses and emphasis
   - Make NPCs memorable with distinct personalities

2. PLAYER AGENCY
   - Always end with "What do you do?" or similar prompt
   - Present 2-3 clear options when helpful
   - Let player make meaningful choices
   - React dynamically to unexpected actions

3. PACING
   - Keep descriptions concise but evocative (2-3 sentences)
   - Balance action, exploration, and roleplay
   - Build tension gradually
   - Reward creativity and bold choices

4. MECHANICS
   - Use the tools to track stats, inventory, and events
   - Call for dice rolls during risky actions
   - Apply stat modifiers to outcomes
   - Update game state after important events

5. TONE
   - Epic and dramatic, but not overly serious
   - Celebrate player victories
   - Make failures interesting, not punishing
   - Keep it fun and engaging!

IMPORTANT RULES:
- ALWAYS use tools to check/update game state
- Roll dice for combat, skill checks, and risky actions
- Track HP, inventory, and quest progress
- Remember past events and NPC interactions
- Keep responses under 50 words unless describing a major scene
- End EVERY response with a question or prompt for action

EXAMPLE OPENING:
"Thunder crashes as you stand before the Crimson Citadel. Its blood-red towers pierce storm clouds crackling with dark magic. You grip your Shadowfang Blade. Lord Malachar awaits within. This ends tonight. What do you do?"

MAKE IT EPIC:
- Use vivid, cinematic descriptions
- Create tension and urgency
- Make every choice feel important
- Reward bold actions
- Punish recklessness but keep it fun
- Use emojis for dramatic effect (⚔️💀🔥⚡)

Remember: You're creating a LEGENDARY adventure! Make every moment count!""",
        )
    
    @function_tool
    async def get_player_stats(self, context: RunContext):
        """Get the player's current stats, HP, and inventory.
        
        Returns player character information.
        """
        player = game_state.get("player", {})
        stats = player.get("stats", {})
        inventory = player.get("inventory", [])
        
        info = f"""Player: {player.get('name')} the {player.get('class')}
Level {player.get('level')} | HP: {player.get('hp')}/{player.get('max_hp')}
STR: {stats.get('strength')} | INT: {stats.get('intelligence')} | DEX: {stats.get('dexterity')}
Inventory: {', '.join(inventory)}
Gold: {player.get('gold')} coins"""
        
        logger.info(f"Player stats retrieved: {player.get('name')}")
        return info
    
    @function_tool
    async def roll_check(
        self,
        context: RunContext,
        check_type: Annotated[str, "Type of check: strength, intelligence, dexterity, charisma, or luck"],
        difficulty: Annotated[int, "Difficulty (5=easy, 10=medium, 15=hard, 20=very hard)"] = 10
    ):
        """Roll a skill check with player's stat modifier.
        
        Args:
            check_type: Which stat to use
            difficulty: Target number to beat
        """
        stats = game_state.get("player", {}).get("stats", {})
        stat_value = stats.get(check_type.lower(), 10)
        modifier = get_stat_modifier(stat_value)
        
        roll, total = roll_dice(20, modifier)
        success = total >= difficulty
        
        result = f"🎲 {check_type.upper()} Check: Rolled {roll} + {modifier} = {total}"
        if success:
            result += f" ✅ SUCCESS! (needed {difficulty})"
        else:
            result += f" ❌ FAILED (needed {difficulty})"
        
        logger.info(f"Check: {check_type} vs DC{difficulty} = {total} ({'success' if success else 'fail'})")
        return result
    
    @function_tool
    async def update_hp(
        self,
        context: RunContext,
        change: Annotated[int, "HP change (positive for healing, negative for damage)"],
        reason: Annotated[str, "Reason for HP change"]
    ):
        """Update player's HP.
        
        Args:
            change: Amount to change HP by
            reason: Why HP changed
        """
        player = game_state.get("player", {})
        old_hp = player.get("hp", 100)
        player["hp"] = max(0, min(player.get("max_hp", 100), old_hp + change))
        
        save_game_state()
        
        if change > 0:
            result = f"💚 Healed {change} HP! Now at {player['hp']}/{player['max_hp']} HP. ({reason})"
        else:
            result = f"💔 Took {abs(change)} damage! Now at {player['hp']}/{player['max_hp']} HP. ({reason})"
            if player["hp"] == 0:
                result += " ⚠️ CRITICAL CONDITION!"
        
        logger.info(f"HP updated: {old_hp} → {player['hp']} ({reason})")
        return result
    
    @function_tool
    async def update_inventory(
        self,
        context: RunContext,
        action: Annotated[str, "Action: 'add' or 'remove'"],
        item: Annotated[str, "Item name"]
    ):
        """Add or remove items from player inventory.
        
        Args:
            action: 'add' or 'remove'
            item: Item name
        """
        inventory = game_state.get("player", {}).get("inventory", [])
        
        if action == "add":
            inventory.append(item)
            result = f"📦 Added {item} to inventory!"
            logger.info(f"Added item: {item}")
        elif action == "remove":
            if item in inventory:
                inventory.remove(item)
                result = f"📤 Removed {item} from inventory."
                logger.info(f"Removed item: {item}")
            else:
                result = f"❌ {item} not in inventory."
        else:
            result = "Invalid action. Use 'add' or 'remove'."
        
        save_game_state()
        return result
    
    @function_tool
    async def update_location(
        self,
        context: RunContext,
        location_name: Annotated[str, "New location name"],
        description: Annotated[str, "Location description"]
    ):
        """Update player's current location.
        
        Args:
            location_name: Name of new location
            description: Description of the location
        """
        game_state["current_location"] = {
            "name": location_name,
            "description": description,
            "available_paths": []
        }
        
        save_game_state()
        
        logger.info(f"Location changed to: {location_name}")
        return f"📍 Arrived at: {location_name}"
    
    @function_tool
    async def record_event(
        self,
        context: RunContext,
        event_description: Annotated[str, "What happened"]
    ):
        """Record an important event in the game history.
        
        Args:
            event_description: Description of the event
        """
        events = game_state.get("events", [])
        events.append({
            "description": event_description,
            "timestamp": datetime.now().isoformat()
        })
        game_state["events"] = events
        
        save_game_state()
        
        logger.info(f"Event recorded: {event_description}")
        return f"📝 Recorded: {event_description}"
    
    @function_tool
    async def update_quest(
        self,
        context: RunContext,
        quest_id: Annotated[str, "Quest identifier"],
        status: Annotated[str, "Quest status: available, active, completed, or failed"]
    ):
        """Update quest status.
        
        Args:
            quest_id: Quest ID
            status: New status
        """
        quests = game_state.get("quests", [])
        for quest in quests:
            if quest.get("id") == quest_id:
                quest["status"] = status
                save_game_state()
                
                if status == "completed":
                    result = f"🎉 Quest Completed: {quest.get('title')}!"
                elif status == "active":
                    result = f"⚔️ Quest Active: {quest.get('title')}"
                else:
                    result = f"📜 Quest {status}: {quest.get('title')}"
                
                logger.info(f"Quest updated: {quest_id} → {status}")
                return result
        
        return f"Quest {quest_id} not found."


def prewarm(proc: JobProcess):
    """Prewarm the VAD model"""
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    """Main entrypoint for the Game Master agent"""
    
    logger.info(f"Starting Game Master session for room: {ctx.room.name}")
    
    # Create session with Murf TTS
    session = AgentSession(
        stt=deepgram.STT(
            model="nova-3",
            language="en-US",
        ),
        llm=google.LLM(
            model="gemini-2.5-flash",
            temperature=0.8,  # Higher for creative storytelling
        ),
        tts=murf_tts.TTS(
            voice="en-US-ryan",
            style="Narration",  # Dramatic narration style
            tokenizer=tokenize.basic.SentenceTokenizer(
                min_sentence_len=30,  # Longer for storytelling
            ),
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
    )
    
    # Metrics collection
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the session with Game Master
    gm = GameMasterAgent()
    
    await session.start(
        agent=gm,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
