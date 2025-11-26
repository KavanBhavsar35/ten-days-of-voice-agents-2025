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
from livekit.plugins import silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel
import murf_tts

logger = logging.getLogger("sdr_agent")

load_dotenv(".env.local")

# Load company FAQ
COMPANY_FAQ_FILE = Path("../shared-data/day5_company_faq.json")
company_data = {}
if COMPANY_FAQ_FILE.exists():
    with open(COMPANY_FAQ_FILE, "r") as f:
        company_data = json.load(f)
        logger.info(f"Loaded company data for {company_data.get('company', {}).get('name', 'Unknown')}")
else:
    logger.warning(f"Company FAQ file not found: {COMPANY_FAQ_FILE}")

# Lead storage
LEADS_FILE = Path("../shared-data/leads.json")
lead_data = {
    "name": None,
    "company": None,
    "email": None,
    "role": None,
    "use_case": None,
    "team_size": None,
    "timeline": None,
    "questions_asked": [],
    "conversation_summary": None,
    "timestamp": None
}


def save_lead():
    """Save the current lead data to JSON file"""
    lead_data["timestamp"] = datetime.now().isoformat()
    
    # Load existing leads
    leads = []
    if LEADS_FILE.exists():
        with open(LEADS_FILE, "r") as f:
            try:
                leads = json.load(f)
            except:
                leads = []
    
    # Add new lead
    leads.append(lead_data.copy())
    
    # Save back
    with open(LEADS_FILE, "w") as f:
        json.dump(leads, f, indent=2)
    
    logger.info(f"Lead saved: {lead_data.get('name')} from {lead_data.get('company')}")


def search_faq(query: str) -> str:
    """Simple keyword search in FAQ"""
    query_lower = query.lower()
    
    # Search in FAQ
    for faq_item in company_data.get("faq", []):
        question = faq_item["question"].lower()
        answer = faq_item["answer"]
        
        # Simple keyword matching
        if any(word in question for word in query_lower.split()):
            return answer
    
    # Search in products
    for product in company_data.get("products", []):
        if any(word in product["name"].lower() for word in query_lower.split()):
            return f"{product['name']}: {product['description']} Best for: {product['use_case']}"
    
    return None


class SDRAgent(Agent):
    def __init__(self) -> None:
        company_name = company_data.get("company", {}).get("name", "our company")
        company_desc = company_data.get("company", {}).get("description", "")
        
        super().__init__(
            instructions=f"""You are a friendly and professional Sales Development Representative (SDR) for {company_name}.

COMPANY OVERVIEW:
{company_desc}

YOUR ROLE:
1. Greet visitors warmly and professionally
2. Ask what brought them here and what they're working on
3. Understand their needs and pain points
4. Answer questions about our products, pricing, and services using the FAQ
5. Naturally collect lead information during the conversation
6. When they're done, summarize the conversation and thank them

LEAD INFORMATION TO COLLECT (ask naturally, don't interrogate):
- Name
- Company name
- Email address
- Their role/position
- What they want to use our product for (use case)
- Team size
- Timeline (now / soon / later)

CONVERSATION STYLE:
- Be warm, friendly, and consultative (not pushy)
- Listen actively and ask follow-up questions
- Focus on understanding their needs first
- Use the tools to answer specific questions
- Keep responses concise and conversational
- When you don't know something, be honest and offer to find out

IMPORTANT:
- Use the search_faq tool when they ask about products, pricing, or features
- Use the collect_lead_info tool to store information as you learn it
- Use the end_call_summary tool when they say they're done or ready to leave
- Don't make up information - only use what's in the FAQ""",
        )
    
    @function_tool
    async def search_faq(self, context: RunContext, query: Annotated[str, "The user's question about the company, product, or pricing"]):
        """Search the company FAQ for answers to user questions.
        
        Args:
            query: The user's question
        """
        logger.info(f"Searching FAQ for: {query}")
        
        # Track questions asked
        if query not in lead_data["questions_asked"]:
            lead_data["questions_asked"].append(query)
        
        answer = search_faq(query)
        
        if answer:
            return f"Based on our FAQ: {answer}"
        else:
            # Return general company info
            return f"I don't have specific information about that in my FAQ. Let me tell you generally: {company_data.get('company', {}).get('description', 'We provide payment solutions for businesses.')}"
    
    @function_tool
    async def collect_lead_info(
        self, 
        context: RunContext,
        field: Annotated[str, "The field name: 'name', 'company', 'email', 'role', 'use_case', 'team_size', or 'timeline'"],
        value: Annotated[str, "The value to store"]
    ):
        """Store lead information as it's collected during the conversation.
        
        Args:
            field: Which field to update
            value: The value to store
        """
        if field in lead_data:
            lead_data[field] = value
            logger.info(f"Collected lead info: {field} = {value}")
            return f"Got it, I've noted that down."
        else:
            return "I couldn't store that information."
    
    @function_tool
    async def end_call_summary(self, context: RunContext, summary: Annotated[str, "A brief summary of the conversation and the lead's needs"]):
        """End the call and provide a summary of the lead.
        
        Args:
            summary: Brief summary of the conversation
        """
        lead_data["conversation_summary"] = summary
        save_lead()
        
        # Create verbal summary
        name = lead_data.get("name", "there")
        company = lead_data.get("company", "your company")
        use_case = lead_data.get("use_case", "your needs")
        timeline = lead_data.get("timeline", "soon")
        
        return f"Thank you so much for your time, {name}! I've captured all the details about {company} and your interest in using our solution for {use_case}. Based on our conversation, it sounds like you're looking to move forward {timeline}. I'll make sure our team follows up with you shortly. Have a great day!"


def prewarm(proc: JobProcess):
    """Prewarm the VAD model"""
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    """Main entrypoint for the SDR agent"""
    
    # Reset lead data for new session
    global lead_data
    lead_data = {
        "name": None,
        "company": None,
        "email": None,
        "role": None,
        "use_case": None,
        "team_size": None,
        "timeline": None,
        "questions_asked": [],
        "conversation_summary": None,
        "timestamp": None
    }
    
    logger.info(f"Starting SDR agent for room: {ctx.room.name}")
    
    # Create session with Murf TTS
    session = AgentSession(
        stt=deepgram.STT(
            model="nova-3",
            language="en-US",
        ),
        llm=google.LLM(
            model="gemini-2.5-flash",
            temperature=0.7,
        ),
        tts=murf_tts.TTS(
            voice="en-US-ryan",
            style="Conversational",
            tokenizer=tokenize.basic.SentenceTokenizer(
                min_sentence_len=5,
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

    # Start the session with SDR agent
    sdr = SDRAgent()
    
    await session.start(
        agent=sdr,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
