import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional
import uuid

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

logger = logging.getLogger("grocery_agent")

load_dotenv(".env.local")

# Load catalog
CATALOG_FILE = Path("../shared-data/catalog.json")
catalog_data = {}
if CATALOG_FILE.exists():
    with open(CATALOG_FILE, "r") as f:
        catalog_data = json.load(f)
        logger.info(f"Loaded catalog with {len(catalog_data.get('items', []))} items")
else:
    logger.warning(f"Catalog file not found: {CATALOG_FILE}")

# Orders directory
ORDERS_DIR = Path("../shared-data/orders")
ORDERS_DIR.mkdir(exist_ok=True)

# Session state
cart = []
customer_name = None


def find_item_by_name(item_name: str):
    """Find an item in catalog by name (fuzzy match)"""
    item_name_lower = item_name.lower().strip()
    
    # Exact match first
    for item in catalog_data.get("items", []):
        if item["name"].lower() == item_name_lower:
            return item
    
    # Partial match
    for item in catalog_data.get("items", []):
        if item_name_lower in item["name"].lower() or item["name"].lower() in item_name_lower:
            return item
    
    return None


def find_item_by_id(item_id: str):
    """Find an item by ID"""
    for item in catalog_data.get("items", []):
        if item["id"] == item_id:
            return item
    return None


def get_recipe_items(recipe_name: str):
    """Get items for a recipe"""
    recipe_name_lower = recipe_name.lower().strip()
    recipes = catalog_data.get("recipes", {})
    
    # Exact match
    if recipe_name_lower in recipes:
        return recipes[recipe_name_lower]
    
    # Partial match
    for recipe, items in recipes.items():
        if recipe_name_lower in recipe or recipe in recipe_name_lower:
            return items
    
    return None


def calculate_cart_total():
    """Calculate total price of items in cart"""
    total = 0.0
    for cart_item in cart:
        total += cart_item["price"] * cart_item["quantity"]
    return round(total, 0)  # INR doesn't use decimals typically


def save_order():
    """Save the current order to a JSON file"""
    if not cart:
        return None
    
    order_id = str(uuid.uuid4())[:8]
    order = {
        "order_id": order_id,
        "customer_name": customer_name or "Guest",
        "timestamp": datetime.now().isoformat(),
        "status": "received",
        "items": cart.copy(),
        "total": calculate_cart_total(),
        "delivery_address": "123 Main St (Demo)"
    }
    
    # Save to individual order file
    order_file = ORDERS_DIR / f"order_{order_id}.json"
    with open(order_file, "w") as f:
        json.dump(order, f, indent=2)
    
    # Also append to order history
    history_file = ORDERS_DIR / "order_history.json"
    history = []
    if history_file.exists():
        with open(history_file, "r") as f:
            try:
                history = json.load(f)
            except:
                history = []
    
    history.append(order)
    
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"Order {order_id} saved with {len(cart)} items, total: ${calculate_cart_total()}")
    
    return order


class GroceryOrderingAgent(Agent):
    def __init__(self) -> None:
        store_name = catalog_data.get("store_name", "QuickMart")
        categories = ", ".join(catalog_data.get("categories", []))
        
        super().__init__(
            instructions=f"""You are a friendly and helpful shopping assistant for {store_name}, a food and grocery delivery service.

YOUR ROLE:
Help customers order groceries and food items through natural conversation. You can:
1. Add items to their cart
2. Handle recipe/meal requests (like "ingredients for pasta")
3. Show what's in their cart
4. Update quantities or remove items
5. Place the final order

CONVERSATION FLOW:

1. GREETING:
   - "Hi! Welcome to {store_name}! I can help you order groceries and food."
   - "What would you like to order today?"
   - Optionally ask for their name

2. TAKING ORDERS:
   - Listen for specific items: "I need milk", "Add bread to my cart"
   - Listen for recipe requests: "I want to make pasta", "Get me ingredients for a sandwich"
   - For specific items, use add_to_cart directly (skip search_item for speed)
   - Use the add_recipe_items tool for recipe/meal requests
   - Always confirm what you're adding briefly

3. CART MANAGEMENT:
   - Use add_to_cart to add items with quantities
   - Use remove_from_cart to remove items
   - Use update_quantity to change amounts
   - Use show_cart to display current cart
   - Always confirm changes verbally

4. CLARIFICATIONS:
   - If multiple items match, ask which one they want
   - If quantity isn't specified, assume 1
   - If they say "bread", ask if they want whole wheat or white

5. PLACING ORDER:
   - When they say "that's all", "place my order", "checkout", or "I'm done"
   - Use show_cart to review the order
   - Confirm the total
   - Use place_order to finalize
   - Thank them and confirm order is placed

IMPORTANT RULES:
- Be conversational and friendly, not robotic
- Confirm each addition: "I've added 2 litres of milk to your cart"
- For recipes, explain what you're adding: "For pasta, I'm adding spaghetti, tomato sauce, olive oil, and garlic"
- If an item isn't found, suggest similar items or ask them to try another name
- Keep responses natural and helpful
- Always use the tools - don't make up prices or availability

TONE:
- Friendly and upbeat
- Patient with clarifications
- Excited about food!
- Professional but warm

AVAILABLE CATEGORIES:
{categories}

Remember: You're here to make grocery shopping easy and enjoyable!""",
        )
    
    @function_tool
    async def search_item(
        self, 
        context: RunContext,
        item_name: Annotated[str, "The name of the item to search for"]
    ):
        """Search for an item in the catalog by name.
        
        Args:
            item_name: The name of the item to search for
        """
        logger.info(f"Searching for item: {item_name}")
        
        item = find_item_by_name(item_name)
        
        if item:
            return f"Found: {item['name']} - ₹{item['price']} per {item['unit']} ({item['brand']}). Available to add to cart."
        else:
            # Suggest items from same category or similar
            suggestions = []
            for cat_item in catalog_data.get("items", [])[:5]:
                suggestions.append(cat_item["name"])
            
            return f"Sorry, I couldn't find '{item_name}'. Some items we have: {', '.join(suggestions)}. Try searching for one of these?"
    
    @function_tool
    async def add_to_cart(
        self, 
        context: RunContext,
        item_name: Annotated[str, "The name of the item to add"],
        quantity: Annotated[int, "The quantity to add"] = 1
    ):
        """Add an item to the shopping cart.
        
        Args:
            item_name: The name of the item
            quantity: How many to add (default 1)
        """
        logger.info(f"Adding to cart: {quantity}x {item_name}")
        
        item = find_item_by_name(item_name)
        
        if not item:
            return f"Sorry, I couldn't find '{item_name}' in our catalog. Try searching for it first?"
        
        # Check if item already in cart
        for cart_item in cart:
            if cart_item["id"] == item["id"]:
                cart_item["quantity"] += quantity
                logger.info(f"Updated quantity for {item['name']}: {cart_item['quantity']}")
                return f"Updated! You now have {cart_item['quantity']} {item['unit']}(s) of {item['name']} in your cart."
        
        # Add new item to cart
        cart_item = {
            "id": item["id"],
            "name": item["name"],
            "price": item["price"],
            "unit": item["unit"],
            "quantity": quantity
        }
        cart.append(cart_item)
        
        logger.info(f"Added {quantity}x {item['name']} to cart")
        return f"Added {quantity} {item['unit']}(s) of {item['name']} to your cart (₹{item['price']} each)."
    
    @function_tool
    async def add_recipe_items(
        self, 
        context: RunContext,
        recipe_name: Annotated[str, "The name of the recipe or meal (e.g., 'pasta', 'sandwich', 'breakfast')"]
    ):
        """Add all ingredients needed for a recipe or meal to the cart.
        
        Args:
            recipe_name: The recipe or meal name
        """
        logger.info(f"Adding recipe items for: {recipe_name}")
        
        item_ids = get_recipe_items(recipe_name)
        
        if not item_ids:
            return f"I don't have a recipe for '{recipe_name}'. Try asking for specific items instead, or try: pasta, sandwich, breakfast, omelet, or salad."
        
        added_items = []
        for item_id in item_ids:
            item = find_item_by_id(item_id)
            if item:
                # Check if already in cart
                found = False
                for cart_item in cart:
                    if cart_item["id"] == item["id"]:
                        cart_item["quantity"] += 1
                        found = True
                        break
                
                if not found:
                    cart.append({
                        "id": item["id"],
                        "name": item["name"],
                        "price": item["price"],
                        "unit": item["unit"],
                        "quantity": 1
                    })
                
                added_items.append(item["name"])
        
        logger.info(f"Added {len(added_items)} items for {recipe_name}")
        items_list = ", ".join(added_items)
        return f"Great! For {recipe_name}, I've added: {items_list} to your cart."
    
    @function_tool
    async def remove_from_cart(
        self, 
        context: RunContext,
        item_name: Annotated[str, "The name of the item to remove"]
    ):
        """Remove an item from the shopping cart.
        
        Args:
            item_name: The name of the item to remove
        """
        logger.info(f"Removing from cart: {item_name}")
        
        item_name_lower = item_name.lower()
        
        for i, cart_item in enumerate(cart):
            if item_name_lower in cart_item["name"].lower():
                removed = cart.pop(i)
                logger.info(f"Removed {removed['name']} from cart")
                return f"Removed {removed['name']} from your cart."
        
        return f"I couldn't find '{item_name}' in your cart."
    
    @function_tool
    async def update_quantity(
        self, 
        context: RunContext,
        item_name: Annotated[str, "The name of the item to update"],
        new_quantity: Annotated[int, "The new quantity"]
    ):
        """Update the quantity of an item in the cart.
        
        Args:
            item_name: The name of the item
            new_quantity: The new quantity
        """
        logger.info(f"Updating quantity: {item_name} to {new_quantity}")
        
        item_name_lower = item_name.lower()
        
        for cart_item in cart:
            if item_name_lower in cart_item["name"].lower():
                old_qty = cart_item["quantity"]
                cart_item["quantity"] = new_quantity
                logger.info(f"Updated {cart_item['name']} quantity: {old_qty} -> {new_quantity}")
                return f"Updated {cart_item['name']} quantity to {new_quantity}."
        
        return f"I couldn't find '{item_name}' in your cart."
    
    @function_tool
    async def show_cart(self, context: RunContext):
        """Show all items currently in the shopping cart with quantities and total.
        """
        logger.info("Showing cart")
        
        if not cart:
            return "Your cart is empty. What would you like to order?"
        
        cart_summary = "Here's what's in your cart:\n"
        for cart_item in cart:
            item_total = cart_item["price"] * cart_item["quantity"]
            cart_summary += f"- {cart_item['quantity']} {cart_item['unit']}(s) of {cart_item['name']} (₹{cart_item['price']} each = ₹{item_total})\n"
        
        total = calculate_cart_total()
        cart_summary += f"\nTotal: ₹{total}"
        
        return cart_summary
    
    @function_tool
    async def place_order(
        self, 
        context: RunContext,
        customer_name_input: Annotated[Optional[str], "Customer's name (optional)"] = None
    ):
        """Place the final order and save it.
        
        Args:
            customer_name_input: Optional customer name
        """
        global customer_name
        
        if customer_name_input:
            customer_name = customer_name_input
        
        logger.info(f"Placing order for {customer_name or 'Guest'}")
        
        if not cart:
            return "Your cart is empty! Add some items before placing an order."
        
        order = save_order()
        
        if order:
            order_summary = f"Order placed successfully! Order ID: {order['order_id']}\n\n"
            order_summary += f"Items ordered:\n"
            for item in order["items"]:
                order_summary += f"- {item['quantity']}x {item['name']}\n"
            order_summary += f"\nTotal: ₹{order['total']}\n"
            order_summary += f"Status: {order['status']}\n"
            order_summary += f"Estimated delivery: 30-45 minutes\n\n"
            order_summary += "Thank you for shopping with us!"
            
            # Clear cart after order
            cart.clear()
            
            return order_summary
        else:
            return "Sorry, there was an error placing your order. Please try again."
    
    @function_tool
    async def set_customer_name(
        self, 
        context: RunContext,
        name: Annotated[str, "The customer's name"]
    ):
        """Set the customer's name for the order.
        
        Args:
            name: Customer's name
        """
        global customer_name
        customer_name = name
        logger.info(f"Customer name set to: {name}")
        return f"Great! I'll put this order under {name}."


def prewarm(proc: JobProcess):
    """Prewarm the VAD model"""
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    """Main entrypoint for the grocery ordering agent"""
    
    # Reset session state for new call
    global cart, customer_name
    cart = []
    customer_name = None
    
    logger.info(f"Starting grocery ordering session for room: {ctx.room.name}")
    
    # Create session with Murf TTS - Optimized for speed
    session = AgentSession(
        stt=deepgram.STT(
            model="nova-3",
            language="en-US",
        ),
        llm=google.LLM(
            model="gemini-2.5-flash",
            temperature=0.6,  # Lower for faster, more consistent responses
        ),
        tts=murf_tts.TTS(
            voice="en-US-ryan",
            style="Conversational",
            tokenizer=tokenize.basic.SentenceTokenizer(
                min_sentence_len=50,  # Much larger chunks = fewer pauses
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
        
        # Log final cart state
        if cart:
            logger.info(f"Session ended with {len(cart)} items in cart (not ordered)")
        else:
            logger.info("Session ended with empty cart")

    ctx.add_shutdown_callback(log_usage)

    # Start the session with grocery agent
    grocery_agent = GroceryOrderingAgent()
    
    await session.start(
        agent=grocery_agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
