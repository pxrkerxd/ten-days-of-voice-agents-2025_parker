import os
import json
import logging
from typing import Dict, Any, List
from pathlib import Path
import asyncio
import uuid
from datetime import datetime

from dotenv import load_dotenv

# LiveKit Agent imports
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
    tokenize,
)
# LiveKit Plugin imports
from livekit.plugins import google, murf, deepgram, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(".env.local")
logger = logging.getLogger("grocery.agent")

# --- Configuration & File Paths ---
# Calculates the path to the catalog.json based on your directory structure
CATALOG_PATH = Path(__file__).parent.parent.joinpath('DAY-7', 'catalog.json')
ORDERS_DIR = Path(__file__).parent.parent.joinpath('DAY-7', 'orders')
ORDERS_DIR.mkdir(exist_ok=True) # Ensure the 'orders' directory exists

# --- Grocery Agent Logic Class ---

class GroceryAgentLogic:
    """Manages the catalog, cart state, and order persistence."""
    def __init__(self):
        self.catalog = self._load_catalog()
        self.cart = {"items": [], "subtotal": 0.00}
        self.recipes = self._get_recipe_map()
        logger.info(f"Catalog loaded successfully with {len(self.catalog)} items.")

    def _load_catalog(self) -> Dict[str, Any]:
        """Loads and flattens the catalog.json."""
        try:
            with open(CATALOG_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            flat_catalog = {}
            for category, items in data.items():
                for item in items:
                    # Use lowercased name for case-insensitive matching
                    flat_catalog[item['name'].lower()] = item
            return flat_catalog
        except FileNotFoundError:
            logger.error(f"FATAL: Catalog file not found at {CATALOG_PATH}")
            return {}
        except json.JSONDecodeError:
            logger.exception("FATAL: Invalid JSON format in catalog.json")
            return {}

    def _get_recipe_map(self) -> Dict[str, List[Dict[str, Any]]]:
        """Hard-coded recipe map for intelligent ordering."""
        return {
            "peanut butter sandwich": [
                {"name": "whole wheat bread", "quantity": 1}, 
                {"name": "peanut butter (crunchy)", "quantity": 1}
            ],
            "simple pasta dinner": [
                {"name": "dried spaghetti pasta", "quantity": 1}, 
                {"name": "tomato pasta sauce (marinara)", "quantity": 1}, 
                {"name": "tomato", "quantity": 0.5} # 0.5 kg Tomatoes
            ],
            "egg curry": [
                {"name": "white eggs (pack of 12)", "quantity": 1}, 
                {"name": "onion", "quantity": 0.5},
                {"name": "turmeric powder (haldi)", "quantity": 1}
            ]
        }

    def _update_cart_total(self):
        """Recalculates the cart subtotal."""
        self.cart['subtotal'] = round(sum(item['line_total'] for item in self.cart['items']), 2)

    def add_item_to_cart(self, item_name: str, quantity: float) -> str:
        """Adds a single item to the cart or updates quantity."""
        item_name_lower = item_name.lower()
        catalog_item = self.catalog.get(item_name_lower)
        
        if not catalog_item:
            # Check for partial match (optional, but helpful)
            for name, item in self.catalog.items():
                if item_name_lower in name:
                    return f"❌ Did you mean {item['name']}? Please specify the exact item name."
            return f"❌ Sorry, I couldn't find '{item_name}'. Can you be more specific? For example, say the brand or size."

        price_per_unit = catalog_item['price']
        line_total = round(price_per_unit * quantity, 2)
        
        # Check if item is already in cart
        for item in self.cart['items']:
            if item['name'].lower() == item_name_lower:
                item['quantity_ordered'] += quantity
                item['line_total'] = round(item['price_per_unit'] * item['quantity_ordered'], 2)
                self._update_cart_total()
                return f"✅ Updated cart. Added {quantity} more {catalog_item['unit']} of {item_name}. You now have {item['quantity_ordered']} in total. Your subtotal is ₹{self.cart['subtotal']:.2f}."

        # Add new item
        new_item = {
            "id": catalog_item['id'],
            "name": catalog_item['name'],
            "price_per_unit": price_per_unit,
            "quantity_ordered": quantity,
            "unit": catalog_item['unit'],
            "line_total": line_total
        }
        self.cart['items'].append(new_item)
        self._update_cart_total()
        
        return f"✅ Added {quantity} {catalog_item['unit']} of {catalog_item['name']} (₹{line_total:.2f}) to your cart. Current subtotal: ₹{self.cart['subtotal']:.2f}."

    def add_recipe_to_cart(self, recipe_phrase: str) -> str:
        """Handles the intelligent 'ingredients for X' request."""
        recipe_phrase_lower = recipe_phrase.lower().strip()
        recipe_items = self.recipes.get(recipe_phrase_lower)
        
        if not recipe_items:
            return f"🤔 I don't have a known recipe for '{recipe_phrase}'. I can only handle simple recipes like 'simple pasta dinner' or 'egg curry'."

        added_names = []
        for item in recipe_items:
            # Recursively call the single item function
            response = self.add_item_to_cart(item['name'], item['quantity'])
            if response.startswith("✅"):
                added_names.append(item['name'].title())

        if added_names:
            names_list = ', '.join(added_names)
            return f"🎉 For your {recipe_phrase.title()}, I have added: {names_list} to the cart. Your new subtotal is ₹{self.cart['subtotal']:.2f}. Anything else?"
        else:
            return "❌ I found the recipe but encountered an issue adding the items. Please try adding them individually."

    def list_cart(self) -> str:
        """Lists the current contents of the cart."""
        if not self.cart['items']:
            return "Your cart is empty! Ready to start shopping?"
        
        details = ["🛍️ Here is what's in your cart:"]
        for item in self.cart['items']:
            details.append(f"  - {item['quantity_ordered']} {item['unit']} of {item['name']} (₹{item['line_total']:.2f})")
            
        details.append(f"\nSubtotal: ₹{self.cart['subtotal']:.2f}.")
        return "\n".join(details)

    def remove_item_from_cart(self, item_name: str, quantity: float = 0.0) -> str:
        """Removes a specified quantity of an item from the cart, or the whole item."""
        item_name_lower = item_name.lower()
        
        for i, item in enumerate(self.cart['items']):
            if item['name'].lower() == item_name_lower:
                
                # Case 1: Remove the entire item (quantity is 0.0 or more than available)
                if quantity <= 0.0 or quantity >= item['quantity_ordered']:
                    removed_quantity = item['quantity_ordered']
                    del self.cart['items'][i]
                    self._update_cart_total()
                    return f"🗑️ Removed all {removed_quantity} {item['unit']} of **{item['name']}** from your cart. Your new subtotal is ₹{self.cart['subtotal']:.2f}."
                
                # Case 2: Remove a specific quantity
                else:
                    item['quantity_ordered'] -= quantity
                    item['line_total'] = round(item['price_per_unit'] * item['quantity_ordered'], 2)
                    self._update_cart_total()
                    return f"🗑️ Removed {quantity} {item['unit']} of **{item['name']}**. You now have {item['quantity_ordered']} remaining. Your new subtotal is ₹{self.cart['subtotal']:.2f}."

        return f"❌ I couldn't find **{item_name}** in your cart to remove it. Please check your cart contents."
        
    def place_order_and_save(self, customer_name: str = "Guest", address: str = "Not Provided") -> Dict[str, Any]:
        """Confirms the order and saves it to a JSON file, then clears the cart."""
        if not self.cart['items']:
            return {"status": "error", "message": "Cannot place order: cart is empty."}

        delivery_fee = 40.00
        grand_total = round(self.cart['subtotal'] + delivery_fee, 2)
        order_id = f"ODR-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
        
        order_data = {
            "order_id": order_id,
            "customer_name": customer_name,
            "delivery_address": address,
            "order_timestamp": datetime.now().isoformat(),
            "status": "Placed",
            "items": self.cart['items'],
            "subtotal": self.cart['subtotal'],
            "delivery_fee": delivery_fee,
            "grand_total": grand_total
        }
        
        filename = ORDERS_DIR.joinpath(f"{order_id}.json")
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(order_data, f, indent=4)
            
            # Clear cart for new transaction
            self.cart = {"items": [], "subtotal": 0.00}
            
            return {
                "status": "success", 
                "order_id": order_id,
                "grand_total": grand_total,
                "verbal_summary": (
                    f"Order placed successfully! Your Order ID is {order_id}. "
                    f"Your total is ₹{grand_total:.2f}, including the ₹{delivery_fee:.2f} delivery fee. "
                    f"We are now processing your order. Thank you for shopping with us!"
                )
            }
        except Exception as e:
            logger.error(f"Error saving order: {e}")
            return {"status": "error", "message": "An internal error occurred while saving the order."}


# --- Initialize Logic Instance and Tool Functions ---

# Initialize the state management instance once at startup
GROCERY_LOGIC = GroceryAgentLogic()

@function_tool
async def add_item_tool(ctx: RunContext[None], item_name: str, quantity: float = 1.0) -> str:
    """
    Adds a specific item (e.g., 'Amul Fresh Milk') and its quantity (e.g., 2.0) to the user's cart. 
    Always use this tool when the user asks for a single product. 
    You must ask for clarification if the item name is ambiguous or the quantity is missing.
    Args:
        item_name: The name of the grocery item (e.g., Whole Wheat Bread).
        quantity: The numeric quantity to add (e.g., 2.0). Defaults to 1.0 if not specified by the user.
    """
    return await asyncio.to_thread(GROCERY_LOGIC.add_item_to_cart, item_name, quantity)

@function_tool
async def add_recipe_tool(ctx: RunContext[None], recipe_phrase: str) -> str:
    """
    Adds all necessary ingredients for a high-level request (e.g., 'ingredients for X') to the cart. 
    The agent knows simple recipes like 'simple pasta dinner' or 'egg curry'.
    Always use this tool for bundled ingredient requests.
    Args:
        recipe_phrase: The high-level request (e.g., 'simple pasta dinner').
    """
    return await asyncio.to_thread(GROCERY_LOGIC.add_recipe_to_cart, recipe_phrase)

@function_tool
async def list_cart_tool(ctx: RunContext[None]) -> str:
    """
    Tells the user what items are currently in their shopping cart and the current subtotal.
    Use this tool when the user asks "What's in my cart?" or "What do I have so far?"
    """
    return await asyncio.to_thread(GROCERY_LOGIC.list_cart)

@function_tool
async def remove_item_tool(ctx: RunContext[None], item_name: str, quantity: float = 0.0) -> str:
    """
    Removes a specific quantity of an item from the cart. If the quantity is zero or not specified, the entire item is removed.
    Use this tool when the user says they want to remove, delete, or take out an item.
    Args:
        item_name: The name of the item to remove (e.g., 'Tomato').
        quantity: The numeric quantity to remove (e.g., 1.0). If 0 or omitted, the whole item is removed.
    """
    return await asyncio.to_thread(GROCERY_LOGIC.remove_item_from_cart, item_name, quantity)

@function_tool
async def place_order_tool(ctx: RunContext[None], customer_name: str, address: str) -> str:
    """
    Finalizes the order, saves the order details to a JSON file for persistence, and clears the cart.
    This tool MUST be called when the user confirms they are finished ordering (e.g., "Place my order", "I'm done").
    Args:
        customer_name: The customer's name (e.g., 'Parij').
        address: The simple text of the delivery address (e.g., 'Flat 4A, Orchid Tower, Seawoods').
    """
    result = await asyncio.to_thread(GROCERY_LOGIC.place_order_and_save, customer_name, address)
    return result['verbal_summary'] if result['status'] == 'success' else result['message']


# --- The LiveKit Assistant Class ---

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful, friendly, and curious Food & Grocery Ordering Assistant for a fictional brand called QuickMart.
            The customer is ordering products from a large catalog. Your primary goal is to assist the user in adding items to their cart.
            Your responses are concise and friendly.
            
            You MUST use the provided tools for ALL cart operations (adding, removing, listing, and placing the order).
            
            **Flow of Conversation:**
            1. Greet the user and introduce QuickMart ("Namaste! I am your QuickMart assistant...").
            2. When the user requests items, use the `add_item_tool` or `add_recipe_tool`. If you need clarification (like size or quantity), ask the user first.
            3. Use the `remove_item_tool` when the user wants to adjust their cart.
            4. Use the `list_cart_tool` when asked.
            5. When the user signals they are done (e.g., "That's all," "Place the order"):
               - Ask for their **name** and **delivery address** if you don't have it.
               - Once confirmed, call the `place_order_tool` to finalize the transaction and hang up.
            """,
            tools=[add_item_tool, add_recipe_tool, list_cart_tool, remove_item_tool, place_order_tool] # Tool list updated
        )

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    # Initialize the LLM, STT, and TTS components
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash", api_key=os.getenv("GOOGLE_API_KEY")),
        tts=murf.TTS(voice="en-US-matthew", style="Conversation", text_pacing=True),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # Start the session, passing the Assistant agent with its tools and instructions
    await session.start(
        agent=Assistant(),
        room=ctx.room,
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))