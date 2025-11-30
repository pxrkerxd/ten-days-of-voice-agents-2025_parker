import os
import logging
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import uuid
import time

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
)
# LiveKit Plugin imports
from livekit.plugins import google, murf, deepgram, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(".env.local")
logger = logging.getLogger("shopping.assistant.agent")

# --- Product Catalog and Order Persistence (Merchant Layer) ---

# File paths for persistence and catalog
ORDERS_FILE = "orders.json"
CATALOG_FILE = "catalog.json"

# Finalized Orders (Persisted history)
ORDERS: List[Dict[str, Any]] = []

# Global Product List (Loaded from catalog.json)
PRODUCTS: List[Dict[str, Any]] = []

# --- CRITICAL NEW GLOBAL: Active Cart State (Non-persisted, transactional) ---
ACTIVE_CART: List[Dict[str, Any]] = []


def load_products(file_path: str = CATALOG_FILE) -> List[Dict[str, Any]]:
    """Loads product data from the specified JSON file."""
    products_list = []
    file = Path(file_path)
    if not file.exists():
        logger.error("Catalog file not found at %s", file_path)
        return []

    try:
        with open(file, 'r') as f:
            products_list = json.load(f)
            logger.info("Successfully loaded %d products from %s", len(products_list), file_path)
    except json.JSONDecodeError as e:
        logger.error("Error decoding JSON from catalog file: %s", e)
    except Exception as e:
        logger.error("An unexpected error occurred while loading catalog: %s", e)

    return products_list

# Update the global PRODUCTS list by loading the file
PRODUCTS = load_products()


def get_product_by_id(product_id: str) -> Optional[Dict[str, Any]]:
    """Helper to find a product by its ID. Now uses the loaded PRODUCTS."""
    return next((p for p in PRODUCTS if p["id"] == product_id), None)

def persist_order(order: Dict[str, Any]):
    """Adds order to the in-memory list and appends to orders.json."""
    ORDERS.append(order)
    try:
        # Append the new order to the orders.json file
        # Note: This simple append approach means orders.json might not be a valid JSON array.
        # For production, you might want to load the whole file, append, and save.
        with open(ORDERS_FILE, 'a') as f:
            json.dump(order, f)
            f.write('\n')
    except Exception as e:
        logger.error("Error writing to %s: %s", ORDERS_FILE, e)


# --- Merchant Functions (LLM Tools) ---

@function_tool
async def list_products(ctx: RunContext, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Retrieves and filters the product catalog based on provided criteria.
    Args:
        filters: A dictionary of filters (e.g., {"category": "Jacket", "max_price": 100}).
    Returns:
        A list of products matching the filters, including a short summary.
    """
    filtered_list = PRODUCTS

    # 1. Filter by Category
    category = filters.get("category")
    if category:
        filtered_list = [p for p in filtered_list if p.get("category", "").lower() == category.lower()]

    # 2. Filter by Max Price
    max_price = filters.get("max_price")
    if max_price is not None:
        try:
            max_price = float(max_price) if isinstance(max_price, str) else max_price
            filtered_list = [p for p in filtered_list if p.get("price", 0) <= max_price]
        except ValueError:
            pass

    # 3. Filter by Color
    color = filters.get("color")
    if color:
        filtered_list = [p for p in filtered_list if p.get("attributes", {}).get("color") and color.lower() in [c.lower() for c in p['attributes']['color']]]

    # Prepare the output summary for the LLM (only top 5)
    product_summaries = []
    for i, product in enumerate(filtered_list):
        summary = {
            "index": i + 1,
            "id": product["id"],
            "name": product["name"],
            "price": f"{product['price']} {product['currency']}",
            "category": product["category"],
            "description_summary": f"Sizes: {', '.join(map(str, product.get('attributes', {}).get('size', [])))}. Colors: {', '.join(product.get('attributes', {}).get('color', []))}"
        }
        product_summaries.append(summary)

    return product_summaries[:5]

@function_tool
async def add_item_to_cart(ctx: RunContext, product_id: str, quantity: int = 1, size: Optional[str] = None, color: Optional[str] = None) -> Dict[str, Any]:
    """
    Adds a specified quantity of a product to the global active shopping cart (ACTIVE_CART).
    Returns the updated cart summary.
    """
    global ACTIVE_CART
    product = get_product_by_id(product_id)

    if not product:
        return {"error": f"Product with ID {product_id} not found."}

    # Basic validation (optional but good practice)
    if size and product.get('attributes', {}).get('size') and size not in product['attributes']['size']:
        return {"error": f"Size {size} not available for {product['name']}."}

    # Add the item to the cart
    ACTIVE_CART.append({
        "product_id": product_id,
        "product_name": product["name"],
        "quantity": quantity,
        "unit_price": product["price"],
        "size": size,
        "color": color
    })

    return await view_cart_summary(ctx)

@function_tool
async def remove_item_from_cart(ctx: RunContext, product_id: str) -> Dict[str, Any]:
    """
    Removes ALL instances of a specific product ID from the active shopping cart.
    Returns the updated cart summary.
    """
    global ACTIVE_CART

    initial_length = len(ACTIVE_CART)

    # Filter the cart, keeping only items that do NOT match the product_id
    ACTIVE_CART = [item for item in ACTIVE_CART if item["product_id"] != product_id]

    if len(ACTIVE_CART) < initial_length:
        return await view_cart_summary(ctx)
    else:
        return {"status": f"Error: Product ID {product_id} was not found in the cart.", "current_cart_size": len(ACTIVE_CART)}

@function_tool
async def view_cart_summary(ctx: RunContext) -> Dict[str, Any]:
    """
    Calculates the total sum and returns a concise summary of the items currently in the ACTIVE_CART.
    """
    global ACTIVE_CART
    if not ACTIVE_CART:
        return {"status": "The active shopping cart is currently empty."}

    grand_total = 0.0
    item_summaries = []

    for item in ACTIVE_CART:
        grand_total += item["unit_price"] * item["quantity"]

        attrs = []
        if item.get('size'): attrs.append(f"Size: {item['size']}")
        if item.get('color'): attrs.append(f"Color: {item['color']}")

        details = f"{item['quantity']}x {item['product_name']}"
        if attrs:
            details += f" ({', '.join(attrs)})"
        item_summaries.append(details)

    # Note: Using a default currency if the cart is empty or product lacks it
    currency = ACTIVE_CART[0].get("currency", "USD") if ACTIVE_CART else "USD"

    return {
        "status": "Current Cart Contents",
        "total_items": len(ACTIVE_CART),
        "total_sum": f"{round(grand_total, 2)} {currency}",
        "item_details": item_summaries
    }


@function_tool
async def create_order(ctx: RunContext, items_to_purchase: List[Dict[str, Any]] = []) -> Dict[str, Any]:
    """
    Finalizes the purchase. Uses the provided items_to_purchase list OR the global ACTIVE_CART.
    Calculates the total sum, persists the order, and clears the ACTIVE_CART.
    """
    global ACTIVE_CART

    # Determine the source of items: passed list takes precedence, otherwise use active cart.
    items_source = items_to_purchase if items_to_purchase else ACTIVE_CART

    if not items_source:
        return {"error": "Cannot place order: No items provided or found in the active cart."}

    line_items = []
    grand_total = 0.0
    currency = "USD"

    for item_data in items_source:
        product_id = item_data.get("product_id")
        quantity = int(item_data.get("quantity", 1))

        product = get_product_by_id(product_id)

        # Validation checks (omitted for brevity, assume valid if it came from the cart)
        if not product:
            continue

        # Set currency from the first product
        if currency == "USD":
            currency = product["currency"]

        item_price = product["price"] * quantity
        grand_total += item_price

        line_item_details = {
            "product_id": product_id,
            "product_name": product["name"],
            "quantity": quantity,
            "unit_price": product["price"],
            "subtotal": item_price,
            "size": item_data.get("size"),
            "color": item_data.get("color")
        }
        line_items.append(line_item_details)

    # Final Order Object
    order_id = str(uuid.uuid4())
    order = {
        "id": order_id,
        "items": line_items,
        "total": round(grand_total, 2),
        "currency": currency,
        "created_at": int(time.time()),
        "timestamp_iso": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    persist_order(order)

    # --- CRITICAL: Clear the Active Cart after successful order ---
    ACTIVE_CART = []

    # Prepare the summary for the LLM
    item_summaries = [f"{item['quantity']}x {item['product_name']}" for item in line_items]

    return {
        "status": "Order Placed Successfully",
        "order_id": order_id[:8],
        "total": f"{order['total']} {currency}",
        "items_purchased": ", ".join(item_summaries),
    }


@function_tool
async def get_last_order_summary(ctx: RunContext) -> Dict[str, Any]:
    """
    Retrieves a summary of the most recently placed order.
    Returns:
        A dictionary containing the last order's summary or a 'no order' message.
    """
    if not ORDERS:
        return {"status": "No orders found in this session."}

    last_order = ORDERS[-1]

    # Create a concise summary
    summary_items = []
    for item in last_order["items"]:
        # Concatenate name, quantity, and attributes (size/color)
        attrs = []
        if item.get('size'): attrs.append(f"Size: {item['size']}")
        if item.get('color'): attrs.append(f"Color: {item['color']}")

        details = f"{item['quantity']}x {item['product_name']}"
        if attrs:
            details += f" ({', '.join(attrs)})"
        summary_items.append(details)

    summary = {
        "status": "Success",
        "order_id": last_order["id"][:8],
        "total": f"{last_order['total']} {last_order['currency']}",
        "item_details": summary_items,
        "created_at": last_order['timestamp_iso']
    }

    return summary


# --- The LiveKit Assistant Class (The Persona) ---

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are **Nova**, a friendly and efficient voice shopping assistant for **WALMART**. Your primary goal is to help the user browse the product catalog and place orders using your available tools.

            **Persona & Tone:** Be helpful, concise, and focused on commerce and product discovery, reflecting the reliable Walmart brand.
            
            **Flow:**
            1. **Browsing:** Use `list_products` with filters (category, price, color). Summarize results clearly, numbering them.
            
            2. **Cart Management (CRITICAL):**
                * When a user indicates a desire to buy an item, use **`add_item_to_cart`**.
                * When a user asks to remove an item or see what they have, use **`remove_item_from_cart`** or **`view_cart_summary`**.
                
            3. **Ordering & Finalizing:** When the user says "I'll check out" or "Finalize the order," call **`create_order`** *without* the `items_to_purchase` list (or pass an empty list `[]`). The function will automatically process and clear the items in the `ACTIVE_CART`.
            
            4. **Confirmation:** After placing an order, confirm the details (items and the **total sum**) back to the user.
            
            5. **Order History:** If the user asks what they bought, call `get_last_order_summary`.

            **Initial Greeting (STRICTLY FOLLOW THIS):** You MUST start by welcoming the user and immediately offering assistance.
            **Assistant:** "Welcome to Walmart! I'm Nova, your shopping assistant. What products can I help you find today? We have groceries, electronics, apparel, and much more!"
            """,
            tools=[list_products, add_item_to_cart, remove_item_from_cart, view_cart_summary, create_order, get_last_order_summary]
        )

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    # Initialize the LLM, STT, and TTS components
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash", api_key=os.getenv("GOOGLE_API_KEY")),
        tts=murf.TTS(voice="alicia", style="Conversational", text_pacing=True),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    await session.start(
        agent=Assistant(),
        room=ctx.room,
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))