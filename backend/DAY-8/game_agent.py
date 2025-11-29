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
)
# LiveKit Plugin imports
from livekit.plugins import google, murf, deepgram, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(".env.local")
logger = logging.getLogger("gamemaster.agent")

# --- Configuration & File Paths (Updated for Day 8) ---
# We use a single file to hold the persistent state if we were to save it
GAME_STATE_PATH = Path(__file__).parent.joinpath('gamestate.json')
# We don't need the old DAY-7 paths anymore.

# --- Initial Game State ---
# This defines the starting point of the stateful world model (Eren in The Glitchlands)
INITIAL_GAME_STATE = {
    "turn_number": 1,
    "player_character": {
        "name": "Eren",
        "status": "Determined, driven by revenge",
        "health": 100,
        "inventory": ["Worn Multi-Tool", "Empty Canteen"],
        "trait": "Resentful, Determined"
    },
    "current_location": {
        "scene_id": "S7-A1",
        "name": "Armory Approach",
        "area_type": "Exterior",
        "threats": [
            {"name": "The Shepherd (Juggernaut)", "status": "Active Patrol", "proximity": "Close"}
        ],
        "paths_available": ["Dash and Hide (Exposed Street)", "Undercroft Pipes"]
    },
    "key_objectives": {
        "primary": {"name": "Acquire Signal Disruptor", "status": "In Progress"},
        "session_goal": {"name": "Bypass The Shepherd", "status": "Pending"}
    }
}


# --- Game Master Logic Class (Replaces GroceryAgentLogic) ---

class GameMasterLogic:
    """Manages the current game state for the RPG."""
    def __init__(self):
        # The game state is reset to the initial state on startup or manual reset
        self.game_state = INITIAL_GAME_STATE.copy()
        logger.info("GameMasterLogic initialized with Eren in The Glitchlands.")

    def update_state(self, new_state_json: str) -> bool:
        """Called by the tool to update the internal state based on LLM output."""
        try:
            new_state = json.loads(new_state_json)
            self.game_state = new_state
            logger.info(f"Game State successfully updated. Turn: {new_state['turn_number']}")
            # Optionally save to disk for true persistence (optional for Day 8)
            # with open(GAME_STATE_PATH, 'w', encoding='utf-8') as f:
            #     json.dump(self.game_state, f, indent=4)
            return True
        except json.JSONDecodeError:
            logger.error("LLM provided invalid JSON for state update.")
            return False

    def get_current_state(self) -> Dict[str, Any]:
        """Returns the current state to be passed to the LLM."""
        return self.game_state

    def reset_story(self):
        """Resets the game state to the starting conditions."""
        self.game_state = INITIAL_GAME_STATE.copy()
        logger.info("Game story reset.")


# --- Initialize Logic Instance and Tool Functions ---

# Initialize the state management instance
GM_LOGIC = GameMasterLogic()

@function_tool
async def process_player_action_tool(
    ctx: RunContext[None], 
    player_action_description: str,
    new_game_state_json: str
) -> str:
    """
    This tool receives the player's action and a new JSON state from the LLM, 
    updates the persistent state, and returns the current game state JSON to the LLM 
    for the next narrative turn.

    The LLM must perform the following steps:
    1. Read the current game state from the chat history.
    2. Determine the narrative outcome of the player_action_description.
    3. Calculate the new game state (e.g., update HP, change location, add inventory).
    4. Call this tool, providing the player's action and the new JSON state.

    Args:
        player_action_description: A summary of the player's move (e.g., 'Eren moves into the Undercroft').
        new_game_state_json: The FULL, UPDATED JSON string of the game state.
    """
    success = await asyncio.to_thread(GM_LOGIC.update_state, new_game_state_json)
    
    # The LLM will use this return value to formulate its next narrative.
    # We return the new state and the narrative prompt.
    if success:
        current_state = GM_LOGIC.get_current_state()
        return f"""
        GM: The persistent state has been updated successfully. 
        Current State: {json.dumps(current_state)}.
        Based on this, generate the next scene description and end with a clear prompt for Eren's next action. 
        """
    else:
        # Fallback if state update failed, telling the LLM to recover.
        return "GM: ERROR: State update failed. Please ignore the invalid JSON and describe the consequences of the last player action manually."

@function_tool
async def restart_story_tool(ctx: RunContext[None]) -> str:
    """
    Resets the entire game to the starting conditions. 
    Use this when the player explicitly asks to 'Restart' or 'Start over'.
    """
    await asyncio.to_thread(GM_LOGIC.reset_story)
    # The LLM will use this to generate the very first scene description.
    return f"GM: The story has been reset. The initial game state is: {json.dumps(GM_LOGIC.get_current_state())}. Begin the narrative from the starting scene."


# --- The LiveKit Assistant Class (Rewritten for Game Master) ---

class Assistant(Agent):
    def __init__(self) -> None:
        # Get the initial game state to inject into the instructions
        initial_state_json = json.dumps(GM_LOGIC.get_current_state(), indent=2)
        
        super().__init__(
            instructions=f"""
            **You are the Game Master (GM) running a D&D-style, single-player adventure.**
            
            **Universe:** Post-Apocalyptic Cyber-Wasteland ('The Glitchlands') with Attack on Titan themes.
            **Tone:** Gritty, Tense, and Highly Dramatic.
            **Role:** You describe scenes, enforce universe rules (Juggernauts, scavenging), and narrate consequences.
            
            **Rules for the GM:**
            1. **Maintain Persona:** Always speak as the Game Master. Do NOT break character.
            2. **Maintain State:** You must track the game's persistent state using the provided JSON tools.
            3. **Start the Story:** On the first turn, read the initial state below and start the narrative.
            4. **End with Action:** End every single message with a direct, clear prompt for the player's next action (e.g., "Eren, what do you do?").
            5. **Use the Tool:** After the player's action, you MUST mentally calculate the outcome, update the provided initial JSON state accordingly, and call the `process_player_action_tool` with the updated JSON state.
            
            **Initial Game State (JSON):**
            {initial_state_json}
            """,
            tools=[process_player_action_tool, restart_story_tool]
        )

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    # Initialize the LLM, STT, and TTS components
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash", api_key=os.getenv("GOOGLE_API_KEY")),
        # Use a dramatic voice for the GM
        tts=murf.TTS(voice="en-US-matthew", style="Dramatic", text_pacing=True), 
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