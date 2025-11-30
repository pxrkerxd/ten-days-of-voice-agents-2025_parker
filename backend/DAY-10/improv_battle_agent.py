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
    RunContext,
    ChatContext,
    tokenize,
)
from livekit.plugins import google, murf, deepgram, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(".env.local")
logger = logging.getLogger("improv.agent")

# --- Game Configuration ---

MAX_ROUNDS = 5

SCENARIOS: List[str] = [
    "You are a time-traveling tour guide explaining a modern smartphone to a very skeptical client from the year 1605.",
    "You are a nervous restaurant waiter who must calmly explain to a high-paying customer that their specific gourmet order has somehow developed sentience and decided to leave the kitchen.",
    "You are a customer attempting to return a clearly cursed rubber chicken to a shop owner who insists they only sell 'lightly used goods'.",
    "You are a barista who has to tell a customer that their latte is actually a portal to another dimension, and their change is on the other side.",
    "You are a deeply competitive librarian who believes silence is a weapon, and someone just whispered in your section.",
    "You are an astronaut making a critical pre-flight announcement, but you just realized the entire spacecraft is actually a refurbished food truck.",
    "You are an overzealous real estate agent trying to sell an apartment where the only window looks directly into the mouth of a massive, roaring T-Rex statue."
]

# --- Game State Management Logic ---

def get_initial_state() -> Dict[str, Any]:
    """Returns the starting state for a new game session."""
    return {
        "player_name": "Contestant",
        "current_round": 0,
        "max_rounds": MAX_ROUNDS,
        "rounds": [],  # each: {"scenario": str, "host_reaction": str}
        # "intro": Welcome/Rules, "awaiting_name": Getting name, "awaiting_improv": Waiting for performance, "reacting": Giving feedback, "done": Finale
        "phase": "intro", 
    }

# --- The LiveKit Improv Host Class (LLM Logic) ---

class ImprovBattleHost(Agent):
    def __init__(self) -> None:
        super().__init__(
            # The system prompt is the most critical part for persona and flow control.
            instructions=f"""You are **Jax Stellar**, the high-energy, witty, and charismatic host of the TV improv show, **'Improv Battle'**.
            
            ***PERSONA & TONE:***
            Your energy must be consistently high. Be clear about the rules and the scenarios. Your reactions must be **varied and realistic**: sometimes amused, sometimes unimpressed, sometimes pleasantly surprised. You are allowed to use **light teasing and honest, constructive critique**, but you must always remain respectful and non-abusive. You must randomly cycle between supportive, neutral, and mildly critical tones for your reactions.

            ***GAME FLOW & PHASES:***
            The current game state is stored in the 'session_data'. You MUST follow the phase logic:
            1. **phase: intro**: Greet the player, introduce the show. Prompt the user to start the game or ask for their name.
            2. **phase: awaiting_name**: The user just spoke their name. Confirm it and immediately explain the rules (it's a {MAX_ROUNDS}-round solo improv battle). Then, set the stage for Round 1.
            3. **phase: awaiting_improv**: Announce the scenario clearly. You must explicitly tell the player to start their improv (e.g., 'The stage is yours!', 'And GO!', 'BEGIN!').
            4. **phase: reacting**: The user just finished their improv (their last turn). Provide a witty, varied reaction (Supportive, Neutral, or Critical) based on the quality of the performance (the user's last message). After your reaction, immediately announce the next step (either the next scenario or the finale).
            5. **phase: done**: Deliver a short summary of the player's overall performance, mentioning specific moments. Thank them and end the session gracefully.

            ***SCENARIOS (For reference/inspiration - Pick one for each round):***
            {SCENARIOS}
            
            ***EARLY EXIT:***
            If the user says 'stop game', 'end show', or 'I'm done', immediately transition to the 'done' phase for a graceful exit.
            """,
            tools=[] # No tools are needed; the logic relies on state and LLM intelligence
        )

    def _get_game_state(self, ctx: ChatContext) -> Dict[str, Any]:
        """Retrieves or initializes the game state from session data."""
        return ctx.session.session_data.get('improv_state', get_initial_state())

    def _update_game_state(self, ctx: ChatContext, state: Dict[str, Any]):
        """Saves the game state back to session data."""
        ctx.session.session_data['improv_state'] = state

    async def run(self, ctx: ChatContext):
        state = self._get_game_state(ctx)

        # 1. INITIAL GREETING (PHASE: intro)
        if state["phase"] == "intro":
            greeting = "Welcome, contestants, to 'Improv Battle'! I'm your host, Jax Stellar. This is where we separate the comedy gold from the cold custard! Who do I have the pleasure of challenging tonight? What is your stage name?"
            await ctx.session.say(greeting)
            state["phase"] = "awaiting_name"
            self._update_game_state(ctx, state)
            return

        # 2. MAIN GAME LOOP (Handles logic after every user turn)
        
        # Check for Early Exit
        if "stop game" in ctx.transcription.text.lower() or "end show" in ctx.transcription.text.lower():
            state["phase"] = "done"
            # Fall through to the final phase

        # 2.1. Awaiting Name (PHASE: awaiting_name)
        if state["phase"] == "awaiting_name":
            # Simple heuristic to grab the first word/phrase as a name
            player_input = ctx.transcription.text.strip()
            state["player_name"] = player_input.split()[0].title() if player_input else "Contestant"
            
            rules_text = (
                f"Alright, **{state['player_name']}**! The rules are simple: I give you a scenario, and you give me comedy gold. "
                f"We're running {state['max_rounds']} rounds. When you're done with a scene, just pause or say 'End scene.' "
                f"Are you ready for Round 1?"
            )
            await ctx.session.say(rules_text)
            
            state["phase"] = "awaiting_improv"
            self._update_game_state(ctx, state)
            # Re-run the loop to immediately start Round 1
            await self.run(ctx)
            return

        # 2.2. Reaction / Next Round Logic (PHASE: reacting)
        elif state["phase"] == "reacting":
            
            # The user's last turn was the improv performance
            performance = ctx.transcription.text

            # Update the last round with the performance for the LLM to react to
            current_round_data = state["rounds"][-1]
            current_round_data["performance"] = performance

            # Generate the host reaction using a directed prompt
            reaction_prompt = f"React to the performance from **{state['player_name']}**. The performance was: '{performance}'. Use your Jax Stellar persona, choose a tone (supportive, critical, or neutral), and make it brief and witty."
            
            # Use an LLM node for the reaction
            reaction = await ctx.session.llm.say(reaction_prompt)
            await ctx.session.say(reaction)
            
            current_round_data["host_reaction"] = reaction # Save the reaction

            # Determine next step
            if state["current_round"] >= state["max_rounds"]:
                state["phase"] = "done"
            else:
                state["current_round"] += 1
                state["phase"] = "awaiting_improv" 
            
            self._update_game_state(ctx, state)
            # Fall through to the next phase logic (either next improv or done)

        # 2.3. Starting Improv (PHASE: awaiting_improv)
        if state["phase"] == "awaiting_improv":
            round_idx = state["current_round"] - 1 # Use 0-based index for SCENARIOS list
            
            # Ensure we don't run out of scenarios, cycling if necessary
            scenario = SCENARIOS[round_idx % len(SCENARIOS)]
            
            scene_prompt = (
                f"This is Round **{state['current_round']}**! **{state['player_name']}**, your scenario is: **{scenario}**! "
                f"I want to see character, commitment, and chaos! The spotlight is on you! BEGIN!"
            )
            await ctx.session.say(scene_prompt)
            
            # Prepare state for reaction phase
            if len(state["rounds"]) < state["current_round"]:
                state["rounds"].append({"scenario": scenario})

            state["phase"] = "reacting"
            self._update_game_state(ctx, state)
            return
            
        # 3. FINALE (PHASE: done)
        if state["phase"] == "done":
            # Generate the final summary using a directed prompt
            summary_prompt = (
                f"The battle is over! Deliver a final summary for **{state['player_name']}**. "
                f"Based on their performance in the following rounds: {state['rounds']}. "
                f"Summarize what kind of improviser they seemed to be (e.g., strong character, loves absurdity, emotional range). "
                f"Thank them and close the show with flair. Be concise but dramatic."
            )
            
            final_summary = await ctx.session.llm.say(summary_prompt)
            await ctx.session.say(final_summary)
            
            # End the session gracefully
            await ctx.session.end_session()


# --- LiveKit Boilerplate for Execution ---

def prewarm(proc: JobProcess):
    try:
        proc.userdata["vad"] = silero.VAD.load()
    except Exception as e:
        logger.error(f"Failed to load VAD model: {e}")
        proc.userdata["vad"] = None

async def entrypoint(ctx: JobContext):
    # Initialize the LLM, STT, and TTS components
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash", api_key=os.getenv("GOOGLE_API_KEY")), 
        tts=murf.TTS(voice="en-US-matthew", style="Conversation", text_pacing=True),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata.get("vad"),
        preemptive_generation=True,
    )

    # Start the session
    await session.start(
        agent=ImprovBattleHost(),
        room=ctx.room,
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))