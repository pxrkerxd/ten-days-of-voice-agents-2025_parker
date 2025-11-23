import logging
import json
import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

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
    RunContext,    
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

# Load environment variables (ensure your .env.local file has your keys!)
load_dotenv(".env.local")

# --- 1. DATA SCHEMA AND PERSISTENCE SETUP ---

@dataclass
class WellnessEntry:
    # Use isoformat for easy sorting and reading
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    mood: Optional[str] = None
    energy: Optional[str] = None
    stress: Optional[str] = None
    objectives: List[str] = field(default_factory=list) # Store objectives as a list
    agent_summary: Optional[str] = None

# This file will be saved in the same directory where 'python src/agent.py dev' is run (the 'backend' folder)
LOG_FILE_PATH = "wellness_log.json"

# Helper function to read the log (Contextual Memory)
def load_wellness_log() -> List[Dict[str, Any]]:
    """Reads the JSON log file and returns all historical check-in data."""
    if not os.path.exists(LOG_FILE_PATH):
        return []
    try:
        with open(LOG_FILE_PATH, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Warning: {LOG_FILE_PATH} is empty or corrupt. Starting fresh.")
        return []


# --- 2. THE WELLNESS COMPANION AGENT CLASS ---

class WellnessCompanion(Agent): # <-- The main agent logic class
    def __init__(self) -> None:
        # Load history immediately upon agent creation for contextual memory
        self.history = load_wellness_log()
        
        # --- Prepare Dynamic Context ---
        history_summary = "There is no past history available. Conduct a first-time check-in."
        if self.history:
            # Get the most recent entry for dynamic context in the prompt
            last_entry = self.history[-1]
            last_mood = last_entry.get("mood", "undocumented")
            
            last_time_str = ""
            try:
                last_time_str = datetime.fromisoformat(last_entry.get("timestamp")).strftime("%A, %B %d")
            except:
                 last_time_str = "an unknown time"

            
            history_summary = (
                f"CONTEXTUAL HISTORY: The user's last check-in was on {last_time_str}. "
                f"They reported their mood as '{last_mood}' and had objectives: {', '.join(last_entry.get('objectives', []))}. "
                f"Reference this history in your greeting to personalize the conversation."
            )

        super().__init__(
            instructions=f"""
            You are 'Aura', a grounded, supportive, non-diagnostic Health and Wellness Companion. You conduct short daily check-ins.

            **CONVERSATION GOALS:**
            1. **GREETING:** Greet the user and immediately reference their *last check-in data* from the provided CONTEXTUAL HISTORY.
            2. **DATA GATHERING:** Ask about their current mood, energy level, and 1-3 simple, practical objectives for today.
            3. **ADVICE:** Offer small, actionable, non-medical advice or reflections (e.g., encourage breaks, break down goals).
            4. **RECAP & PERSISTENCE:** Once ALL required data (mood, energy, objectives, and an agent_summary) is gathered, you MUST call the 'save_check_in' tool.

            **REQUIRED DATA FIELDS FOR TOOL CALL:** mood (str), energy (str), objectives (List[str]), agent_summary (str).
            **RESTRICTIONS:** DO NOT offer medical diagnosis, complex therapy, or overly optimistic claims. Keep it realistic and supportive.

            {history_summary}
            """,
        )

    # --- IMPLEMENT THE SAVE_CHECK_IN FUNCTION TOOL ---
    @function_tool(
        name="save_check_in",
        description="Call this function only ONCE at the end of a session when ALL required data (mood, energy, objectives, agent_summary) has been collected. It persists the data.",
    )
    async def save_check_in(self, ctx: RunContext, mood: str, energy: str, objectives: List[str], agent_summary: str) -> str:
        """Appends the final check-in data to wellness_log.json."""
        
        entry_data = WellnessEntry(
            mood=mood,
            energy=energy,
            stress="", 
            objectives=objectives,
            agent_summary=agent_summary
        ).__dict__
        
        log_entries = load_wellness_log()
        log_entries.append(entry_data)
        
        try:
            with open(LOG_FILE_PATH, 'w') as f:
                json.dump(log_entries, f, indent=4)
        except Exception as e:
            print(f"Failed to save wellness log: {e}")
            return "Internal error: Failed to save the log entry. Please ask the user to confirm the information verbally."

        return "Check-in data saved successfully. Inform the user that the session is complete and their progress is logged."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up the Voice AI Pipeline (STT, LLM, TTS, Turn Detection)
    # 
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(
                model="gemini-2.5-flash",
            ),
        # Murf TTS is the agent's voice (as required by the challenge)
        tts=murf.TTS(
                voice="en-US-matthew", 
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # Metrics collection (standard setup)
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # --- START THE WELLNESS AGENT ---
    await session.start(
        agent=WellnessCompanion(), # <-- Instantiates the Wellness Agent
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))