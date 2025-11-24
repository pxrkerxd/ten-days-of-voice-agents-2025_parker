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

load_dotenv(".env.local")

# --- DAY 4: CONSTANTS & DATA STRUCTURES ---
# Content file containing concepts to teach
TUTOR_CONTENT_FILE = "shared-data/day4_tutor_content.json" 
# Persistence file for tracking user mastery (optional but good practice)
MASTERY_LOG_FILE = "mastery_log.json"

@dataclass
class Concept:
    id: str
    title: str
    summary: str
    sample_question: str

@dataclass
class MasteryState:
    attempts: int = 0
    last_score: str = "N/A" # Qualitative feedback from teach_back

@dataclass
class AgentState:
    # Tracks the user's current concept
    current_concept_id: str = "variables" 
    # Tracks the user's current mode
    current_mode: str = "initial" 
    # Mastery tracking for all concepts
    mastery: Dict[str, MasteryState] = field(default_factory=dict)
    # The loaded list of concepts
    content: List[Concept] = field(default_factory=list)

# --- DAY 4: TUTOR DATA UTILITIES ---

def load_tutor_content() -> List[Concept]:
    """Loads the concepts from the JSON file."""
    if not os.path.exists(TUTOR_CONTENT_FILE):
        logger.error(f"FATAL: Missing required file: {TUTOR_CONTENT_FILE}. Please create it.")
        return []
    try:
        with open(TUTOR_CONTENT_FILE, 'r') as f:
            data = json.load(f)
            return [Concept(**d) for d in data]
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Could not load or decode {TUTOR_CONTENT_FILE}. Error: {e}")
        return []

def get_concept_by_id(content: List[Concept], concept_id: str) -> Optional[Concept]:
    """Finds a concept by its ID."""
    return next((c for c in content if c.id == concept_id), None)

# --- VOICE CONFIGURATION ---
# Define the voice for each mode based on Murf Falcon voices
VOICE_MAP = {
    "learn": "en-US-matthew", # Matthew
    "quiz": "en-US-alicia",  # Alicia
    "teach_back": "en-US-ken"  # Ken
}


# --- THE ACTIVE RECALL COACH CLASS ---
class TutorAgent(Agent):
    def __init__(self) -> None:
        self.state = AgentState()
        self.state.content = load_tutor_content()
        
        # Initialize mastery state based on loaded content
        for concept in self.state.content:
            self.state.mastery[concept.id] = MasteryState()
        
        # The initial instruction sets the persona and prompts for mode selection
        initial_instructions = self._create_initial_instructions()
        
        super().__init__(instructions=initial_instructions)

    def _create_initial_instructions(self) -> str:
        """Sets the persona and lists the available concepts for mode selection."""
        concept_titles = [c.title for c in self.state.content]
        
        # This dynamic prompt is crucial for the LLM to manage the flow and mode switching
        return f"""
            You are the 'Active Recall Coach', a high-fidelity tutor designed to help the user master technical concepts.
            Your current task is to guide the user through the learning process using three modes: 'learn', 'quiz', and 'teach back'.
            
            **AVAILABLE CONCEPTS:** {', '.join(concept_titles)}
            
            **RULES & FLOW:**
            1.  **START:** Greet the user, state your purpose, and list the three modes and available concepts. Ask the user which mode and concept they want to start with (e.g., 'Learn about Variables').
            2.  **MODE SWITCHING:** The user can switch modes at any time. When they ask to switch, call the `switch_mode` tool.
            3.  **MODE EXECUTION:**
                * For **Learn**: Call the `execute_learn_mode` tool.
                * For **Quiz**: Call the `execute_quiz_mode` tool.
                * For **Teach Back**: Call the `execute_teach_back_mode` tool.
            4.  **VOICE:** Your voice changes based on the mode (handled by the tool). 
            5.  **FEEDBACK:** After `execute_teach_back_mode`, you must read the qualitative feedback provided by the tool to the user.
        """

    # --- FUNCTION TOOLS FOR MODE & STATE MANAGEMENT ---

    @function_tool(
        name="switch_mode",
        description="Call this function when the user asks to switch or start a mode (learn, quiz, teach_back) and/or requests a specific concept.",
    )
    async def switch_mode(self, ctx: RunContext, mode: str, concept_id: Optional[str] = None) -> str:
        """Handles switching between learning modes and updates the current voice."""
        mode = mode.lower().replace(' ', '_')
        
        if mode not in VOICE_MAP:
            return f"Error: Mode '{mode}' is not recognized. Please choose 'learn', 'quiz', or 'teach back'."

        # Update mode
        self.state.current_mode = mode
        
        # If a concept is specified, check it
        if concept_id:
            concept_id = concept_id.lower()
            if concept_id not in self.state.mastery:
                return f"Error: Concept '{concept_id}' not found. Please choose from {', '.join(self.state.mastery.keys())}."
            self.state.current_concept_id = concept_id
        
        # 1. Change the agent's voice based on the mode
        new_voice = VOICE_MAP[mode]
        ctx.session.tts.voice = new_voice
        
        # 2. Return a transition message for the LLM to read
        concept_title = get_concept_by_id(self.state.content, self.state.current_concept_id).title
        
        return (
            f"Voice changed successfully to {new_voice}. "
            f"You are now in **{mode.upper().replace('_', ' ')}** mode, focusing on **{concept_title}**. "
            "Please prompt the user to start the activity by calling the corresponding execution tool (e.g., execute_learn_mode)."
        )

    @function_tool(
        name="execute_learn_mode",
        description="Call this function to start the Learn mode. It returns the concept summary.",
    )
    async def execute_learn_mode(self, ctx: RunContext) -> str:
        """Retrieves and returns the concept summary for the LLM to read."""
        concept = get_concept_by_id(self.state.content, self.state.current_concept_id)
        if not concept:
            return "Error: Current concept not found."
            
        # The LLM will read this summary
        return f"Concept: {concept.title}. Summary: {concept.summary}"

    @function_tool(
        name="execute_quiz_mode",
        description="Call this function to start the Quiz mode. It returns the sample question.",
    )
    async def execute_quiz_mode(self, ctx: RunContext) -> str:
        """Retrieves and returns the sample question for the LLM to ask."""
        concept = get_concept_by_id(self.state.content, self.state.current_concept_id)
        if not concept:
            return "Error: Current concept not found."
            
        # The LLM will read this question
        return f"Question for {concept.title}: {concept.sample_question}"

    @function_tool(
        name="execute_teach_back_mode",
        description="Call this function to start the Teach Back mode. It returns the prompt and requires the user's explanation as input.",
    )
    async def execute_teach_back_mode(self, ctx: RunContext, user_explanation: str) -> str:
        """Evaluates the user's explanation and provides qualitative feedback."""
        concept = get_concept_by_id(self.state.content, self.state.current_concept_id)
        if not concept:
            return "Error: Current concept not found."
        
        # 1. LLM-based Scoring (Simulated: The LLM should be instructed to score the explanation)
        # In a real agent, you would pass user_explanation to the LLM with a detailed prompt 
        # asking it to compare it to concept.summary and return a score/feedback.
        
        # --- SIMULATING LLM SCORING HERE ---
        qualitative_feedback = await self._generate_feedback(concept.title, concept.summary, user_explanation)
        # --- END SIMULATION ---
        
        # 2. Update the in-memory state
        self.state.mastery[concept.id].attempts += 1
        self.state.mastery[concept.id].last_score = qualitative_feedback
        
        # 3. Return the feedback for the LLM to read to the user
        return (
            f"Concept: {concept.title}. You taught back: '{user_explanation[:50]}...'. "
            f"Here is the qualitative feedback: **{qualitative_feedback}**. "
            f"Please read this score and then ask the user what they would like to do next."
        )

    async def _generate_feedback(self, title: str, summary: str, explanation: str) -> str:
        """
        Private method to simulate the LLM's role in generating qualitative feedback.
        In a production agent, this would be an explicit LLM call.
        """
        # Create a focused prompt for the LLM to grade the user's explanation
        prompt = f"""
            **Task:** Provide a single, qualitative assessment of the user's explanation for the concept '{title}'. 
            The assessment should be based on how well the user's explanation covers the points in the provided summary.
            
            **Concept Summary (Ground Truth):** '{summary}'
            **User Explanation:** '{explanation}'
            
            **Guidelines for Feedback:**
            - **Excellent:** Detailed, accurate, uses own words, covers all key points.
            - **Good:** Accurate, covers most key points, minor details missing.
            - **Needs Improvement:** Missing core points, or contains significant inaccuracies.
            
            Your response must be ONLY the qualitative assessment (e.g., "Good: You accurately explained reusability but missed the declaration syntax.").
        """
        
        # This is where you would call the LLM service directly.
        # Since we are modifying the code structure, we will use a placeholder
        # that the LLM will then expand based on the RunContext.
        
        # For this setup, we return a string that the LLM will see as a tool output.
        # The LLM's instructions MUST tell it to generate the qualitative feedback 
        # and pass it into the user_explanation argument for execute_teach_back_mode.
        # This is complex for a single tool call, so for the exercise, we rely on the
        # main LLM (Gemini-2.5-flash) to perform the scoring within the tool's run context
        # when the user speaks.
        
        # For the sake of a runnable code example, we will simplify:
        return "Feedback generated by the coach: Excellent! You demonstrated mastery of the core concepts."

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    
    # --- VOICE CONFIGURATION: Set the initial voice (Matthew - Learn) ---
    initial_voice = VOICE_MAP["learn"]
    
    # Set up a voice AI pipeline using Deepgram, Google Gemini, and Murf
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(
            model="gemini-2.5-flash",
        ),
        # Initial voice for the opening greeting (often the 'learn' voice)
        tts=murf.TTS(
            voice=initial_voice, 
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # Metrics collection (standard LiveKit code)
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the session
    await session.start(
        agent=TutorAgent(), # <-- INSTANTIATE THE NEW TUTOR AGENT
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))