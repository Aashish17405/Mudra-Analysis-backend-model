import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

# Knowledge Base: Meanings of Bharatanatyam Steps
STEP_MEANINGS = {
    "Alarippu": "an invocation piece that symbolizes the blossoming of the dancer's body and mind, offering respects to God, the teacher, and the audience.",
    "Jathiswaram": "a pure dance sequence (Nritta) focusing on technical brilliance, complex footwork, and rhythm, without any specific interpretive meaning.",
    "Shabdam": "an interpretive dance (Abhinaya) where the dancer introduces expressions and simple storytelling, typically praising a deity or king.",
    "Varnam": "the centerpiece of the recital, effectively combining pure dance (Nritta) and expressive storytelling (Nritya) to explore complex emotional themes and devotion.",
    "Padam": "a deeply emotional and expressive piece (Nritya), focusing on slow tempo and subtle facial expressions to depict themes of love, longing, or devotion.",
    "Tillana": "a fast-paced, rhythmic conclusion to the recital, characterized by intricate footwork and statuesque poses, celebrating joy and dynamism."
}

def generate_storyline(timeline: List[Dict]) -> str:
    """
    Generates a narrative story based on the sequence of inferred steps.
    
    Args:
        timeline: List of dicts [{'step': 'Name', 'start_frame': ...}, ...]
        
    Returns:
        str: The generated story text.
    """
    if not timeline:
        return "The video does not appear to contain any recognizable dance steps."

    # Extract unique sequence of steps
    # Merge consecutive identical steps just in case, though inference logic handles it.
    unique_sequence_info = []
    last_step = None
    for item in timeline:
        step_name = item['step']
        meaning = item.get('meaning', STEP_MEANINGS.get(step_name, "a traditional dance sequence."))
        
        if step_name != last_step:
            unique_sequence_info.append((step_name, meaning))
            last_step = step_name
            
    story_parts = []
    
    # Intro
    story_parts.append(f"The performance captured in this video weaves a narrative through {len(unique_sequence_info)} distinct phases of Bharatanatyam.")
    
    # Body
    for idx, (step, meaning) in enumerate(unique_sequence_info):
        # meaning = STEP_MEANINGS.get(step, "a traditional dance sequence.")
        
        if idx == 0:
            story_parts.append(f"It begins with **{step}**, which is {meaning}")
        elif idx == len(unique_sequence_info) - 1:
            story_parts.append(f"Finally, the recital concludes with **{step}**, {meaning}")
        else:
            transition_words = ["Following this,", "The dancer then transitions into", "Next comes"]
            trans_word = transition_words[idx % len(transition_words)]
            story_parts.append(f"{trans_word} **{step}**, {meaning}")
            
    # Conclusion
    story_parts.append("\nOverall, this sequence demonstrates a classic progression from technical warmup to deep expression and rhythmic complexity.")
    
    full_story = "\n\n".join(story_parts)
    return full_story
