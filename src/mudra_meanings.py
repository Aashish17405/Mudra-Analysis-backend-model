"""
Mudra Knowledge Base
Comprehensive meanings for Bharatanatyam mudras.
"""

# Single-Hand Mudras (Asamyuktha Hasta)
MUDRA_MEANINGS = {
    # Asamyuktha Hastas (Single Hand Mudras)
    "Pataka": "Flag - represents cloud, forest, ocean, denial, or a sword",
    "Pathaka": "Flag - represents cloud, forest, ocean, denial, or a sword", # Alternate spelling
    "Tripataka": "Three parts of flag - represents a crown, tree, thunderbolt, or flame",
    "Ardhapataka": "Half flag - represents a leaf, peg, or knife",
    "Ardhapathaka": "Half flag - represents a leaf, peg, or knife",
    "Kartarimukha": "Scissors face - represents lightning, separation, or the beak of a bird",
    "Mayura": "Peacock - represents a peacock, applying tilak, or holding flowers",
    "Ardhachandra": "Half moon - represents the moon, meditation, or the phases of the moon",
    "Ardhachandran": "Half moon - represents the moon, meditation, or the phases of the moon",
    "Arala": "Bent - represents drinking poison, tasting nectar, or determination",
    "Aralam": "Bent - represents drinking poison, tasting nectar, or determination",
    "Shukatundaka": "Parrot's beak - represents shooting an arrow, whispering, or lifting,",
    "Shukatundam": "Parrot's beak - represents shooting an arrow, whispering, or lifting,",
    "Mushti": "Fist - represents firmness, steadiness, or holding something",
    "Shikhara": "Peak - represents Lord Shiva, lingam, or questioning",
    "Sikharam": "Peak - represents Lord Shiva, lingam, or questioning",
    "Kapitta": "Elephant apple - represents holding Lakshmi, goddess Saraswati",
    "Kapith": "Elephant apple - represents holding Lakshmi, goddess Saraswati",
    "Katakamukha": "Opening in a bracelet - represents picking flowers, holding a necklace",
    "Suchi": "Needle - represents one, unity, or pointing",
    "Chandrakala": "Digit of moon - represents the crescent moon, reading, or tracing",
    "Padmakosha": "Lotus bud - represents a lotus bud, fruit, or offering",
    "Sarpashirsha": "Snake's head - represents snakes, sprinkling water, or demons",
    "Sarpasirsha": "Snake's head - represents snakes, sprinkling water, or demons",
    "Mrigashirsha": "Deer's head - represents calling, fear, affection, or shyness",
    "Mrigasirsha": "Deer's head - represents calling, fear, affection, or shyness",
    "Simhamukha": "Lion's face - represents perfume vessel, lion, or Ganesha",
    "Simhamukham": "Lion's face - represents perfume vessel, lion, or Ganesha",
    "Kangula": "Tail - represents Goddess Saraswati, plucking flowers",
    "Kangulam": "Tail - represents Goddess Saraswati, plucking flowers",
    "Alapadma": "Fully bloomed lotus - represents lotus, teaching, giving, receiving",
    "Alapadmam": "Fully bloomed lotus - represents lotus, teaching, giving, receiving",
    "Chatura": "Four - represents the number four, picking up, or holding",
    "Chaturam": "Four - represents the number four, picking up, or holding",
    "Bhramara": "Bee - represents a bee, union of Shiva and Shakti",
    "Bramaram": "Bee - represents a bee, union of Shiva and Shakti",
    "Hamsasya": "Swan's head - represents a swan, beauty, grace",
    "Hamsasyam": "Swan's head - represents a swan, beauty, grace",
    "Hamsapaksha": "Swan's wing - represents marking tilak, massage",
    "Sandamsha": "Tongs - represents picking up things, holding arrows",
    "Mukula": "Bud - represents a flower bud, food items, combining elements",
    "Mukulam": "Bud - represents a flower bud, food items, combining elements",
    "Tamrachuda": "Rooster - represents a rooster, hen, writing",
    "Tamarachudam": "Rooster - represents a rooster, hen, writing",
    "Trishula": "Trident - represents Lord Shiva's weapon, the number three",
    "Trishulam": "Trident - represents Lord Shiva's weapon, the number three",
    
    # Samyuktha Hastas (Double Hand Mudras)
    "Anjali": "Prayer - represents prayer, offering, salutation",
    "Kapota": "Dove - represents embracing, carrying child, support",
    "Kapotham": "Dove - represents embracing, carrying child, support",
    "Karkata": "Crab - represents Yama (god of death), crab, prison",
    "Karkatta": "Crab - represents Yama (god of death), crab, prison",
    "Swastika": "Auspicious - represents being firm, rigid devotion",
    "Swastikam": "Auspicious - represents being firm, rigid devotion",
    "Dola": "Swing - represents swinging, sporting in water",
    "Pushpaputa": "Handful of flowers - represents offering flowers, giving",
    "Utsanga": "Embrace - represents resting on the lap, sitting",
    "Shivalinga": "Lord Shiva's icon - represents lingam, union",
    "Katakavardana": "Bracelet chain - represents wearing garlands, decorating",
    "Katakavardhana": "Bracelet chain - represents wearing garlands, decorating",
    "Kartariswastika": "Crossed scissors - represents rejection, denial, prohibition",
    "Shakata": "Cart - represents cart, bed, moving",
    "Sakata": "Cart - represents cart, bed, moving",
    "Shankha": "Conch - represents playing conch, conch shell",
    "Shanka": "Conch - represents playing conch, conch shell",
    "Chakra": "Wheel - represents Vishnu's disc, wheel of time",
    "Garuda": "Eagle - represents Garuda bird, flight",
    "Pasha": "Noose - represents bondage, quarrel, noose",
    "Kilaka": "Bond - represents deep affection, conversation",
    "Matsya": "Fish - represents fish, Matsya avatar of Vishnu",
    "Kurma": "Tortoise - represents tortoise, Kurma avatar",
    "Varaha": "Boar - represents boar, Varaha avatar",
    "Nagabandha": "Serpent tie - represents snakes",
    "Khatva": "Bed - represents bed, palanquin",
    "Berunda": "Double-headed bird - represents bird pair",
    "Samputa": "Casket - represents keeping secrets, box"
}

# Emotional context for mudras
MUDRA_EMOTIONS = {
    "Pataka": "neutral",
    "Anjali": "devotional",
    "Mayura": "joyful",
    "Simhamukha": "powerful",
    "Bhramara": "romantic",
    "Ardhachandra": "serene",
    "Mushti": "determined",
    "Alapadma": "graceful"
}

# Category mapping
MUDRA_CATEGORIES = {
    "single_hand": [
        "Pataka", "Tripataka", "Ardhapataka", "Kartarimukha", "Mayura",
        "Ardhachandra", "Arala", "Shukatundaka", "Mushti", "Shikhara",
        "Kapitta", "Katakamukha", "Suchi", "Chandrakala", "Padmakosha",
        "Sarpashirsha", "Mrigashirsha", "Simhamukha", "Kangula", "Alapadma",
        "Chatura", "Bhramara", "Hamsasya", "Hamsapaksha", "Sandamsha",
        "Mukula", "Tamrachuda", "Trishula"
    ],
    "double_hand": [
        "Anjali", "Kapota", "Karkata", "Swastika", "Dola",
        "Pushpaputa", "Utsanga", "Shivalinga", "Katakavardana", "Kartariswastika",
        "Shakata", "Shankha", "Chakra", "Samputa", "Pasha",
        "Kilaka", "Matsya", "Kurma", "Varaha", "Garuda",
        "Nagabandha", "Khatva"
    ]
}


def get_mudra_meaning(mudra_name: str) -> str:
    """Get the meaning of a mudra."""
    return MUDRA_MEANINGS.get(mudra_name, "A traditional hand gesture with symbolic meaning.")


def get_mudra_category(mudra_name: str) -> str:
    """Get the category of a mudra."""
    if mudra_name in MUDRA_CATEGORIES["single_hand"]:
        return "Asamyuktha Hasta (Single Hand)"
    elif mudra_name in MUDRA_CATEGORIES["double_hand"]:
        return "Samyuktha Hasta (Double Hand)"
    else:
        return "Unknown Category"


def generate_mudra_narrative(mudra_detections: list) -> str:
    """
    Generate narrative from detected mudras.
    
    Args:
        mudra_detections: List of {'frame': int, 'mudra': str, 'confidence': float}
    
    Returns:
        str: Narrative description
    """
    if not mudra_detections:
        return "No mudras were detected in this performance."
    
    from collections import Counter
    
    # Count mudra occurrences
    mudra_counts = Counter([m['label'] for m in mudra_detections])
    
    narrative_parts = []
    narrative_parts.append(f"Throughout the performance, {len(mudra_detections)} mudra gestures were identified.")
    
    # Top 5 mudras
    top_mudras = mudra_counts.most_common(5)
    
    if top_mudras:
        narrative_parts.append("\nThe most prominent mudras were:")
        for idx, (mudra, count) in enumerate(top_mudras, 1):
            meaning = get_mudra_meaning(mudra)
            category = get_mudra_category(mudra)
            percentage = (count / len(mudra_detections)) * 100
            narrative_parts.append(
                f"{idx}. **{mudra}** ({category}) - appeared {count} times ({percentage:.1f}%), "
                f"representing {meaning}"
            )
    
    # Categorize by type
    single_hand_count = sum(1 for m in mudra_detections 
                           if m['label'] in MUDRA_CATEGORIES['single_hand'])
    double_hand_count = sum(1 for m in mudra_detections 
                           if m['label'] in MUDRA_CATEGORIES['double_hand'])
    
    if single_hand_count and double_hand_count:
        ratio = single_hand_count / double_hand_count
        narrative_parts.append(
            f"\nThe performance showed a balance of Asamyuktha (single-hand, {single_hand_count}) "
            f"and Samyuktha (double-hand, {double_hand_count}) mudras, "
            f"demonstrating versatility in expression."
        )
    
    return "\n".join(narrative_parts)
