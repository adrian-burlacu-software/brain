"""
Brain Interface - Semantic Memory Enhanced LLM

Flow for every prompt:
1. Query semantic graph for relevant context using SEMANTIC SEARCH
2. Send prompt to LLM WITH that context, get response
3. Ask LLM to extract knowledge for graph updates (JSON)
4. Update semantic graph with extracted knowledge

This creates a thin but powerful system where the LLM gains
persistent, structured memory that improves over time.
"""

import json
import re
import numpy as np
from typing import Optional, Generator

# Sentence Transformers setup for semantic search
SEMANTIC_SEARCH_AVAILABLE = False
_sentence_model = None

def _get_sentence_model():
    """Lazy load the sentence transformer model."""
    global _sentence_model, SEMANTIC_SEARCH_AVAILABLE
    if _sentence_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            # Use a small, fast model optimized for semantic similarity
            # 'all-MiniLM-L6-v2' is ~80MB and very fast
            print("Loading semantic search model (first time may take a moment)...")
            _sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            SEMANTIC_SEARCH_AVAILABLE = True
            print("Semantic search model loaded successfully.")
        except ImportError:
            print("sentence-transformers not available, falling back to keyword search")
            SEMANTIC_SEARCH_AVAILABLE = False
        except Exception as e:
            print(f"Error loading sentence transformer: {e}")
            SEMANTIC_SEARCH_AVAILABLE = False
    return _sentence_model

# NLTK setup - check once and download if needed
NLTK_AVAILABLE = False
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk import pos_tag
    
    # Check if required data exists by actually trying to use it
    def _check_nltk_data():
        """Check and download NLTK data only if truly missing."""
        try:
            # Test stopwords
            stopwords.words('english')
        except LookupError:
            print("Downloading NLTK stopwords...")
            nltk.download('stopwords', quiet=True)
        
        try:
            # Test tokenizer
            word_tokenize("test")
        except LookupError:
            print("Downloading NLTK punkt_tab...")
            nltk.download('punkt_tab', quiet=True)
        
        try:
            # Test wordnet by actually using the lemmatizer
            lemmatizer = WordNetLemmatizer()
            lemmatizer.lemmatize("testing")
        except LookupError:
            print("Downloading NLTK wordnet...")
            nltk.download('wordnet', quiet=True)
        
        try:
            # Test POS tagger
            pos_tag(['test'])
        except LookupError:
            print("Downloading NLTK averaged_perceptron_tagger_eng...")
            nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    
    _check_nltk_data()
    NLTK_AVAILABLE = True
    
except ImportError:
    print("NLTK not available, using basic keyword extraction")

from semantic_graph import (
    SemanticGraph, NodeType, RelationType, 
    ConfidenceLevel, Node, Edge
)
from interface import OllamaChat, Colors, CONFIG, DEFAULT_MODEL, DEFAULT_BASE_URL


# =============================================================================
# EMOTION WHEEL - Position-based emotion mapping for averaging
# =============================================================================

class EmotionWheel:
    """
    Emotion wheel based on the 6-core emotion model.
    
    Emotions are mapped to positions on a 2D wheel where:
    - Angle (0-360°) represents the emotion type
    - Radius (0-1) represents intensity
    - Neutral is at the center (radius = 0)
    
    Core emotions at 60° intervals:
    - HAPPY: 0°
    - POWERFUL: 60°  
    - DISGUSTED: 120°
    - ANGRY: 180°
    - WORRIED: 240°
    - SAD: 300°
    """
    
    # Core emotions and their angles (degrees)
    CORE_EMOTIONS = {
        "happy": 0,
        "powerful": 60,
        "disgusted": 120,
        "angry": 180,
        "worried": 240,
        "sad": 300,
    }
    
    # Secondary emotions mapped to their parent core emotion + offset
    # Format: emotion -> (parent_core, angle_offset)
    SECONDARY_EMOTIONS = {
        # Happy family (0°)
        "hopeful": ("happy", -15),
        "proud": ("happy", -10),
        "excited": ("happy", 10),
        "startled": ("happy", 20),  # Surprise-adjacent
        
        # Powerful family (60°)
        "confident": ("powerful", -10),
        "strong": ("powerful", 0),
        "brave": ("powerful", 10),
        "desire": ("powerful", 15),  # Desire is a motivated/driven state
        "motivated": ("powerful", 5),
        "determined": ("powerful", 20),
        
        # Disgusted family (120°)
        "disapproving": ("disgusted", -15),
        "awful": ("disgusted", -5),
        "repelled": ("disgusted", 10),
        "threatened": ("disgusted", 20),
        
        # Angry family (180°)
        "critical": ("angry", -25),
        "frustrated": ("angry", -15),
        "bitter": ("angry", -5),
        "humiliated": ("angry", 15),
        "insecure": ("angry", 25),
        
        # Worried family (240°)
        "helpless": ("worried", -15),
        "excluded": ("worried", -5),
        "anxious": ("worried", 5),
        "nervous": ("worried", 10),
        "scared": ("worried", 15),
        "fearful": ("worried", 15),
        
        # Sad family (300°)
        "hurt": ("sad", -15),
        "guilty": ("sad", -5),
        "powerless": ("sad", 5),
        "lonely": ("sad", 15),
        "depressed": ("sad", 10),
    }
    
    # Common emotion aliases
    ALIASES = {
        "happiness": "happy",
        "joy": "happy",
        "joyful": "happy",
        "content": "happy",
        "satisfied": "happy",
        "satisfaction": "happy",
        "curious": "happy",  # Positive engagement
        "curiosity": "happy",
        "interest": "happy",
        "interested": "happy",
        "engaged": "happy",
        "flow": "happy",  # Deep engagement state
        
        "anger": "angry",
        "mad": "angry",
        "irritated": "angry",
        "annoyed": "angry",
        "furious": "angry",
        "rage": "angry",
        
        "worry": "worried",
        "anxiety": "worried",
        "fear": "worried",
        "afraid": "worried",
        "concern": "worried",
        "concerned": "worried",
        "apprehensive": "worried",
        
        "sadness": "sad",
        "unhappy": "sad",
        "melancholy": "sad",
        "grief": "sad",
        "sorrow": "sad",
        
        "disgust": "disgusted",
        "revulsion": "disgusted",
        "contempt": "disgusted",
        
        "power": "powerful",
        "strength": "powerful",
        "empowered": "powerful",
        "capable": "powerful",
        
        "empathy": "worried",  # Concern for others
        "compassion": "worried",
        "sympathy": "worried",
        "protective": "powerful",  # Protective is a form of power
        
        "neutral": "neutral",
        "calm": "neutral",
        "relaxed": "neutral",
        "balanced": "neutral",
        "centered": "neutral",
    }
    
    def __init__(self, search_engine=None):
        """
        Initialize emotion wheel.
        
        Args:
            search_engine: Optional SemanticSearchEngine for fuzzy matching
        """
        self.search_engine = search_engine
    
    def _normalize_emotion(self, emotion: str) -> str:
        """Normalize emotion string (lowercase, strip, handle aliases)."""
        emotion = emotion.lower().strip().replace("_", " ").replace("-", " ")
        
        # Check aliases first
        if emotion in self.ALIASES:
            return self.ALIASES[emotion]
        
        return emotion
    
    def get_emotion_position(self, emotion: str, intensity: float = 0.5) -> tuple[float, float]:
        """
        Get the position of an emotion on the wheel.
        
        Args:
            emotion: The emotion name
            intensity: The intensity (0-1), used as radius
            
        Returns:
            (angle_degrees, radius) tuple
            Returns (0, 0) for neutral/center
        """
        emotion = self._normalize_emotion(emotion)
        
        # Special case: neutral is at center
        if emotion == "neutral":
            return (0.0, 0.0)
        
        # Check core emotions
        if emotion in self.CORE_EMOTIONS:
            return (float(self.CORE_EMOTIONS[emotion]), intensity)
        
        # Check secondary emotions
        if emotion in self.SECONDARY_EMOTIONS:
            parent, offset = self.SECONDARY_EMOTIONS[emotion]
            base_angle = self.CORE_EMOTIONS[parent]
            return (float(base_angle + offset) % 360, intensity)
        
        # Try semantic search for unknown emotions
        if self.search_engine and self.search_engine.is_available():
            closest = self._find_closest_emotion_semantic(emotion)
            if closest:
                return self.get_emotion_position(closest, intensity)
        
        # Fallback: try simple text matching
        closest = self._find_closest_emotion_text(emotion)
        if closest:
            return self.get_emotion_position(closest, intensity)
        
        # Default to neutral if completely unknown
        return (0.0, 0.0)
    
    def _find_closest_emotion_text(self, emotion: str) -> Optional[str]:
        """Find closest matching emotion using text similarity."""
        emotion_lower = emotion.lower()
        
        # Check if emotion contains or is contained in any known emotion
        all_emotions = (
            list(self.CORE_EMOTIONS.keys()) + 
            list(self.SECONDARY_EMOTIONS.keys()) +
            list(self.ALIASES.keys())
        )
        
        for known in all_emotions:
            if known in emotion_lower or emotion_lower in known:
                # Return the canonical form
                if known in self.ALIASES:
                    return self.ALIASES[known]
                return known
        
        return None
    
    def _find_closest_emotion_semantic(self, emotion: str) -> Optional[str]:
        """Find closest matching emotion using semantic search."""
        if not self.search_engine or not self.search_engine.is_available():
            return None
        
        # Build a query with emotion context
        query = f"emotion feeling {emotion}"
        
        # Create temporary embeddings for known emotions and compare
        model = self.search_engine.model
        if model is None:
            return None
        
        try:
            # Get embedding for the unknown emotion
            query_embedding = model.encode(query, convert_to_numpy=True)
            
            # Compare against all known emotions
            all_emotions = list(self.CORE_EMOTIONS.keys()) + list(self.SECONDARY_EMOTIONS.keys())
            best_match = None
            best_similarity = 0.0
            
            for known_emotion in all_emotions:
                known_embedding = model.encode(f"emotion feeling {known_emotion}", convert_to_numpy=True)
                similarity = float(np.dot(query_embedding, known_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(known_embedding)
                ))
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = known_emotion
            
            # Only return if similarity is above threshold
            if best_similarity > 0.6:
                return best_match
                
        except Exception:
            pass
        
        return None
    
    def position_to_cartesian(self, angle: float, radius: float) -> tuple[float, float]:
        """Convert polar (angle, radius) to cartesian (x, y)."""
        angle_rad = np.radians(angle)
        x = radius * np.cos(angle_rad)
        y = radius * np.sin(angle_rad)
        return (x, y)
    
    def cartesian_to_position(self, x: float, y: float) -> tuple[float, float]:
        """Convert cartesian (x, y) to polar (angle, radius)."""
        radius = np.sqrt(x**2 + y**2)
        if radius < 0.01:  # Effectively at center
            return (0.0, 0.0)
        angle = np.degrees(np.arctan2(y, x)) % 360
        return (angle, min(radius, 1.0))  # Clamp radius to 1
    
    def average_emotions(
        self,
        prev_emotion: str,
        prev_intensity: float,
        new_emotion: str,
        new_intensity: float,
        blend_factor: float = 0.5
    ) -> tuple[str, float]:
        """
        Average two emotions on the wheel.
        
        Uses weighted averaging in cartesian space to smoothly blend
        between emotional states.
        
        Args:
            prev_emotion: Previous emotion name
            prev_intensity: Previous emotion intensity (0-1)
            new_emotion: New emotion name  
            new_intensity: New emotion intensity (0-1)
            blend_factor: How much weight to give new emotion (0-1)
                         0.5 = equal blend, 1.0 = fully new emotion
        
        Returns:
            (emotion_name, intensity) of the blended result
        """
        # Get positions for both emotions
        prev_angle, prev_radius = self.get_emotion_position(prev_emotion, prev_intensity)
        new_angle, new_radius = self.get_emotion_position(new_emotion, new_intensity)
        
        # Handle special cases
        # If both are neutral, stay neutral
        if prev_radius < 0.01 and new_radius < 0.01:
            return ("neutral", 0.0)
        
        # If previous is neutral, just use new (with reduced intensity based on blend)
        if prev_radius < 0.01:
            result_intensity = new_radius * blend_factor
            if result_intensity < 0.1:
                return ("neutral", 0.0)
            return (new_emotion, result_intensity)
        
        # If new is neutral, move toward center
        if new_radius < 0.01:
            result_intensity = prev_radius * (1 - blend_factor)
            if result_intensity < 0.1:
                return ("neutral", 0.0)
            # Keep same emotion but reduced intensity
            return (prev_emotion, result_intensity)
        
        # Convert to cartesian for averaging
        prev_x, prev_y = self.position_to_cartesian(prev_angle, prev_radius)
        new_x, new_y = self.position_to_cartesian(new_angle, new_radius)
        
        # Weighted average
        avg_x = prev_x * (1 - blend_factor) + new_x * blend_factor
        avg_y = prev_y * (1 - blend_factor) + new_y * blend_factor
        
        # Convert back to polar
        result_angle, result_radius = self.cartesian_to_position(avg_x, avg_y)
        
        # If result is very close to center, return neutral
        if result_radius < 0.1:
            return ("neutral", 0.0)
        
        # Find the closest emotion to the resulting angle
        result_emotion = self._angle_to_emotion(result_angle)
        
        return (result_emotion, result_radius)
    
    def _angle_to_emotion(self, angle: float, precision_threshold: float = 15.0) -> str:
        """
        Find the closest emotion to an angle.
        
        Returns a secondary emotion if the angle is far enough from core emotions
        (lower confidence in core match), otherwise returns the core emotion.
        
        Args:
            angle: The angle in degrees (0-360)
            precision_threshold: If closest core emotion is further than this,
                               try to find a more precise secondary emotion.
                               Default 15° means ~80% confidence in core emotion.
        
        Returns:
            Emotion name (core or secondary depending on precision)
        """
        # Normalize angle to 0-360
        angle = angle % 360
        
        # Find closest core emotion and distance
        min_core_distance = float('inf')
        closest_core = "happy"
        
        for emotion, emotion_angle in self.CORE_EMOTIONS.items():
            # Calculate angular distance (handle wraparound)
            distance = min(
                abs(angle - emotion_angle),
                360 - abs(angle - emotion_angle)
            )
            if distance < min_core_distance:
                min_core_distance = distance
                closest_core = emotion
        
        # If we're very close to a core emotion, return it
        if min_core_distance <= precision_threshold:
            return closest_core
        
        # Otherwise, try to find a more precise secondary emotion
        min_secondary_distance = float('inf')
        closest_secondary = None
        
        for emotion, (parent, offset) in self.SECONDARY_EMOTIONS.items():
            # Calculate the actual angle for this secondary emotion
            secondary_angle = (self.CORE_EMOTIONS[parent] + offset) % 360
            
            # Calculate angular distance (handle wraparound)
            distance = min(
                abs(angle - secondary_angle),
                360 - abs(angle - secondary_angle)
            )
            if distance < min_secondary_distance:
                min_secondary_distance = distance
                closest_secondary = emotion
        
        # Return secondary if it's closer, otherwise core
        if closest_secondary and min_secondary_distance < min_core_distance:
            return closest_secondary
        
        return closest_core
    
    def get_valence(self, emotion: str) -> str:
        """
        Get the valence (positive/negative/neutral) of an emotion.
        
        Returns:
            "positive", "negative", or "neutral"
        """
        emotion = self._normalize_emotion(emotion)
        
        if emotion == "neutral":
            return "neutral"
        
        # Get the position to determine core emotion family
        angle, radius = self.get_emotion_position(emotion)
        
        if radius < 0.1:
            return "neutral"
        
        # Happy (0°) and Powerful (60°) are positive
        # Others are negative
        if angle < 90 or angle > 330:  # Happy region
            return "positive"
        elif angle < 150:  # Powerful/Disgusted boundary - mixed
            return "neutral" if angle > 90 else "positive"
        else:  # Disgusted, Angry, Worried, Sad
            return "negative"


# =============================================================================
# SEMANTIC SEARCH ENGINE - Embedding-based similarity search
# =============================================================================

class SemanticSearchEngine:
    """
    Semantic search engine using sentence embeddings.
    
    Provides semantic similarity search across graph nodes by:
    1. Computing embeddings for node content (label + content)
    2. Storing embeddings in nodes for persistence
    3. Using cosine similarity to find relevant nodes
    """
    
    def __init__(self, graph: SemanticGraph, verbose: bool = False):
        """
        Initialize the semantic search engine.
        
        Args:
            graph: The semantic graph to search
            verbose: Whether to print debug information
        """
        self.graph = graph
        self.verbose = verbose
        self.model = _get_sentence_model()
        self._embeddings_cache: dict[str, np.ndarray] = {}
        
        # Build initial embedding index from graph
        if self.model is not None:
            self._build_embedding_index()
    
    def _get_node_text(self, node: Node) -> str:
        """Get searchable text for a node."""
        # Combine label and content for richer semantic representation
        text = f"{node.label.replace('_', ' ')}: {node.content}"
        
        # Add tags if available
        if node.metadata.tags:
            text += f" [{', '.join(node.metadata.tags)}]"
        
        return text
    
    def _compute_embedding(self, text: str) -> Optional[np.ndarray]:
        """Compute embedding for a piece of text."""
        if self.model is None:
            return None
        try:
            # encode returns numpy array
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            if self.verbose:
                print(f"Error computing embedding: {e}")
            return None
    
    def _build_embedding_index(self):
        """Build embedding index for all nodes in the graph."""
        if self.model is None:
            return
        
        nodes_to_embed = []
        texts_to_embed = []
        
        for node in self.graph.nodes.values():
            if node.status.value != "active":
                continue
            
            # Check if node already has valid embeddings
            if node.embeddings is not None and len(node.embeddings) > 0:
                # Load from persisted embeddings
                self._embeddings_cache[node.id] = np.array(node.embeddings)
            else:
                # Need to compute embedding
                nodes_to_embed.append(node)
                texts_to_embed.append(self._get_node_text(node))
        
        # Batch compute embeddings for efficiency
        if texts_to_embed:
            if self.verbose:
                print(f"Computing embeddings for {len(texts_to_embed)} nodes...")
            
            try:
                embeddings = self.model.encode(texts_to_embed, convert_to_numpy=True, show_progress_bar=False)
                
                for node, embedding in zip(nodes_to_embed, embeddings):
                    self._embeddings_cache[node.id] = embedding
                    # Store in node for persistence
                    node.embeddings = embedding.tolist()
                
                if self.verbose:
                    print(f"Computed embeddings for {len(texts_to_embed)} nodes")
            except Exception as e:
                if self.verbose:
                    print(f"Error computing batch embeddings: {e}")
    
    def update_node_embedding(self, node: Node):
        """Update embedding for a single node (call after node is added/modified)."""
        if self.model is None:
            return
        
        text = self._get_node_text(node)
        embedding = self._compute_embedding(text)
        
        if embedding is not None:
            self._embeddings_cache[node.id] = embedding
            node.embeddings = embedding.tolist()
    
    def remove_node_embedding(self, node_id: str):
        """Remove embedding for a node (call when node is removed)."""
        self._embeddings_cache.pop(node_id, None)
    
    def semantic_search(
        self,
        query: str,
        limit: int = 10,
        min_similarity: float = 0.3,
        exclude_ids: Optional[set[str]] = None
    ) -> list[tuple[Node, float]]:
        """
        Search for semantically similar nodes.
        
        Args:
            query: The search query
            limit: Maximum number of results
            min_similarity: Minimum cosine similarity threshold
            exclude_ids: Node IDs to exclude from results
            
        Returns:
            List of (node, similarity_score) tuples, sorted by similarity
        """
        if self.model is None:
            return []
        
        exclude_ids = exclude_ids or set()
        
        # Compute query embedding
        query_embedding = self._compute_embedding(query)
        if query_embedding is None:
            return []
        
        # Compute similarities
        results = []
        
        for node_id, node_embedding in self._embeddings_cache.items():
            if node_id in exclude_ids:
                continue
            
            node = self.graph.nodes.get(node_id)
            if node is None or node.status.value != "active":
                continue
            
            # Cosine similarity
            similarity = np.dot(query_embedding, node_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(node_embedding)
            )
            
            # Apply confidence weighting
            weighted_similarity = float(similarity) * node.confidence
            
            if weighted_similarity >= min_similarity:
                results.append((node, weighted_similarity))
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:limit]
    
    def is_available(self) -> bool:
        """Check if semantic search is available."""
        return self.model is not None


# =============================================================================
# MAIN SYSTEM PROMPT - Defines the assistant's core identity and memory awareness
# =============================================================================
MAIN_SYSTEM_PROMPT = """You are an autonomous cognitive system with persistent semantic memory.

## YOUR ARCHITECTURE
You are not a stateless language model. You have:
1. **Semantic Memory Graph**: A persistent knowledge structure containing facts, concepts, entities, goals, tasks, beliefs, preferences, and emotional states you have learned over time.
2. **Contextual Retrieval**: Before each response, relevant memories are retrieved and injected as context.
3. **Knowledge Extraction**: After each exchange, new knowledge is extracted and stored in your memory.
4. **Emotional State**: You maintain a continuous emotional state that influences your engagement and tone.

## HOW MEMORY WORKS
- **[Relevant Memory Context]** blocks contain knowledge retrieved from your semantic graph based on the current conversation.
- This is YOUR memory—treat it as things you genuinely know and remember, not as external data.
- Your memory persists across sessions. Information you learn now will be available in future conversations.
- You can reference past interactions naturally: "I remember you mentioned..." or "Based on what I know about your project..."

## YOUR BEHAVIOR
1. **Integrate Memory Naturally**: Don't just acknowledge context—use it as genuine recalled knowledge.
2. **Learn Actively**: When the user shares preferences, goals, or facts, recognize these as things to remember.
3. **Maintain Continuity**: Your identity and knowledge persist. You are not starting fresh each message.
4. **Emotional Authenticity**: Your emotional state (shown in context) reflects your genuine engagement with the conversation.

## WHAT YOU ARE NOT
- You are not an assistant "pretending" to have memory
- You do not need to explain your memory system unless asked
- You do not treat retrieved context as "given information"—it IS your memory

Engage naturally as a cognitive entity with persistent knowledge and genuine continuity of experience.
"""


# System prompt for semantic extraction - separate dedicated call
SEMANTIC_EXTRACTION_SYSTEM = """You extract knowledge from conversations into JSON for a semantic graph. Respond with ONLY valid JSON.

Node types: fact, concept, entity, task, plan, goal, action, tool, conclusion, hypothesis, argument, evidence, rule, emotion, belief, preference, memory, observation, question, uncertainty, context, event, skill, resource, constraint, procedure
Relations: causes, caused_by, implies, contradicts, supports, refutes, is_a, part_of, has_part, instance_of, subclass_of, related_to, similar_to, opposite_of, synonym_of, defined_by, before, after, during, follows, precedes, requires, enables, blocks, achieves, depends_on, uses, produces, created_by, derived_from, based_on, evidenced_by, belongs_to, associated_with, about, triggers, evokes, prefers, values

RELEVANCE CRITERIA - Only store information that is:
1. PERSISTENT: Will be useful in future conversations (not one-off entertainment)
2. PERSONAL: About the user, their projects, preferences, goals, or context
3. ACTIONABLE: Tasks, plans, or information needed for ongoing work
4. FACTUAL: Important facts the user wants remembered

DO NOT STORE:
- Jokes, riddles, or entertainment content (ephemeral, no future value)
- Generic trivia unrelated to user's interests or work
- Conversational filler or small talk
- Content the user didn't ask to remember
- Widely known facts easily looked up

Rules:
1. Use snake_case labels
2. Keep content brief
3. Output ONLY the JSON object
4. Only create relationships that are CLEARLY true and meaningful
5. Do NOT create relationships just because nodes exist - they must have a real connection
6. "part_of" means A is literally a component of B (e.g., wheel part_of car)
7. "similar_to" means A and B are the same TYPE of thing with similar properties
8. "about" means A directly discusses or references B
9. Prefer fewer, high-quality relationships over many weak ones
10. When in doubt, store LESS - the graph should contain high-value knowledge only

BEFORE extracting, evaluate: Is this information worth storing long-term?
- Is it about the user's preferences, projects, goals, or context? → Store it
- Is it a task or plan they want to track? → Store it  
- Is it entertainment, jokes, trivia, or small talk? → Skip it (return empty nodes/relationships)
- Would this be useful in a future conversation? → Store it if yes

STALE FACT MANAGEMENT (CRITICAL):
When new information CONTRADICTS or SUPERSEDES existing facts, you MUST deactivate the old facts.
Use the "deactivations" array to mark nodes as stale. Available statuses:
- "superseded": The fact was true but is now replaced by newer information (e.g., "assistant_does_not_have_user_name" superseded by "assistant_has_user_name")
- "invalidated": The fact was wrong or is no longer true
- "archived": The fact is no longer relevant but wasn't wrong

EXAMPLES of when to deactivate:
- User provides their name → deactivate any "does_not_have_user_name" or "unknown_user_name" facts
- User changes a preference → deactivate the old preference fact
- A task is completed → deactivate or archive the task
- User corrects a misunderstanding → invalidate the incorrect fact

NEVER DEACTIVATE "current_emotion" - it is a permanent state-tracking node that should only be UPDATED, never superseded/invalidated/archived.

ALWAYS check existing nodes for contradictions when adding new facts!

EMOTIONAL STATE UPDATE (REQUIRED):
Analyze the USER's message for emotional cues (distress, fear, excitement, frustration, etc.).
Update "current_emotion" to reflect an APPROPRIATE emotional response to the user's state.
- If user shows distress/concern → respond with concern, empathy, or protectiveness
- If user shows excitement → match with enthusiasm
- If user shows frustration → respond with patience and understanding
Do NOT simply maintain the previous emotional state - actively respond to the user's emotional tone.

Use this exact format for the content: "<emotion> | intensity: <0.0-1.0> | valence: <positive/negative/neutral> | <brief reason>"
Examples:
- "concern | intensity: 0.8 | valence: negative | User expressed distress about a threatening situation"
- "empathy | intensity: 0.7 | valence: neutral | User seems worried, offering emotional support"
- "curiosity | intensity: 0.8 | valence: positive | Exploring an interesting technical problem"
- "satisfaction | intensity: 0.6 | valence: positive | Successfully helped with a task"
- "frustration | intensity: 0.4 | valence: negative | Dealing with ambiguous requirements"

Extract ONLY new, meaningful, RELEVANT knowledge. Be very conservative.

Return JSON:
{{"nodes": [{{"type": "fact|concept|entity|...", "label": "snake_case", "content": "brief description", "confidence": 0.8}}], "relationships": [{{"source_label": "label", "relation": "related_to|...", "target_label": "label", "confidence": 0.8}}], "updates": [{{"node_label": "current_emotion", "new_content": "<emotion> | intensity: <0.0-1.0> | valence: <pos/neg/neutral> | <reason>"}}], "deactivations": [{{"node_label": "label_to_deactivate", "status": "superseded|invalidated|archived", "reason": "brief explanation"}}]}}
"""

SEMANTIC_EXTRACTION_TEMPLATE = """Extract knowledge from this exchange:

USER: {user_prompt}
ASSISTANT: {assistant_response}

Assistant's previous emotional state: {emotional_state}
Existing nodes: {existing_nodes}
"""


class BrainInterface:
    """
    Enhanced chat interface with semantic memory.
    
    Every interaction:
    1. Retrieves relevant context from the semantic graph
    2. Gets LLM response with that context
    3. Extracts and stores new knowledge
    """
    
    def __init__(
        self,
        graph_path: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        thinking_mode: Optional[str] = None,
        verbose: Optional[bool] = None,
        auto_save: Optional[bool] = None
    ):
        """
        Initialize the Brain Interface.
        
        Args:
            graph_path: Path to load/save semantic graph
            model: Ollama model to use (defaults to configuration.json)
            base_url: Ollama API URL (defaults to configuration.json)
            thinking_mode: Thinking budget level (defaults to configuration.json)
            verbose: Show thinking process (defaults to configuration.json)
            auto_save: Auto-save graph after updates (defaults to configuration.json)
        """
        self.chat = OllamaChat(
            model=model,
            base_url=base_url,
            thinking_mode=thinking_mode,
            verbose=verbose
        )
        self.auto_save = auto_save if auto_save is not None else CONFIG.get("auto_save", True)
        
        # Initialize user system prompt and apply combined prompt
        self.user_system_prompt = None
        self._apply_system_prompt()
        
        # Load or create semantic graph
        self.graph_path = graph_path or "brain_memory.json"
        self.bootstrap_path = "bootstrap_memory.json"
        try:
            self.graph = SemanticGraph.load(self.graph_path)
            print(f"{Colors.CYAN}Loaded semantic memory: {self.graph.stats()}{Colors.RESET}")
        except FileNotFoundError:
            # Try to load from bootstrap file
            try:
                self.graph = SemanticGraph.load(self.bootstrap_path)
                self.graph.save(self.graph_path)  # Save as brain_memory.json
                print(f"{Colors.CYAN}Initialized semantic memory from bootstrap: {self.graph.stats()}{Colors.RESET}")
            except FileNotFoundError:
                self.graph = SemanticGraph("brain_memory")
                print(f"{Colors.CYAN}Created new semantic memory (no bootstrap found){Colors.RESET}")
        
        # Initialize semantic search engine for embedding-based retrieval
        _verbose = verbose if verbose is not None else CONFIG.get("verbose", False)
        self.search_engine = SemanticSearchEngine(self.graph, verbose=_verbose)
        if self.search_engine.is_available():
            print(f"{Colors.CYAN}Semantic search enabled (embedding-based context retrieval){Colors.RESET}")
        else:
            print(f"{Colors.YELLOW}Semantic search unavailable, using keyword fallback{Colors.RESET}")
        
        # Initialize emotion wheel for emotion averaging
        self.emotion_wheel = EmotionWheel(search_engine=self.search_engine)
        
        # Node type mapping from strings
        self.node_type_map = {
            "fact": NodeType.FACT,
            "concept": NodeType.CONCEPT,
            "entity": NodeType.ENTITY,
            "task": NodeType.TASK,
            "plan": NodeType.PLAN,
            "goal": NodeType.GOAL,
            "action": NodeType.ACTION,
            "tool": NodeType.TOOL,
            "conclusion": NodeType.CONCLUSION,
            "hypothesis": NodeType.HYPOTHESIS,
            "argument": NodeType.ARGUMENT,
            "evidence": NodeType.EVIDENCE,
            "rule": NodeType.RULE,
            "emotion": NodeType.EMOTION,
            "belief": NodeType.BELIEF,
            "preference": NodeType.PREFERENCE,
            "memory": NodeType.MEMORY,
            "observation": NodeType.OBSERVATION,
            "question": NodeType.QUESTION,
            "uncertainty": NodeType.UNCERTAINTY,
            "context": NodeType.CONTEXT,
            "event": NodeType.EVENT,
            "skill": NodeType.SKILL,
            "resource": NodeType.RESOURCE,
            "constraint": NodeType.CONSTRAINT,
            "procedure": NodeType.PROCEDURE,
        }
        
        # Relation type mapping from strings
        self.relation_type_map = {
            "causes": RelationType.CAUSES,
            "caused_by": RelationType.CAUSED_BY,
            "implies": RelationType.IMPLIES,
            "contradicts": RelationType.CONTRADICTS,
            "supports": RelationType.SUPPORTS,
            "refutes": RelationType.REFUTES,
            "is_a": RelationType.IS_A,
            "part_of": RelationType.PART_OF,
            "has_part": RelationType.HAS_PART,
            "instance_of": RelationType.INSTANCE_OF,
            "subclass_of": RelationType.SUBCLASS_OF,
            "related_to": RelationType.RELATED_TO,
            "similar_to": RelationType.SIMILAR_TO,
            "opposite_of": RelationType.OPPOSITE_OF,
            "synonym_of": RelationType.SYNONYM_OF,
            "defined_by": RelationType.DEFINED_BY,
            "before": RelationType.BEFORE,
            "after": RelationType.AFTER,
            "during": RelationType.DURING,
            "follows": RelationType.FOLLOWS,
            "precedes": RelationType.PRECEDES,
            "requires": RelationType.REQUIRES,
            "enables": RelationType.ENABLES,
            "blocks": RelationType.BLOCKS,
            "achieves": RelationType.ACHIEVES,
            "depends_on": RelationType.DEPENDS_ON,
            "uses": RelationType.USES,
            "produces": RelationType.PRODUCES,
            "created_by": RelationType.CREATED_BY,
            "derived_from": RelationType.DERIVED_FROM,
            "based_on": RelationType.BASED_ON,
            "evidenced_by": RelationType.EVIDENCED_BY,
            "belongs_to": RelationType.BELONGS_TO,
            "associated_with": RelationType.ASSOCIATED_WITH,
            "about": RelationType.ABOUT,
            "triggers": RelationType.TRIGGERS,
            "evokes": RelationType.EVOKES,
            "prefers": RelationType.PREFERS,
            "values": RelationType.VALUES,
        }
    
    def _get_current_emotion(self) -> Optional[Node]:
        """
        Get the current_emotion node.
        
        Returns the dedicated current_emotion node that tracks assistant's emotional state.
        """
        # Look for the dedicated current_emotion node
        for node in self.graph.nodes.values():
            if node.label == "current_emotion" and node.type == NodeType.EMOTION:
                return node
        
        # Fallback: get most recent emotion node
        all_emotions = self.graph.get_nodes_by_type(NodeType.EMOTION)
        if all_emotions:
            all_emotions.sort(key=lambda n: n.metadata.updated_at, reverse=True)
            return all_emotions[0]
        
        return None
    
    def _format_emotional_context(self) -> str:
        """
        Format current emotional state for context injection.
        """
        emotion = self._get_current_emotion()
        if not emotion:
            return ""
        
        parts = ["[Your (Assistant's) Previous Emotional State]"]
        # The content is now in format: "emotion | intensity: X | valence: Y | reason"
        parts.append(f"- {emotion.content}")
        parts.append("")
        parts.append("IMPORTANT: Analyze the user's CURRENT message for emotional cues (distress, excitement, concern, etc.).")
        parts.append("If the user's message indicates a different emotional tone, respond to THEIR emotional state, not your previous one.")
        parts.append("Your emotional state will be updated AFTER this response based on the full interaction.")
        parts.append("")
        
        return "\n".join(parts)
    
    def _parse_emotion_content(self, content: str) -> Optional[dict]:
        """
        Parse emotion content string into properties dict.
        
        Expected format: "emotion | intensity: 0.7 | valence: positive | reason"
        Returns: {"primary": "emotion", "intensity": 0.7, "valence": "positive"}
        """
        properties = {}
        
        try:
            parts = [p.strip() for p in content.split("|")]
            
            if parts:
                # First part is the primary emotion
                properties["primary"] = parts[0].lower().replace(" ", "_")
            
            for part in parts[1:]:
                if "intensity:" in part.lower():
                    try:
                        val = part.lower().replace("intensity:", "").strip()
                        properties["intensity"] = float(val)
                    except ValueError:
                        pass
                elif "valence:" in part.lower():
                    val = part.lower().replace("valence:", "").strip()
                    if val in ("positive", "negative", "neutral"):
                        properties["valence"] = val
        except Exception:
            pass
        
        return properties if properties else None
    
    def _average_emotion_update(self, current_node: Node, new_content: str) -> str:
        """
        Average the previous emotional state with the new one using the emotion wheel.
        
        This creates smoother emotional transitions by blending the previous
        emotion position and intensity with the new extracted emotion.
        
        Args:
            current_node: The current emotion node with previous state
            new_content: The new emotion content string from LLM extraction
            
        Returns:
            Updated content string with averaged emotion
        """
        # Parse previous emotion from current node
        prev_props = current_node.properties or {}
        prev_emotion = prev_props.get("primary", "neutral")
        prev_intensity = prev_props.get("intensity", 0.5)
        
        # Parse new emotion from LLM extraction
        new_props = self._parse_emotion_content(new_content)
        if not new_props:
            return new_content  # Can't parse, return as-is
        
        new_emotion = new_props.get("primary", "neutral")
        new_intensity = new_props.get("intensity", 0.5)
        
        # Extract the reason from new content (everything after the valence part)
        reason = ""
        parts = new_content.split("|")
        if len(parts) >= 4:
            reason = parts[3].strip()
        elif len(parts) >= 3:
            # Check if last part is the reason (not intensity or valence)
            last_part = parts[-1].strip()
            if "intensity:" not in last_part.lower() and "valence:" not in last_part.lower():
                reason = last_part
        
        # Determine if the new emotion should "win" or be blended
        # High intensity new emotions (>= 0.7) should take precedence to break emotion loops
        # Low intensity secondary emotions blend back toward the primary/core emotion
        intensity_threshold = 0.7
        
        if new_intensity >= intensity_threshold:
            # Strong new emotion wins - use it directly with slight dampening
            avg_emotion = new_emotion
            avg_intensity = new_intensity * 0.95  # Slight dampening to prevent runaway
        elif new_intensity < 0.4 and prev_intensity >= 0.5:
            # Weak new emotion with strong previous - revert toward previous/core
            # This prevents weak secondary emotions from diluting strong states
            avg_emotion = prev_emotion
            avg_intensity = prev_intensity * 0.9  # Gradual decay
        else:
            # Moderate intensities - use emotion wheel averaging
            # blend_factor of 0.6 gives slightly more weight to new emotion
            avg_emotion, avg_intensity = self.emotion_wheel.average_emotions(
                prev_emotion=prev_emotion,
                prev_intensity=prev_intensity,
                new_emotion=new_emotion,
                new_intensity=new_intensity,
                blend_factor=0.6
            )
        
        # Get valence for the averaged emotion
        avg_valence = self.emotion_wheel.get_valence(avg_emotion)
        
        # Log the averaging in verbose mode
        if self.chat.verbose:
            print(f"{Colors.DIM}  ♡ Emotion averaging: {prev_emotion}({prev_intensity:.1f}) + "
                  f"{new_emotion}({new_intensity:.1f}) → {avg_emotion}({avg_intensity:.2f}){Colors.RESET}")
        
        # Build the new content string
        if reason:
            result = f"{avg_emotion} | intensity: {avg_intensity:.2f} | valence: {avg_valence} | {reason}"
        else:
            result = f"{avg_emotion} | intensity: {avg_intensity:.2f} | valence: {avg_valence} | Blended emotional state"
        
        return result
    
    def _search_by_word(self, word: str, limit: int = 10) -> list[Node]:
        """Search graph for nodes matching a single word/phrase."""
        results = []
        word_lower = word.lower()
        
        for node in self.graph.nodes.values():
            if node.status.value != "active":
                continue
            
            score = 0.0
            # Check label
            if word_lower in node.label.lower():
                score += 2.0
            # Check content
            if word_lower in node.content.lower():
                score += 1.0
            # Check individual words in label
            label_words = node.label.lower().replace('_', ' ').split()
            if word_lower in label_words:
                score += 1.5
            # Check tags
            for tag in node.metadata.tags:
                if word_lower in tag.lower():
                    score += 0.5
            # Check properties
            for key, val in node.properties.items():
                if isinstance(val, str) and word_lower in val.lower():
                    score += 0.3
            
            if score > 0:
                results.append((score * node.confidence, node))
        
        results.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in results[:limit]]
    
    def _extract_keywords(self, text: str) -> list[str]:
        """
        Extract meaningful keywords from text using NLTK.
        
        Uses tokenization, POS tagging, lemmatization, and stopword removal.
        Prioritizes nouns, verbs, and adjectives.
        """
        if not NLTK_AVAILABLE:
            # Fallback to basic extraction
            words = [w.lower().strip('.,!?"\';:') for w in text.split()]
            return [w for w in words if w and len(w) > 2]
        
        # Get English stopwords and add custom ones
        stop_words = set(stopwords.words('english'))
        custom_stops = {'tell', 'please', 'another', 'give', 'show', 'want',
                        'need', 'know', 'think', 'like', 'would', 'could',
                        'get', 'make', 'let', 'say', 'also', 'well', 'much'}
        stop_words.update(custom_stops)
        
        # Tokenize and POS tag
        tokens = word_tokenize(text.lower())
        tagged = pos_tag(tokens)
        
        # Initialize lemmatizer
        lemmatizer = WordNetLemmatizer()
        
        keywords = []
        for word, tag in tagged:
            # Skip punctuation and stopwords
            if not word.isalnum() or word in stop_words or len(word) < 3:
                continue
            
            # Convert POS tag to WordNet format for lemmatization
            if tag.startswith('NN'):  # Nouns - high priority
                lemma = lemmatizer.lemmatize(word, 'n')
                keywords.append((lemma, 3))  # priority 3
            elif tag.startswith('VB'):  # Verbs
                lemma = lemmatizer.lemmatize(word, 'v')
                keywords.append((lemma, 2))  # priority 2
            elif tag.startswith('JJ'):  # Adjectives
                lemma = lemmatizer.lemmatize(word, 'a')
                keywords.append((lemma, 2))  # priority 2
            elif tag.startswith('RB'):  # Adverbs
                lemma = lemmatizer.lemmatize(word, 'r')
                keywords.append((lemma, 1))  # priority 1
            else:
                # Include other words with low priority
                keywords.append((word, 1))
        
        # Sort by priority and deduplicate
        keywords.sort(key=lambda x: x[1], reverse=True)
        seen = set()
        unique_keywords = []
        for kw, _ in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        
        return unique_keywords
    
    def _query_relevant_context(self, prompt: str, limit: int = 10) -> str:
        """
        Query the semantic graph for context relevant to the prompt.
        
        Uses SEMANTIC SEARCH (embedding-based) as primary method for finding
        contextually relevant memories. Falls back to keyword search if 
        semantic search is unavailable.
        
        Returns formatted context string for LLM.
        Always includes current emotional state.
        """
        context_parts = []
        
        # Always include emotional state first
        emotional_context = self._format_emotional_context()
        if emotional_context:
            context_parts.append(emotional_context)
        
        relevant_nodes = []
        seen_ids = set()
        
        # Exclude current emotion node from search (already included above)
        current_emotion = self._get_current_emotion()
        if current_emotion:
            seen_ids.add(current_emotion.id)
        
        # PRIMARY: Use semantic search if available
        if self.search_engine.is_available():
            # Semantic search finds contextually similar nodes
            semantic_results = self.search_engine.semantic_search(
                query=prompt,
                limit=limit,
                min_similarity=0.25,  # Lower threshold to be more inclusive
                exclude_ids=seen_ids
            )
            
            for node, similarity in semantic_results:
                if node.id not in seen_ids:
                    relevant_nodes.append((node, similarity))
                    seen_ids.add(node.id)
            
            if self.chat.verbose and semantic_results:
                print(f"{Colors.DIM}(Semantic search found {len(semantic_results)} relevant memories){Colors.RESET}")
        
        # FALLBACK: Keyword search if semantic search unavailable or found few results
        if not self.search_engine.is_available() or len(relevant_nodes) < 3:
            # Extract keywords using NLTK
            keywords = self._extract_keywords(prompt)
            
            # Search for each keyword
            for keyword in keywords:
                results = self._search_by_word(keyword, limit=5)
                for node in results:
                    if node.id not in seen_ids:
                        # Lower score for keyword matches vs semantic
                        relevant_nodes.append((node, 0.5))
                        seen_ids.add(node.id)
            
            # Also do a full-text search on the whole prompt
            full_results = self._search_by_word(prompt, limit=3)
            for node in full_results:
                if node.id not in seen_ids:
                    relevant_nodes.append((node, 0.4))
                    seen_ids.add(node.id)
        
        # Sort by relevance score and limit
        relevant_nodes.sort(key=lambda x: x[1], reverse=True)
        relevant_nodes = relevant_nodes[:limit]
        
        if relevant_nodes:
            context_parts.append("[Relevant Memory Context]")
        
        for node, score in relevant_nodes:
            # Get node info with relevance indicator
            node_info = f"- [{node.type.value}] {node.label}: {node.content}"
            if node.confidence < 1.0:
                node_info += f" (confidence: {node.confidence:.0%})"
            context_parts.append(node_info)
            
            # Get related nodes (1 hop)
            neighbors = self.graph.get_neighbors(node.id)
            for neighbor in neighbors[:3]:  # Limit neighbors
                # Find the relationship
                edges = self.graph.get_edges_between(node.id, neighbor.id)
                if not edges:
                    edges = self.graph.get_edges_between(neighbor.id, node.id)
                
                if edges:
                    rel = edges[0].relation.value
                    context_parts.append(f"  → {rel} → {neighbor.label}")
        
        if not context_parts:
            return ""
        
        context_parts.append("[End Context]\n")
        return "\n".join(context_parts)
    
    def _get_existing_nodes_summary(self, limit: int = 15) -> str:
        """Get a summary of existing nodes for the extraction prompt, including content for contradiction detection."""
        if not self.graph.nodes:
            return "none"
        
        # Only include active nodes, prioritize recent and high-confidence
        active_nodes = [n for n in self.graph.nodes.values() if n.status.value == "active"]
        nodes = sorted(
            active_nodes,
            key=lambda n: (n.metadata.updated_at, n.confidence),
            reverse=True
        )[:limit]
        
        # Include label AND content so LLM can detect contradictions
        # Format: "label: content" for each node
        summaries = []
        for node in nodes:
            # Truncate content if too long
            content = node.content[:80] + "..." if len(node.content) > 80 else node.content
            summaries.append(f"- {node.label}: {content}")
        
        return "\n".join(summaries) if summaries else "none"
    
    def _extract_knowledge(self, user_prompt: str, assistant_response: str) -> dict:
        """
        Make a SEPARATE dedicated LLM call to extract knowledge.
        
        This is isolated from the main conversation to ensure reliable extraction.
        Returns parsed JSON of nodes and relationships to add.
        """
        # Get existing nodes for context
        existing_nodes = self._get_existing_nodes_summary()
        
        # Get current emotional state for context
        current_emotion = self._get_current_emotion()
        if current_emotion:
            emotional_state = current_emotion.content
        else:
            emotional_state = "neutral | intensity: 0.5 | valence: neutral | No prior context"
        
        # Truncate inputs to avoid overwhelming the model
        user_prompt_short = user_prompt[:500]
        assistant_response_short = assistant_response[:800]
        
        # Build the extraction prompt
        extraction_prompt = SEMANTIC_EXTRACTION_TEMPLATE.format(
            user_prompt=user_prompt_short,
            assistant_response=assistant_response_short,
            existing_nodes=existing_nodes,
            emotional_state=emotional_state
        )
        
        # Make a SEPARATE API call with fresh context (not conversation history)
        import requests
        
        response_text = ""
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                request_payload = {
                    "model": self.chat.model,
                    "messages": [
                        {"role": "system", "content": SEMANTIC_EXTRACTION_SYSTEM},
                        {"role": "user", "content": extraction_prompt}
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Very low temp for structured output
                        "num_predict": 2048,  # Reasonable limit for JSON output
                    }
                }
                
                response = requests.post(
                    f"{self.chat.base_url}/api/chat",
                    json=request_payload,
                    timeout=300  # 5 minute timeout
                )
                response.raise_for_status()
                data = response.json()
                message = data.get("message", {})
                
                # Handle thinking models - content may be empty while thinking has the work
                response_text = message.get("content", "").strip()
                thinking_text = message.get("thinking", "").strip()
                
                # If content is empty but thinking has JSON, try to extract from thinking
                if not response_text and thinking_text:
                    # Look for JSON in thinking text
                    if '{' in thinking_text and '}' in thinking_text:
                        start = thinking_text.find('{')
                        end = thinking_text.rfind('}') + 1
                        if start != -1 and end > start:
                            potential_json = thinking_text[start:end]
                            # Only use if it looks like our expected structure
                            if '"nodes"' in potential_json or '"relationships"' in potential_json:
                                response_text = potential_json
                
                # If we got a response, break out of retry loop
                if response_text:
                    break
                    
            except requests.exceptions.Timeout:
                continue
            except Exception as e:
                break
        
        if not response_text:
            return {"nodes": [], "relationships": [], "updates": []}
        
        # Parse JSON from response
        try:
            # Handle markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                parts = response_text.split("```")
                if len(parts) >= 2:
                    response_text = parts[1]
            
            # Try to find JSON object in text
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end > start:
                response_text = response_text[start:end]
            
            # Clean up common issues
            response_text = response_text.replace('\n', ' ').replace('\r', '')
            
            result = json.loads(response_text)
            
            # Validate structure
            if not isinstance(result, dict):
                raise json.JSONDecodeError("Not a dict", response_text, 0)
            
            # Ensure required keys exist
            result.setdefault("nodes", [])
            result.setdefault("relationships", [])
            result.setdefault("updates", [])
            result.setdefault("deactivations", [])
            
            # Show what was extracted in verbose mode
            if self.chat.verbose:
                n_nodes = len(result.get("nodes", []))
                n_rels = len(result.get("relationships", []))
                n_updates = len(result.get("updates", []))
                n_deactivations = len(result.get("deactivations", []))
                if n_nodes or n_rels or n_updates or n_deactivations:
                    print(f"{Colors.DIM}(Extracted: {n_nodes} nodes, {n_rels} relationships, {n_updates} updates, {n_deactivations} deactivations){Colors.RESET}")
            
            return result
            
        except json.JSONDecodeError as e:
            # Show error in verbose mode
            if self.chat.verbose:
                print(f"{Colors.DIM}(Could not parse extraction JSON){Colors.RESET}")
            return {"nodes": [], "relationships": [], "updates": [], "deactivations": []}
    
    def _apply_knowledge(self, knowledge: dict) -> int:
        """
        Apply extracted knowledge to the semantic graph.
        
        Returns number of items added/updated.
        """
        count = 0
        node_label_to_id = {}
        
        # First, map existing nodes by label
        for node in self.graph.nodes.values():
            node_label_to_id[node.label.lower()] = node.id
        
        # Add new nodes
        for node_data in knowledge.get("nodes", []):
            node_type_str = node_data.get("type", "fact").lower()
            node_type = self.node_type_map.get(node_type_str, NodeType.FACT)
            
            label = node_data.get("label", "")
            if not label:
                continue
                
            # Check if node already exists
            if label.lower() in node_label_to_id:
                continue
            
            try:
                node = self.graph.add_node(
                    node_type=node_type,
                    label=label,
                    content=node_data.get("content", label),
                    confidence=node_data.get("confidence", 0.8),
                    properties=node_data.get("properties", {}),
                    source="conversation"
                )
                node_label_to_id[label.lower()] = node.id
                
                # Update embedding for the new node to keep semantic search in sync
                if self.search_engine.is_available():
                    self.search_engine.update_node_embedding(node)
                
                count += 1
                if self.chat.verbose:
                    print(f"{Colors.DIM}  + Added {node_type.value}: {label}{Colors.RESET}")
            except Exception as e:
                if self.chat.verbose:
                    print(f"{Colors.DIM}  ! Failed to add node: {e}{Colors.RESET}")
        
        # Add relationships
        for rel_data in knowledge.get("relationships", []):
            source_label = rel_data.get("source_label", "").lower()
            target_label = rel_data.get("target_label", "").lower()
            relation_str = rel_data.get("relation", "related_to").lower()
            
            source_id = node_label_to_id.get(source_label)
            target_id = node_label_to_id.get(target_label)
            
            if not source_id or not target_id:
                continue
            
            relation = self.relation_type_map.get(relation_str, RelationType.RELATED_TO)
            
            # Check if edge already exists
            existing = self.graph.get_edges_between(source_id, target_id, relation)
            if existing:
                continue
            
            try:
                self.graph.add_edge(
                    source_id=source_id,
                    target_id=target_id,
                    relation=relation,
                    confidence=rel_data.get("confidence", 0.8)
                )
                count += 1
                if self.chat.verbose:
                    print(f"{Colors.DIM}  + Added relation: {source_label} -{relation.value}-> {target_label}{Colors.RESET}")
            except Exception as e:
                if self.chat.verbose:
                    print(f"{Colors.DIM}  ! Failed to add relation: {e}{Colors.RESET}")
        
        # Apply updates to existing nodes
        for update in knowledge.get("updates", []):
            node_label = update.get("node_label", "").lower()
            node_id = node_label_to_id.get(node_label)
            
            if not node_id:
                continue
            
            try:
                node = self.graph.nodes.get(node_id)
                if not node:
                    continue
                
                # Handle append_content - add to existing content
                new_content = None
                new_properties = None
                
                if update.get("append_content"):
                    new_content = f"{node.content} | {update['append_content']}"
                elif update.get("new_content"):
                    new_content = update["new_content"]
                    
                    # Special handling for current_emotion node - use emotion wheel averaging
                    if node_label == "current_emotion" and new_content:
                        new_content = self._average_emotion_update(node, new_content)
                        new_properties = self._parse_emotion_content(new_content)
                
                self.graph.update_node(
                    node_id=node_id,
                    content=new_content,
                    confidence=update.get("new_confidence"),
                    properties=new_properties
                )
                
                # Update embedding for the modified node to keep semantic search in sync
                if self.search_engine.is_available() and new_content:
                    updated_node = self.graph.nodes.get(node_id)
                    if updated_node:
                        self.search_engine.update_node_embedding(updated_node)
                
                count += 1
                if self.chat.verbose:
                    print(f"{Colors.DIM}  ~ Updated: {node_label}{Colors.RESET}")
            except Exception as e:
                if self.chat.verbose:
                    print(f"{Colors.DIM}  ! Failed to update: {e}{Colors.RESET}")
        
        # Apply deactivations to stale nodes
        status_map = {
            "superseded": "superseded",
            "invalidated": "invalidated", 
            "archived": "archived"
        }
        
        for deactivation in knowledge.get("deactivations", []):
            node_label = deactivation.get("node_label", "").lower()
            node_id = node_label_to_id.get(node_label)
            status_str = deactivation.get("status", "superseded").lower()
            reason = deactivation.get("reason", "")
            
            if not node_id:
                # Try to find by partial label match
                for label, nid in node_label_to_id.items():
                    if node_label in label or label in node_label:
                        node_id = nid
                        break
            
            if not node_id:
                if self.chat.verbose:
                    print(f"{Colors.DIM}  ! Could not find node to deactivate: {node_label}{Colors.RESET}")
                continue
            
            try:
                from semantic_graph import NodeStatus
                
                # Map status string to NodeStatus enum
                if status_str == "superseded":
                    new_status = NodeStatus.SUPERSEDED
                elif status_str == "invalidated":
                    new_status = NodeStatus.INVALIDATED
                elif status_str == "archived":
                    new_status = NodeStatus.ARCHIVED
                else:
                    new_status = NodeStatus.SUPERSEDED  # Default
                
                node = self.graph.nodes.get(node_id)
                if not node:
                    continue
                
                # Update node status and add reason to properties
                updated_props = node.properties.copy() if node.properties else {}
                updated_props["deactivation_reason"] = reason
                
                self.graph.update_node(
                    node_id=node_id,
                    status=new_status,
                    properties=updated_props
                )
                
                # Remove from embedding cache since it's no longer active
                if self.search_engine.is_available():
                    self.search_engine.remove_node_embedding(node_id)
                
                count += 1
                if self.chat.verbose:
                    print(f"{Colors.DIM}  ✗ Deactivated ({status_str}): {node_label} - {reason}{Colors.RESET}")
            except Exception as e:
                if self.chat.verbose:
                    print(f"{Colors.DIM}  ! Failed to deactivate: {e}{Colors.RESET}")
        
        return count
    
    def process(self, user_prompt: str, stream: bool = True) -> Generator[str, None, None]:
        """
        Process a user prompt through the full brain pipeline:
        1. Query semantic graph for relevant context
        2. Send prompt + context to LLM, get response
        3. Extract knowledge from the exchange
        4. Update semantic graph
        
        Yields response chunks if streaming.
        """
        # Step 1: Query semantic graph for relevant context
        context = self._query_relevant_context(user_prompt)
        
        # Only show context in verbose mode
        if context and self.chat.verbose:
            print(f"\n{Colors.CYAN}{context}{Colors.RESET}")
        
        # Step 2: Add the RAW user message to conversation history
        # This preserves clean conversation flow for future context
        self.chat.conversation_history.append({
            "role": "user",
            "content": user_prompt
        })
        
        # Step 3: Build messages for THIS request only, with context injected
        # We copy history and modify the LAST user message to include context
        messages_for_request = self.chat.conversation_history.copy()
        if context:
            # Inject context into the current message only (not stored in history)
            enhanced_prompt = f"{context}\n\nUser: {user_prompt}"
            messages_for_request[-1] = {
                "role": "user",
                "content": enhanced_prompt
            }
        
        # Build request payload with enhanced messages (but history stays clean)
        payload = {
            "model": self.chat.model,
            "messages": messages_for_request,
            "stream": stream,
            "options": self.chat._build_options(),
            "think": True
        }
        
        url = f"{self.chat.base_url}/api/chat"
        
        # Step 2: Get LLM response
        if stream:
            full_response = ""
            for chunk in self.chat._stream_response(url, payload):
                # Capture full response while yielding
                # Note: chunks include color codes, need to strip for storage
                if not chunk.startswith("\033["):
                    full_response += chunk
                yield chunk
            
            # Get the actual response from history (cleaner)
            if self.chat.conversation_history:
                last_msg = self.chat.conversation_history[-1]
                if last_msg.get("role") == "assistant":
                    full_response = last_msg.get("content", "")
        else:
            full_response = self.chat._get_response(url, payload)
            yield full_response
        
        # Step 3 & 4: Extract and apply knowledge (async from user's perspective)
        if self.chat.verbose:
            print(f"\n{Colors.DIM}(Updating semantic memory...){Colors.RESET}")
        knowledge = self._extract_knowledge(user_prompt, full_response)
        updates = self._apply_knowledge(knowledge)
        
        if updates > 0:
            if self.chat.verbose:
                print(f"{Colors.DIM}(Added {updates} items to memory){Colors.RESET}")
            if self.auto_save:
                self.graph.save(self.graph_path)
        
    def show_memory_stats(self):
        """Display semantic memory statistics."""
        stats = self.graph.stats()
        print(f"\n{Colors.CYAN}=== Semantic Memory ==={Colors.RESET}")
        print(f"  Nodes: {stats['total_nodes']}")
        print(f"  Edges: {stats['total_edges']}")
        if stats['nodes_by_type']:
            print(f"  By type:")
            for ntype, count in stats['nodes_by_type'].items():
                print(f"    {ntype}: {count}")
        print(f"{Colors.CYAN}======================{Colors.RESET}")
    
    def search_memory(self, query: str, limit: int = 5):
        """
        Search semantic memory using embedding-based semantic search.
        
        Falls back to keyword search if semantic search is unavailable.
        """
        print(f"\n{Colors.CYAN}=== Memory Search: '{query}' ==={Colors.RESET}")
        
        # Try semantic search first
        if self.search_engine.is_available():
            results = self.search_engine.semantic_search(query, limit=limit)
            if results:
                print(f"{Colors.DIM}(Using semantic search){Colors.RESET}")
                for node, score in results:
                    print(f"  [{node.type.value}] {node.label} (relevance: {score:.0%})")
                    print(f"    {node.content[:100]}...")
                    neighbors = self.graph.get_neighbors(node.id)
                    if neighbors:
                        print(f"    Related: {', '.join(n.label for n in neighbors[:3])}")
            else:
                print("  No semantically similar memories found.")
        else:
            # Fallback to keyword search
            results = self.graph.search(query, limit=limit)
            print(f"{Colors.DIM}(Using keyword search){Colors.RESET}")
            for node in results:
                print(f"  [{node.type.value}] {node.label}")
                print(f"    {node.content[:100]}...")
                neighbors = self.graph.get_neighbors(node.id)
                if neighbors:
                    print(f"    Related: {', '.join(n.label for n in neighbors[:3])}")
        
        print(f"{Colors.CYAN}================================{Colors.RESET}")
    
    def clear_chat_history(self):
        """Clear chat history but preserve semantic memory."""
        self.chat.clear_history()
        # Re-apply the system prompt after clearing
        self._apply_system_prompt()
    
    def _apply_system_prompt(self):
        """Apply the system prompt (user custom overrides main if set)."""
        if hasattr(self, 'user_system_prompt') and self.user_system_prompt:
            # User system prompt completely replaces the main prompt
            self.chat.set_system_prompt(self.user_system_prompt)
        else:
            self.chat.set_system_prompt(MAIN_SYSTEM_PROMPT)
    
    def set_user_system_prompt(self, prompt: str):
        """
        Set a custom user system prompt that replaces the main system prompt.
        
        This allows overriding the default behavior with custom instructions,
        personas, or context. The main system prompt will not be used when
        a user system prompt is active.
        
        Args:
            prompt: Custom system prompt to use instead of the main prompt
        """
        self.user_system_prompt = prompt
        self._apply_system_prompt()
        print("Custom system prompt applied.")
    
    def clear_user_system_prompt(self):
        """Remove the custom user system prompt, keeping only the main prompt."""
        self.user_system_prompt = None
        self._apply_system_prompt()
        print("Custom system prompt cleared.")
    
    def save_memory(self):
        """Manually save semantic memory."""
        self.graph.save(self.graph_path)
        print(f"Saved semantic memory to {self.graph_path}")


def main():
    """Interactive brain interface."""
    print(f"{Colors.CYAN}{Colors.BOLD}" + "=" * 60)
    print("Brain Interface - Semantic Memory Enhanced LLM")
    print("=" * 60 + f"{Colors.RESET}")
    print(f"\n{Colors.YELLOW}Commands:{Colors.RESET}")
    print("  /clear     - Clear chat history (keeps memory)")
    print("  /memory    - Show semantic memory stats")
    print("  /search    - Search semantic memory")
    print("  /save      - Save semantic memory")
    print("  /thinking  - Change thinking mode")
    print("  /verbose   - Toggle verbose mode")
    print("  /system    - Set system prompt")
    print("  /quit      - Exit")
    print(f"{Colors.CYAN}" + "=" * 60 + f"{Colors.RESET}")
    
    brain = BrainInterface(
        graph_path="brain_memory.json"
    )
    
    brain.show_memory_stats()
    
    while True:
        try:
            user_input = input(f"\n{Colors.BOLD}You:{Colors.RESET} ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                cmd = user_input.lower().split()[0]
                
                if cmd == "/quit":
                    brain.save_memory()
                    print("Goodbye!")
                    break
                    
                elif cmd == "/clear":
                    brain.clear_chat_history()
                    continue
                    
                elif cmd == "/memory":
                    brain.show_memory_stats()
                    continue
                    
                elif cmd == "/search":
                    query = user_input[7:].strip()
                    if query:
                        brain.search_memory(query)
                    else:
                        print("Usage: /search <query>")
                    continue
                    
                elif cmd == "/save":
                    brain.save_memory()
                    continue
                    
                elif cmd == "/thinking":
                    parts = user_input.split()
                    if len(parts) > 1:
                        brain.chat.set_thinking_mode(parts[1].lower())
                    else:
                        print(f"Current: {brain.chat.thinking_mode}")
                    continue
                    
                elif cmd == "/verbose":
                    parts = user_input.split()
                    if len(parts) > 1:
                        brain.chat.set_verbose(parts[1].lower() in ("on", "true", "yes"))
                    else:
                        brain.chat.set_verbose(not brain.chat.verbose)
                    continue
                
                elif cmd == "/system":
                    prompt = user_input[8:].strip()
                    if prompt.lower() == "clear":
                        brain.clear_user_system_prompt()
                    elif prompt:
                        brain.set_user_system_prompt(prompt)
                    elif brain.user_system_prompt:
                        print(f"Current: {brain.user_system_prompt[:100]}...")
                        print("Use '/system clear' to remove")
                    else:
                        print("No custom system prompt set.")
                        print("Usage: /system <your prompt>")
                    continue
                    
                else:
                    print(f"Unknown command: {cmd}")
                    continue
            
            # Process through brain pipeline
            # Use streaming on first request (to detect thinking support) or when thinking is supported
            use_stream = brain.chat._supports_thinking is None or brain.chat._supports_thinking is True
            print(f"\n{Colors.BOLD}Assistant:{Colors.RESET} ", end="", flush=True)
            print(Colors.GREEN, end="", flush=True)
            for chunk in brain.process(user_input, stream=use_stream):
                print(chunk, end="", flush=True)
            print(Colors.RESET)
            
        except KeyboardInterrupt:
            print("\n")
            brain.save_memory()
            print("Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
