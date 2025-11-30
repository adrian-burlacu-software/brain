"""
Brain Interface - Semantic Memory Enhanced LLM

Flow for every prompt:
1. Query semantic graph for relevant context
2. Send prompt to LLM WITH that context, get response
3. Ask LLM to extract knowledge for graph updates (JSON)
4. Update semantic graph with extracted knowledge

This creates a thin but powerful system where the LLM gains
persistent, structured memory that improves over time.
"""

import json
import re
from typing import Optional, Generator

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
from interface import OllamaChat, Colors


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
{{"nodes": [{{"type": "fact|concept|entity|...", "label": "snake_case", "content": "brief description", "confidence": 0.8}}], "relationships": [{{"source_label": "label", "relation": "related_to|...", "target_label": "label", "confidence": 0.8}}], "updates": [{{"node_label": "current_emotion", "new_content": "<emotion> | intensity: <0.0-1.0> | valence: <pos/neg/neutral> | <reason>"}}]}}
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
        model: str = "gpt-oss:20b",
        base_url: str = "http://localhost:11434",
        thinking_mode: str = "medium",
        verbose: bool = False,
        auto_save: bool = True
    ):
        """
        Initialize the Brain Interface.
        
        Args:
            graph_path: Path to load/save semantic graph
            model: Ollama model to use
            base_url: Ollama API URL
            thinking_mode: Thinking budget level
            verbose: Show thinking process
            auto_save: Auto-save graph after updates
        """
        self.chat = OllamaChat(
            model=model,
            base_url=base_url,
            thinking_mode=thinking_mode,
            verbose=verbose
        )
        
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
        
        self.auto_save = auto_save
        
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
        
        Returns formatted context string for LLM.
        Always includes current emotional state.
        """
        context_parts = []
        
        # Always include emotional state first
        emotional_context = self._format_emotional_context()
        if emotional_context:
            context_parts.append(emotional_context)
        
        # Extract keywords using NLTK
        keywords = self._extract_keywords(prompt)
        
        relevant_nodes = []
        seen_ids = set()
        
        # Exclude current emotion node from keyword search (already included above)
        current_emotion = self._get_current_emotion()
        if current_emotion:
            seen_ids.add(current_emotion.id)
        
        # Search for each keyword
        for keyword in keywords:
            results = self._search_by_word(keyword, limit=5)
            for node in results:
                if node.id not in seen_ids:
                    relevant_nodes.append(node)
                    seen_ids.add(node.id)
        
        # Also do a full-text search on the whole prompt
        full_results = self._search_by_word(prompt, limit=3)
        for node in full_results:
            if node.id not in seen_ids:
                relevant_nodes.append(node)
                seen_ids.add(node.id)
        
        # Limit total results
        relevant_nodes = relevant_nodes[:limit]
        
        if relevant_nodes:
            context_parts.append("[Relevant Memory Context]")
        
        for node in relevant_nodes:
            # Get node info
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
    
    def _get_existing_nodes_summary(self, limit: int = 10) -> str:
        """Get a summary of existing nodes for the extraction prompt."""
        if not self.graph.nodes:
            return "none"
        
        # Prioritize recent and high-confidence nodes
        nodes = sorted(
            self.graph.nodes.values(),
            key=lambda n: (n.metadata.updated_at, n.confidence),
            reverse=True
        )[:limit]
        
        # Very compact format to save tokens
        labels = [node.label for node in nodes]
        return ", ".join(labels) if labels else "none"
    
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
                response = requests.post(
                    f"{self.chat.base_url}/api/chat",
                    json={
                        "model": self.chat.model,
                        "messages": [
                            {"role": "system", "content": SEMANTIC_EXTRACTION_SYSTEM},
                            {"role": "user", "content": extraction_prompt}
                        ],
                        "stream": False,
                        "think": True,  # Enable thinking for better extraction
                        "options": {
                            "temperature": 0.1,  # Very low temp for structured output
                            "num_predict": -1,  # Unlimited output tokens
                            "thinking_budget": 8192,  # Medium-high thinking budget
                        }
                    },
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
        
        # Log raw response in verbose mode
        if self.chat.verbose:
            print(f"{Colors.DIM}[Extraction raw response]:{Colors.RESET}")
            print(f"{Colors.DIM}{response_text}{Colors.RESET}")
        
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
            
            # Show what was extracted in verbose mode
            if self.chat.verbose:
                n_nodes = len(result.get("nodes", []))
                n_rels = len(result.get("relationships", []))
                n_updates = len(result.get("updates", []))
                if n_nodes or n_rels or n_updates:
                    print(f"{Colors.DIM}(Extracted: {n_nodes} nodes, {n_rels} relationships, {n_updates} updates){Colors.RESET}")
            
            return result
            
        except json.JSONDecodeError as e:
            # Show error in verbose mode
            if self.chat.verbose and response_text:
                print(f"{Colors.DIM}(Could not parse extraction JSON: {e}){Colors.RESET}")
                print(f"{Colors.DIM}Raw response ({len(response_text)} chars):{Colors.RESET}")
                print(f"{Colors.DIM}{response_text}{Colors.RESET}")
            return {"nodes": [], "relationships": [], "updates": []}
    
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
                    
                    # Special handling for current_emotion node - parse properties from content
                    if node_label == "current_emotion" and new_content:
                        new_properties = self._parse_emotion_content(new_content)
                
                self.graph.update_node(
                    node_id=node_id,
                    content=new_content,
                    confidence=update.get("new_confidence"),
                    properties=new_properties
                )
                count += 1
                if self.chat.verbose:
                    print(f"{Colors.DIM}  ~ Updated: {node_label}{Colors.RESET}")
            except Exception as e:
                if self.chat.verbose:
                    print(f"{Colors.DIM}  ! Failed to update: {e}{Colors.RESET}")
        
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
        """Search semantic memory."""
        results = self.graph.search(query, limit=limit)
        print(f"\n{Colors.CYAN}=== Memory Search: '{query}' ==={Colors.RESET}")
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
        """Apply the combined system prompt (main + user custom)."""
        if hasattr(self, 'user_system_prompt') and self.user_system_prompt:
            combined = MAIN_SYSTEM_PROMPT + "\n\n" + self.user_system_prompt
        else:
            combined = MAIN_SYSTEM_PROMPT
        self.chat.set_system_prompt(combined)
    
    def set_user_system_prompt(self, prompt: str):
        """
        Set a custom user system prompt that extends the main system prompt.
        
        This allows adding custom instructions, personas, or context
        on top of the core memory-aware system prompt.
        
        Args:
            prompt: Custom system prompt to append to the main prompt
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
        graph_path="brain_memory.json",
        model="gpt-oss:20b",
        thinking_mode="medium",
        verbose=False,
        auto_save=True
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
            print(f"\n{Colors.BOLD}Assistant:{Colors.RESET} ", end="", flush=True)
            for chunk in brain.process(user_input, stream=True):
                print(chunk, end="", flush=True)
            print()
            
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
