"""LayaSource — replaces the TypeSafe API client in jev-ultrafast's model.py.
Same interface (choose → choice/probabilities/confidence), backed by a locally
fine-tuned Laya model. Zero cloud calls, zero API keys."""

import json, re, math

class LayaSource:
    """Drop-in replacement for the TypeSafe API client.
    Uses a fine-tuned Laya model for operation/target choice questions."""
    
    def __init__(self, checkpoint_path):
        from laya import Agent as LayaAgent
        self.agent = LayaAgent(checkpoint_path)
        self._last_questions = None
    
    def choose(self, state, goal, history):
        """Build typed questions from the page state, call laya.predict,
        return the answer in the same shape as the TypeSafe API."""
        from .questions import NEXT_ACTION, TARGET
        
        # Build the state text (what the model sees)
        state_text = json.dumps({
            "url": state.get("url", ""),
            "title": state.get("title", ""),
            "elements": state.get("elements", []),
            "goal": goal,
            "recent_actions": [h.get("decision", {}).get("choice", "") for h in (history or [])[-5:]],
        })
        
        # ask laya for the next operation
        result = self.agent.decide(state_text)
        
        # laya returns choice + probabilities in the typed-decisions format
        return {
            "choice": result.get("choice", "DONE"),
            "probabilities": result.get("probabilities", {}),
            "confidence": result.get("confidence", 0.5),
        }
    
    def choose_target(self, state, goal, operation, targets):
        """Choose a target element for the given operation."""
        result = self.agent.decide(json.dumps({
            "state": state, "operation": operation, "targets": targets, "goal": goal
        }))
        return result

def validate_choice(answer, ids):
    """Same validation as the TypeSafe path."""
    try:
        probabilities = answer["probabilities"]
        numbers = [*probabilities.values(), answer["confidence"]]
        valid = (
            answer["choice"] in ids
            and set(probabilities) == set(ids)
            and all(type(n) in (int, float) and math.isfinite(n) and 0 <= n <= 1 for n in numbers)
            and abs(sum(probabilities.values()) - 1) < 0.02
            and probabilities[answer["choice"]] >= max(probabilities.values()) - 1e-6
        )
    except (KeyError, TypeError, ValueError):
        valid = False
    return answer if valid else None
