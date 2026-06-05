"""
Reflection cycle: analyzes trades and proposes ONE strategy change.
"""
import json
import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def reflect(trades: list, strategy_path: str = "state/strategy.yaml") -> dict:
    """Analyze trades and propose ONE variable change."""
    if not trades or len(trades) < 5:
        return {"status": "waiting", "trades_count": len(trades)}
    
    if not Path(strategy_path).exists():
        strategy_path = f"/app/{strategy_path}"
    
    with open(strategy_path) as f:
        strategy = yaml.safe_load(f)
    
    wins = [t for t in trades if t.get("pnl", 0) > 0]
    losses = [t for t in trades if t.get("pnl", 0) <= 0]
    win_rate = len(wins) / len(trades) if trades else 0
    
    hypothesis = None
    if win_rate < 0.5:
        hypothesis = {
            "variable": "step_3_risk_reward.minimum_ratio",
            "old_value": 2.5,
            "new_value": 3.0,
            "reason": f"Win rate {win_rate:.1%} below 50%, raising R:R minimum to filter out lower probability trades",
        }
    
    return {
        "status": "proposed" if hypothesis else "waiting",
        "trades_analyzed": len(trades),
        "win_rate": win_rate,
        "hypothesis": hypothesis,
    }
