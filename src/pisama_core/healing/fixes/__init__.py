"""Built-in fix implementations."""

from pisama_core.healing.fixes.break_loop import BreakLoopFix
from pisama_core.healing.fixes.escalate import EscalateFix
from pisama_core.healing.fixes.switch_strategy import SwitchStrategyFix

__all__ = ["BreakLoopFix", "SwitchStrategyFix", "EscalateFix"]
