"""Agent implementations: BaseAgent protocol, random, myopic, Q-learning, deep-Q, LLM."""
from hotelling.agents.chain_ceo import ChainCEO
from hotelling.agents.entrant_llm import EntrantLLM
from hotelling.agents.store_rl import StoreQLearner

__all__ = ["ChainCEO", "EntrantLLM", "StoreQLearner"]
