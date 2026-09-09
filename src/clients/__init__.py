# API clients module

# Install narrow OpenRouter live-trade cost guards when the clients package loads.
from src.clients.openrouter_cost_guard import install_openrouter_cost_guard

install_openrouter_cost_guard()
