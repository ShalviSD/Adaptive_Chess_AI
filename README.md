# Adaptive_Chess_AI
## Overview

The Adaptive Chess AI project introduces a new approach to computer chess by blending traditional rule-based analysis with the contextual understanding and natural language abilities of Large Language Models (LLMs). This aims to create an intelligent chess opponent and tutor that enhances learning through dynamic difficulty and insightful explanations. The system's core innovation is its dual analysis of the game, evaluating chess features to predict playing styles (Aggressive, Defensive, etc.) while the Gemini 2.0 Flash LLM independently assesses these styles. Comparing these analyses allows for more nuanced decision-making, and Gemini 2.0 Flash also generates human-readable explanations for moves, enriching the learning experience.

## Tech Stack

The project leverages a modern technology stack focused on cloud-based LLM integration:

* **Large Language Model (LLM):**
    * **Gemini 2.0 Flash:** The primary LLM, chosen for its rapid inference capabilities, enabling real-time responses and the generation of natural language explanations for moves and strategic reasoning.
* **Cloud Platform:**
    * **Google Vertex AI:** This unified cloud-based platform provides the scalable and low-latency infrastructure necessary for hosting and deploying the Gemini 2.0 Flash model.
* **Programming Languages:**
    * **Python:** Likely the primary language for implementing the AI logic, including the traditional chess evaluation components and interaction with the Gemini LLM.
    * **JavaScript:** Likely used for the development of the web-based user interface.
* **Chess Engine/Evaluation:**
    * **Traditional Chess Evaluation Algorithms:** Rule-based systems are employed to analyze the game state across different phases, providing a structured assessment of board features and strategic implications.

## Features

The Adaptive Chess AI system offers a range of features designed to enhance the chess playing and learning experience:

* **Adaptive Difficulty:** The AI is designed to adjust its playing strength dynamically based on the user's skill level (potentially leveraging ELO ratings or performance history) to provide a challenging yet engaging experience.
* **Natural Language Explanations:** Utilizing the Gemini 2.0 Flash model, the AI can generate human-readable explanations for its chosen moves, detailing the underlying strategic thinking and potential consequences, which is invaluable for learning.
* **Phase-Specific Evaluation:** The AI employs distinct evaluation strategies tailored to the current phase of the game (opening, middlegame, endgame), allowing for more contextually relevant and strategic move selection.
* **Playing Style Analysis:** The system analyzes the game context to identify and predict prevalent playing styles (e.g., Aggressive, Defensive, Tactical, Materialistic, Endgame Grinder). This analysis is performed through both algorithmic evaluation and the Gemini LLM, with a comparison of the results.
* **Real-Time Move Generation:** The integration of Gemini 2.0 Flash on Google Vertex AI enables fast and efficient move generation, crucial for a fluid and interactive gameplay experience.
* **Web Application:** A user-friendly web interface allows users to interact with the Adaptive Chess AI, providing a platform for playing games and receiving move explanations.
