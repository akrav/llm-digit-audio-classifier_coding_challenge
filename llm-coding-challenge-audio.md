# LLM Coding Challenge - Digit Classification from Audio

This challenge is designed to understand how you use LLMs as development partners — to code faster, reason better, and improve the quality and flexibility of what you build.

You are expected to use tools like Cursor, Claude Code, or Gemini Code Assist for this coding challenge. We are just as interested in how you work with LLMs as we are in what you produce.

Please record up to 30 minutes of your development process — any moments that show how you think, prompt, debug, or experiment. No need to narrate or polish anything. Just a raw glimpse into your process is more than enough.

Time investment: A couple of hours — not days!

## What you will build
Your task is to build a lightweight prototype that listens to spoken digits (0–9) and predicts the correct number. The goal is to find the lightest effective solution you can create within a couple of hours — with the support from LLM coding tools.

If you have time, feel free to extend the challenge by simulating microphone noise or testing model robustness. But the core objective is simple: audio in, digit out — fast, clean, and functional.

## Bonus: Microphone Integration

As an optional challenge, you can add live microphone input to test your model in real time. This helps explore real-world performance, including latency, noise handling, and usability under less controlled conditions.

## Dataset

This challenge uses the Free Spoken Digit Dataset (FSDD), an open dataset of WAV recordings with spoken digits (0–9) spoken by multiple English speakers at 8kHz. You can access the Hugging Face dataset [here](https://huggingface.co/datasets/mteb/free-spoken-digit-dataset/viewer/default/train?views%5B%5D=train).

## What to Submit

Please submit your code together with a short README.md, and some recordings showing how you worked with an LLM during development. You can share everything via GitHub. The README should briefly explain your approach and present key results.

## Evaluation Criteria

- Modeling choices: Are the audio features and model type appropriate for the task?
- Model performance: Is performance measured with relevant metrics, and are the results strong?
- Responsiveness: Is there minimal delay between input and output?
- Code architecture: Is the code clean, modular, and easy to extend?
- LLM collaboration: Is there strong evidence of prompting, debugging, or architectural reasoning with an LLM?
- Creative energy: Does the submission reflect curiosity or a desire to push boundaries?