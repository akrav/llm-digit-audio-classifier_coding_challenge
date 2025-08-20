### Cursor Prompt: Execute Ticket TICKET-4002

Please execute this ticket:

Ticket to execute:
- TICKET_ID: 4002
- TICKET_NAME: TICKET-4002 - Implement Live Microphone Integration.md
- TICKET_FILE: /Users/adam/Documents/GitHub/CloudWalk_Techinical_Challenge/Tickets/Sprint 4/TICKET-4002 - Implement Live Microphone Integration.md

Permanent references (always follow):
- API Reference: /Users/adam/Documents/GitHub/CloudWalk_Techinical_Challenge/Build Documentation/API-Reference.md
- Database Schema: /Users/adam/Documents/GitHub/CloudWalk_Techinical_Challenge/Build Documentation/Schemas.md
- Project Structure: /Users/adam/Documents/GitHub/CloudWalk_Techinical_Challenge/Build Documentation/structure.md
- Troubleshooting: /Users/adam/Documents/GitHub/CloudWalk_Techinical_Challenge/Build Documentation/Troubleshooting.md
- Sprint Progress: /Users/adam/Documents/GitHub/CloudWalk_Techinical_Challenge/Build Documentation/Sprint-Progress.md

Environment and secrets:
- Use .env; never commit secrets.

Objective:
- Build `src/live_inference.py` to capture microphone audio, preprocess consistently, and call `predict_single_audio` for real-time predictions.

Constraints and style:
- Follow best practices; keep changes scoped; no secrets.

Required steps:
1) Read the ticket and acceptance criteria.
2) Implement microphone capture using `sounddevice` and chunked processing.
3) Ensure low latency and stable stream; log predictions.
4) Documentation: update ticket status, Sprint-Progress, Troubleshooting.

Output:
- Concise status and how to run the live script.

Success criteria:
- Real-time predictions with acceptable latency; docs updated. 