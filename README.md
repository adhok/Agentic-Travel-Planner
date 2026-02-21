# Agentic Travel Planner (DeepSeek + SerpApi)

A terminal-based travel planning assistant that uses:
- **DeepSeek** for conversational planning, itinerary generation, hotel/flight suggestions, and day-trip feasibility checks.
- **SerpApi** for live flight and hotel contact lookups.
- **ReportLab** for generating a polished PDF itinerary.

The agent is designed as a guided workflow: it interviews the user, generates one or two trip plans, resolves conflicts, gathers preferences, and outputs a final itinerary PDF.

## What This Project Does

`travel_agent.py` runs an end-to-end travel planning flow:
1. Conversationally collects trip intent.
2. Generates structured plans (single or compare mode).
3. Surfaces conflicts and asks for your decisions.
4. Fetches live flight pricing signals via SerpApi.
5. Lets you pick preferred flight and hotel options.
6. Handles multi-city hotel logic, including day-trip return prompts.
7. Generates a travel itinerary PDF and saves session history.

## Core Features

- Natural language travel interview with context memory per session.
- Structured planning output with budget breakdowns.
- Conflict handling before finalizing plans.
- Live flight data lookup (`google_flights` via SerpApi).
- Hotel suggestion + contact enrichment lookup.
- Multi-city accommodation logic:
  - Transit destination extraction (`A to B` -> `B`).
  - Day-trip feasibility check from base city.
  - Explicit user prompt for same-day return (to avoid extra hotel when possible).
  - Safeguard: multi-day city stays always get a hotel.
- Unicode-aware PDF font selection fallback.
- Wider accommodation column + wrapping for long addresses.
- Session history persisted in JSON.

## Repository Layout

- `travel_agent.py`: Main and latest agentic workflow.
- `travel.py`: Earlier/alternate implementation.
- `config.json`: API key config (local secrets file).
- `requirements.txt`: Python dependencies.
- `agent_history.json`: Generated run history.
- `ITINERARY_*.pdf`: Output itinerary files.

## Requirements

- Python **3.10+** (your env: `tf_3_10_new`)
- DeepSeek API key
- SerpApi key
- macOS/Linux terminal (colorized CLI output)

Python packages (already in `requirements.txt`):
- `openai>=1.0.0,<2.0.0`
- `google-search-results>=2.4.2,<3.0.0`
- `reportlab>=4.0.0,<5.0.0`

## Setup

### 1. Create/activate environment

```bash
conda activate tf_3_10_new
```

If needed, create it first:

```bash
conda create -n tf_3_10_new python=3.10 -y
conda activate tf_3_10_new
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API keys

Create/update `config.json`:

```json
{
  "deepseek_api_key": "YOUR_DEEPSEEK_KEY",
  "serpapi_key": "YOUR_SERPAPI_KEY"
}
```

## Run

```bash
python travel_agent.py
```

## End-to-End Runtime Flow

The script follows these phases:

1. **Interview phase**
   - Chat-based discovery until model emits `READY_TO_PLAN` token.

2. **Planning phase**
   - DeepSeek returns structured JSON plans.
   - Optional conflict objects are shown for user resolution.
   - Plans are regenerated after conflict choices.

3. **Plan presentation + approval**
   - Shows plan details, budgets, destination highlights.
   - Displays cheapest live flight signal and adjusted total.
   - User chooses plan (or auto-selects if only one plan remains).

4. **Flight selection**
   - Model proposes 3 flight options; user picks 1/2/3 or `skip`.

5. **Hotel selection**
   - Captures amenity and budget preferences.
   - Builds city sequence from itinerary.
   - For feasible day trips, asks whether to return to base city same day.
   - Forces hotel booking for cities appearing on multiple itinerary days.

6. **PDF generation + history save**
   - Collects traveler/passport info.
   - Builds final itinerary table and accommodation contacts.
   - Writes `ITINERARY_<slug>_<NAME>.pdf`.
   - Appends run data to `agent_history.json`.

## Configuration Notes

- `travel_agent.py` expects `config.json` in project root.
- Missing or placeholder API keys cause immediate exit with a clear message.
- DeepSeek client is initialized with:
  - model: `deepseek-chat`
  - base URL: `https://api.deepseek.com`

## Output Files

- **PDF**: `ITINERARY_<destination_slug>_<TRAVELLER_NAME>.pdf`
- **History**: `agent_history.json`

History entries include:
- timestamp
- destination
- full chosen plan JSON
- selected hotels
- selected flight option

## Unicode and Font Behavior

The PDF renderer attempts Unicode-capable fonts in order:
1. `Arial Unicode` (macOS supplemental)
2. `NotoSans`
3. `DejaVuSans`
4. Fallback to Helvetica if none available

This helps render non-English text (for example Chinese/Arabic names and addresses) more reliably.

## Known Limitations

- Flight and hotel options from the LLM are suggestions; always verify directly before booking.
- SerpApi live data can vary by availability and region.
- Date generation currently defaults to `today + 30 days` for outbound and `+duration` for return.
- Network/API failures are handled, but some branches fall back to placeholders (`TBD`, missing contacts).
- PDF layout still depends on content length and chosen fonts.

## Troubleshooting

### `config.json not found` or missing keys
- Ensure `config.json` exists in repo root with both keys populated.

### `ModuleNotFoundError`
- Reinstall dependencies in the active environment:
  ```bash
  pip install -r requirements.txt
  ```

### SerpApi returns no flights/hotels
- Verify key validity and quota.
- Retry with a different destination/date context.

### PDF text or symbols look wrong
- Ensure Unicode fonts exist on system.
- On macOS, `Arial Unicode.ttf` is typically under `/System/Library/Fonts/Supplemental/`.

### Permission error while compiling Python in sandboxed setups
- Use local cache prefix:
  ```bash
  PYTHONPYCACHEPREFIX=.pycache_local python -m py_compile travel_agent.py
  ```

## Development Tips

- Main file to evolve: `travel_agent.py`.
- Fast checks:
  ```bash
  PYTHONPYCACHEPREFIX=.pycache_local python -m py_compile travel_agent.py
  ```
- Search key sections quickly:
  ```bash
  rg -n "^def |READY_TO_PLAN|conflicts|select_hotels|generate_pdf" travel_agent.py
  ```

## Security Notes

- Keep `config.json` out of version control if it contains real keys.
- Rotate API keys if accidentally exposed.

## Suggested Next Improvements

- Add optional command-line args (origin, dates, budget seed) for reproducible runs.
- Add structured validation for planner JSON before downstream use.
- Add test fixtures for itinerary-to-hotel mapping logic.
- Add caching of API calls to reduce cost and latency.
- Add explicit timezone/date controls instead of relative offsets.


