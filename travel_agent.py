#!/usr/bin/env python3
"""
travel_agent.py - Agentic Travel Planner
Powered by DeepSeek V3 + SerpApi

The agent interviews the user, plans autonomously, pauses only on real conflicts.
Human touchpoints: (1) initial intent, (2) conflict resolution, (3) final approval + passport details.

Usage: python travel_agent.py
"""

import json, re, os, sys
from datetime import datetime, timedelta
from openai import OpenAI
from serpapi import GoogleSearch
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable, PageBreak)
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG_FILE  = "config.json"
HISTORY_FILE = "agent_history.json"

RESET="\033[0m"; BOLD="\033[1m"; DIM="\033[2m"
CYAN="\033[96m"; GREEN="\033[92m"; YELLOW="\033[93m"
BLUE="\033[94m"; MAGENTA="\033[95m"; RED="\033[91m"

def c(text, col): return f"{col}{text}{RESET}"
def divider(title=""):
    if title:
        pad = max(2, (56 - len(title) - 2) // 2)
        print(c(f"\n  {chr(9472)*pad} {title} {chr(9472)*pad}", BLUE))
    else:
        print(c(f"  {chr(9472)*58}", DIM))

def agent_print(msg, col=CYAN):
    print(f"\n  {c('Agent >', BOLD+col)} {msg}")

def agent_think(msg):
    print(c(f"  [{msg}]", DIM+YELLOW))

def load_config():
    if not os.path.exists(CONFIG_FILE):
        print(c(f"\n  config.json not found.\n", YELLOW)); sys.exit(1)
    with open(CONFIG_FILE) as f:
        cfg = json.load(f)
    dk = cfg.get("deepseek_api_key","")
    sk = cfg.get("serpapi_key","")
    if not dk or dk == "your-deepseek-key-here":
        print(c("  Missing deepseek_api_key\n", YELLOW)); sys.exit(1)
    if not sk or sk == "your-serpapi-key-here":
        print(c("  Missing serpapi_key\n", YELLOW)); sys.exit(1)
    return dk, sk

# ── Conversation history (agent memory within session) ────────────────────────
conversation = []

def chat(client, system, user_msg, max_tokens=4000, temp=1.0):
    """Single-turn call with full conversation history for context."""
    conversation.append({"role": "user", "content": user_msg})
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "system", "content": system}] + conversation,
        temperature=temp,
        max_tokens=max_tokens,
    )
    reply = response.choices[0].message.content.strip()
    conversation.append({"role": "assistant", "content": reply})
    return reply

def chat_json(client, system, user_msg, max_tokens=4000):
    """Call DeepSeek expecting clean JSON back."""
    raw = chat(client, system, user_msg, max_tokens=max_tokens)
    raw = re.sub(r"^```json\s*","",raw)
    raw = re.sub(r"^```\s*","",raw)
    raw = re.sub(r"\s*```$","",raw)
    return json.loads(raw)

# ── System prompts ────────────────────────────────────────────────────────────
CONSULTANT_PROMPT = """You are an expert travel consultant AI. Your job is to understand what the traveller wants through natural conversation, then plan the best trip(s) for them.

BEHAVIOUR:
- If the user is vague, ask targeted clarifying questions (destination, budget, duration, travel style, dates, origin city).
- Ask only 2-3 questions at a time — don't overwhelm.
- Once you have enough info, tell the user what you're going to do (e.g. "Let me compare Tokyo and Seoul for you").
- If the user mentions two destinations, compare them. If they mention one, plan the single best trip. If they're open, suggest and compare two options yourself.
- Be warm, concise and helpful. No bullet point lists in conversation — speak naturally.

When you have enough information to start planning, end your message with exactly this token on its own line:
READY_TO_PLAN"""

PLANNER_PROMPT = """You are an expert travel planner. Based on the conversation so far, output a complete travel plan as valid JSON only — no markdown, no backticks, no extra text.

RULES:
- destination_region must be specific city and country (e.g. "Tokyo, Japan" not "East Asia")
- Always provide real 3-letter IATA airport codes
- Keep budgets realistic
- If comparing two trips, include both in the plans array
- Flag any budget conflicts clearly in the conflicts array

Return this exact schema:
{
  "mode": "single" or "compare",
  "plans": [
    {
      "id": "A" or "B",
      "trip_summary": {
        "destination_region": "City, Country",
        "origin_city": "string",
        "origin_iata": "string",
        "duration_days": number,
        "total_budget_usd": number,
        "travel_style": "string",
        "best_travel_months": "string",
        "travel_dates": "string (e.g. May 2026)"
      },
      "top_destinations": [
        {
          "city": "string",
          "country": "string",
          "iata_code": "string",
          "why_visit": "string",
          "estimated_daily_cost_usd": number,
          "highlight": "string"
        }
      ],
      "recommended_itinerary": [
        {
          "day": number,
          "location": "string",
          "morning": "string",
          "afternoon": "string",
          "evening": "string",
          "accommodation_type": "string",
          "daily_budget_usd": number
        }
      ],
      "budget_breakdown": {
        "flights_usd": number,
        "accommodation_usd": number,
        "food_usd": number,
        "activities_usd": number,
        "transport_usd": number,
        "misc_usd": number,
        "total_usd": number
      },
      "money_saving_tips": ["string"],
      "warnings": ["string"]
    }
  ],
  "conflicts": [
    {
      "plan_id": "A" or "B" or "both",
      "issue": "short description",
      "options": ["option 1", "option 2"]
    }
  ],
  "agent_summary": "2-3 sentence natural language summary of what was planned and why"
}"""

HOTEL_PROMPT = """You are a hotel expert. Suggest 3 hotels matching preferences and budget. Respond with valid JSON only.

{
  "hotels": [
    {
      "option": number,
      "name": "Full hotel name",
      "area": "Neighbourhood",
      "stars": number,
      "est_nightly_usd": number,
      "breakfast_included": boolean,
      "room_service": boolean,
      "spa": boolean,
      "near_transport": boolean,
      "why": "One sentence"
    }
  ]
}"""

FLIGHT_PROMPT = """You are a flight expert. Suggest 3 realistic flight options. Respond with valid JSON only.

{
  "flight_options": [
    {
      "option": number,
      "label": "Budget / Best Value / Premium",
      "airline": "string",
      "stops": number,
      "layover_cities": ["string"],
      "approx_duration": "string",
      "estimated_price_usd": number,
      "notes": "string"
    }
  ]
}"""

# ── SerpApi helpers ───────────────────────────────────────────────────────────
def fetch_live_flight(sk, origin, dest_iata, outbound, ret):
    try:
        results = GoogleSearch({
            "engine":"google_flights","departure_id":origin,
            "arrival_id":dest_iata,"outbound_date":outbound,
            "return_date":ret,"currency":"USD","hl":"en","api_key":sk,
        }).get_dict()
        flights = results.get("best_flights") or results.get("other_flights") or []
        if not flights: return {"price":None,"note":"No flights found"}
        best = min(flights, key=lambda x: x.get("price",9999))
        legs = best.get("flights",[{}])
        return {"price":best.get("price"),"airline":legs[0].get("airline","?") if legs else "?",
                "duration":best.get("total_duration"),"stops":len(legs)-1}
    except Exception as e:
        return {"price":None,"note":str(e)}

def fetch_hotel_contact(sk, hotel_name, city):
    try:
        results = GoogleSearch({
            "engine":"google","q":f"{hotel_name} {city} hotel phone address","api_key":sk,
        }).get_dict()
        kg = results.get("knowledge_graph",{})
        phone   = kg.get("phone","")
        address = kg.get("address","")
        if phone or address:
            return {"phone":phone,"address":address}
        organic = results.get("organic_results",[])
        snippet = organic[0].get("snippet","")[:100] if organic else ""
        return {"phone":"","address":snippet}
    except:
        return {"phone":"","address":""}

def fmt_dur(m): return f"{m//60}h {m%60}m" if m else "--"
def cheapest_live(fd):
    prices = [v["price"] for v in fd.values() if v.get("price")]
    return min(prices) if prices else None

def get_pdf_font_names():
    """Prefer a Unicode-capable font to avoid non-English encoding failures in PDFs."""
    candidates = [
        ("ArialUnicode", "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        ("NotoSans", "/System/Library/Fonts/Supplemental/NotoSans-Regular.ttf"),
        ("DejaVuSans", "/Library/Fonts/DejaVuSans.ttf"),
    ]
    registered = set(pdfmetrics.getRegisteredFontNames())
    for name, path in candidates:
        if not os.path.exists(path):
            continue
        if name not in registered:
            try:
                pdfmetrics.registerFont(TTFont(name, path))
            except Exception:
                continue
        return name, name
    return "Helvetica", "Helvetica-Bold"

# ── Phase 1: Agent interview ───────────────────────────────────────────────────
def interview_phase(client):
    divider("TRAVEL AGENT")
    agent_print("Hi! I'm your travel planning agent. Tell me where you'd like to go, "
                "or just describe what kind of trip you have in mind — I'll help narrow it down.", GREEN)
    print()

    while True:
        user_input = input(c("  You > ", BOLD+MAGENTA)).strip()
        if not user_input:
            continue

        reply = chat(client, CONSULTANT_PROMPT, user_input, max_tokens=600, temp=1.0)

        if "READY_TO_PLAN" in reply:
            # Strip token from display
            display = reply.replace("READY_TO_PLAN","").strip()
            if display:
                agent_print(display, GREEN)
            print()
            agent_think("Gathering all details — starting planning phase...")
            return
        else:
            agent_print(reply, GREEN)
            print()

# ── Phase 2: Planning ─────────────────────────────────────────────────────────
def planning_phase(client, sk):
    agent_think("Generating travel plan(s) from DeepSeek...")

    plan_data = chat_json(client, PLANNER_PROMPT,
                          "Based on our conversation, generate the complete travel plan JSON now.")

    plans     = plan_data.get("plans", [])
    conflicts = plan_data.get("conflicts", [])
    summary   = plan_data.get("agent_summary","")
    mode      = plan_data.get("mode","single")

    if summary:
        agent_print(summary, GREEN)

    # ── Resolve conflicts before proceeding ───────────────────────────────────
    if conflicts:
        for conflict in conflicts:
            print()
            divider(f"CONFLICT — PLAN {conflict.get('plan_id','?')}")
            agent_print(f"{conflict['issue']}", YELLOW)
            opts = conflict.get("options",[])
            for i, opt in enumerate(opts, 1):
                print(f"  {c(str(i)+'.', CYAN)} {opt}")
            print()
            pick = input(c("  Your choice > ", BOLD+MAGENTA)).strip()
            # Feed resolution back into conversation so agent can adjust
            resolution = f"For the conflict '{conflict['issue']}', the user chose: {pick}. " \
                         f"Options were: {opts}. Please note this for the final plan."
            chat(client, CONSULTANT_PROMPT, resolution, max_tokens=100)

        # Re-plan with conflict resolutions applied
        agent_think("Adjusting plan based on your choices...")
        plan_data = chat_json(client, PLANNER_PROMPT,
                              "Regenerate the travel plan JSON with all conflict resolutions applied.")
        plans     = plan_data.get("plans", [])
        summary   = plan_data.get("agent_summary","")
        if summary:
            agent_print(summary, GREEN)

    # ── Fetch live flights for each plan ──────────────────────────────────────
    all_flight_data = {}
    for plan in plans:
        pid    = plan["id"]
        ts     = plan["trip_summary"]
        origin = ts["origin_iata"]
        days   = ts["duration_days"]
        out    = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        ret    = (datetime.now() + timedelta(days=30+days)).strftime("%Y-%m-%d")
        fd     = {}
        agent_think(f"Fetching live flights for Plan {pid}: {ts['destination_region']}...")
        for dest in plan["top_destinations"]:
            iata = dest.get("iata_code","")
            city = dest["city"]
            if iata:
                fd[city] = fetch_live_flight(sk, origin, iata, out, ret)
            else:
                fd[city] = {"price":None,"note":"No IATA"}
        all_flight_data[pid] = {"fd":fd, "out":out, "ret":ret}

    return plans, all_flight_data, mode

# ── Phase 3: Present & approve ────────────────────────────────────────────────
def present_plans(plans, all_flight_data, mode):
    for plan in plans:
        pid = plan["id"]
        ts  = plan["trip_summary"]
        b   = plan["budget_breakdown"]
        fd  = all_flight_data[pid]["fd"]
        out = all_flight_data[pid]["out"]
        ret = all_flight_data[pid]["ret"]
        cl  = cheapest_live(fd)
        adj = (b["total_usd"]-b["flights_usd"]+cl) if cl else b["total_usd"]

        divider(f"PLAN {pid} — {ts['destination_region'].upper()}")
        print(f"  {c('Duration:',BOLD)}  {ts['duration_days']} days  |  "
              f"{c('Budget:',BOLD)} ${ts['total_budget_usd']:,}  |  "
              f"{c('Style:',BOLD)} {ts['travel_style']}")
        print(f"  {c('Dates:',BOLD)}    {out} -> {ret}")
        print(f"  {c('Cheapest live flight:',BOLD)} "
              f"{c('$'+str(cl) if cl else 'N/A', GREEN)}  |  "
              f"{c('Adjusted total:',BOLD)} {c('$'+str(adj), GREEN)}")

        print()
        for i, dest in enumerate(plan["top_destinations"],1):
            city = dest["city"]
            f    = fd.get(city,{})
            p    = c(f"${f['price']:,}",GREEN) if f.get("price") else c(f.get("note","N/A"),DIM)
            print(f"  {c(str(i)+'.',CYAN)} {c(city+', '+dest['country'],BOLD)}")
            print(f"     {dest['why_visit']}")
            print(f"     {c(chr(9733)+' Must do:',YELLOW)} {dest['highlight']}")
            print(f"     {c('Live flight:',BOLD)} {p}  |  {f.get('airline','--')}  |  "
                  f"{f.get('stops','-')} stop(s)  |  {fmt_dur(f.get('duration'))}")
            print()

        if plan.get("money_saving_tips"):
            print(f"  {c('Tips:',BOLD)}")
            for tip in plan["money_saving_tips"]:
                print(f"    {c(chr(9679),YELLOW)} {tip}")
        if plan.get("warnings"):
            print(f"  {c('Heads up:',BOLD)}")
            for w in plan["warnings"]:
                print(f"    {c('!',RED)} {w}")

    # Compare summary if two plans
    if mode == "compare" and len(plans) == 2:
        p_a   = plans[0]; p_b = plans[1]
        fd_a  = all_flight_data["A"]["fd"]; fd_b = all_flight_data["B"]["fd"]
        cl_a  = cheapest_live(fd_a); cl_b = cheapest_live(fd_b)
        b_a   = p_a["budget_breakdown"]; b_b = p_b["budget_breakdown"]
        adj_a = (b_a["total_usd"]-b_a["flights_usd"]+cl_a) if cl_a else b_a["total_usd"]
        adj_b = (b_b["total_usd"]-b_b["flights_usd"]+cl_b) if cl_b else b_b["total_usd"]
        winner = "A" if adj_a <= adj_b else "B"
        diff   = abs(adj_a-adj_b)
        divider("AGENT RECOMMENDATION")
        agent_print(
            f"Plan {winner} is cheaper by ${diff:,} "
            f"({'$'+str(adj_a) if winner=='A' else '$'+str(adj_b)} adjusted total). "
            f"I recommend Plan {winner} based on your budget.", GREEN)

# ── Phase 4: Approval ─────────────────────────────────────────────────────────
def approval_phase(plans, mode):
    print()
    divider("YOUR APPROVAL")

    available_ids = [p["id"] for p in plans]

    # If only one plan remains after conflict resolution, auto-select it
    if len(plans) == 1:
        chosen = plans[0]
        agent_print(f"Proceeding with Plan {chosen['id']}: "
                    f"{chosen['trip_summary']['destination_region']}.", GREEN)
        return chosen

    # Build a quick summary for context
    plan_summaries = {p["id"]: p["trip_summary"]["destination_region"] for p in plans}
    summary_str = "  |  ".join(f"Plan {k}: {v}" for k,v in plan_summaries.items())
    agent_print(f"Here's what I planned — {summary_str}", GREEN)
    agent_print("Which would you like to proceed with?", GREEN)
    print()

    while True:
        reply = input(c("  You > ", BOLD+MAGENTA)).strip()
        reply_upper = reply.upper()

        # Direct A/B pick
        for pid in available_ids:
            if pid in reply_upper or plan_summaries[pid].split(",")[0].lower() in reply.lower():
                chosen = next(p for p in plans if p["id"]==pid)
                agent_print(f"Plan {pid} — {plan_summaries[pid]}. Let me set up your itinerary.", GREEN)
                return chosen

        # User asking a question about a plan
        if "?" in reply or any(w in reply.lower() for w in ["what","tell me","explain","describe"]):
            # Answer naturally using plan data
            for pid in available_ids:
                if pid in reply_upper or plan_summaries[pid].split(",")[0].lower() in reply.lower():
                    p = next(p for p in plans if p["id"]==pid)
                    ts = p["trip_summary"]
                    dests = ", ".join(d["city"] for d in p["top_destinations"])
                    agent_print(
                        f"Plan {pid} is {ts['duration_days']} days in {ts['destination_region']}. "
                        f"Destinations: {dests}. Style: {ts['travel_style']}. "
                        f"Budget: ${ts['total_budget_usd']:,}. Best months: {ts['best_travel_months']}.",
                        GREEN)
                    agent_print("Would you like to go with this plan?", GREEN)
                    break
            else:
                # General question — answer about all plans
                for pid, region in plan_summaries.items():
                    p = next(pl for pl in plans if pl["id"]==pid)
                    dests = ", ".join(d["city"] for d in p["top_destinations"])
                    agent_print(f"Plan {pid}: {region} — covering {dests}. "
                                f"Style: {p['trip_summary']['travel_style']}.", GREEN)
                agent_print("Which one interests you more?", GREEN)
            continue

        # ok / yes → first plan
        if reply.lower() in ("ok","yes","sure","proceed","looks good","confirmed"):
            chosen = plans[0]
            agent_print(f"Great — going with Plan {chosen['id']}.", GREEN)
            return chosen

        agent_print(
            f"Just type A or B to choose — "
            f"or ask me anything about either plan.", YELLOW)

# ── Phase 5: Flight + Hotel selection ─────────────────────────────────────────

DAY_TRIP_PROMPT = """You are a geography and transport expert. Answer with valid JSON only.

Given a base city and a day-trip city, determine if someone can comfortably travel there and back in one day using public transport.

{
  "is_day_trip": true or false,
  "reason": "one sentence explanation",
  "transport": "e.g. train, metro, bus"
}"""

def extract_transit_destination(loc):
    """From 'Sapa to Hanoi' extract 'Hanoi'. Returns None if not a transit loc."""
    loc_lower = loc.lower().strip()
    if " to " in loc_lower:
        return loc.split(" to ")[-1].strip()
    if " - " in loc_lower:
        return loc.split(" - ")[-1].strip()
    return None

def is_day_trip(client, base_city, visit_city, country):
    """Ask DeepSeek if visit_city is a feasible day trip from base_city."""
    try:
        raw = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": DAY_TRIP_PROMPT},
                {"role": "user", "content":
                 f"Base city: {base_city}, {country}. "
                 f"Day trip destination: {visit_city}. "
                 f"Can someone visit {visit_city} as a day trip from {base_city} "
                 f"using public transport and return the same evening?"}
            ],
            temperature=0.5, max_tokens=200,
        ).choices[0].message.content.strip()
        raw = re.sub(r"^```json\s*","",raw)
        raw = re.sub(r"^```\s*","",raw)
        raw = re.sub(r"\s*```$","",raw)
        result = json.loads(raw)
        return result.get("is_day_trip", False), result.get("reason",""), result.get("transport","")
    except Exception:
        return False, "", ""

def select_flight(client, chosen, all_flight_data):
    ts  = chosen["trip_summary"]
    pid = chosen["id"]
    fd  = all_flight_data[pid]
    out = fd["out"]; ret = fd["ret"]
    origin = ts["origin_city"]
    region = ts["destination_region"]

    agent_think(f"Generating flight options: {origin} -> {region}...")
    raw = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role":"system","content":FLIGHT_PROMPT},
            {"role":"user","content":
             f"3 flight options from {origin} to {region}. "
             f"Outbound {out}, return {ret}. Budget/mid/premium."}
        ],
        temperature=1.0, max_tokens=800,
    ).choices[0].message.content.strip()
    raw = re.sub(r"^```json\s*","",raw); raw = re.sub(r"^```\s*","",raw)
    raw = re.sub(r"\s*```$","",raw)
    try:
        options = json.loads(raw).get("flight_options",[])
    except:
        options = []

    if not options:
        agent_print("Couldn't fetch flight options — skipping.", YELLOW)
        return None

    divider("FLIGHT OPTIONS")
    agent_print(f"Here are your flight options from {origin} to {region}:\n", GREEN)
    for opt in options:
        via = ", ".join(opt.get("layover_cities",[])) or "Direct"
        print(f"  {c(str(opt['option'])+'.',CYAN)} {c(opt['label'],BOLD)}  —  {opt['airline']}")
        print(f"     {opt['approx_duration']}  |  {opt['stops']} stop(s)  |  Via: {via}")
        print(f"     {c('~$'+str(opt['estimated_price_usd']),GREEN)}  —  {c(opt['notes'],DIM)}\n")

    while True:
        pick = input(c("  You > ",BOLD+MAGENTA)).strip()
        if pick in ("1","2","3"):
            chosen_flight = options[int(pick)-1]
            agent_print(f"Selected: {chosen_flight['label']} — {chosen_flight['airline']} "
                        f"~${chosen_flight['estimated_price_usd']:,}", GREEN)
            return chosen_flight
        if pick.lower() == "skip":
            return None
        agent_print("Please type 1, 2, 3, or 'skip'.", YELLOW)

def select_hotels(client, sk, chosen):
    ts     = chosen["trip_summary"]
    budget = ts["total_budget_usd"]
    days   = ts["duration_days"]

    # Ask preferences once
    divider("HOTEL PREFERENCES")
    agent_print("A few quick questions about your hotel preferences:", GREEN)
    print()
    def yn(q):
        return input(c(f"  {q} (y/n) > ", BOLD)).strip().lower() == "y"
    prefs = {
        "breakfast":      yn("Breakfast included?"),
        "room_service":   yn("Room service?"),
        "spa":            yn("Spa / wellness?"),
        "near_transport": yn("Close to public transport?"),
    }
    print()
    raw_budget = input(c("  Daily budget per person (USD) > ", BOLD)).strip()
    clean = re.sub(r"[^0-9]","",raw_budget)
    daily = int(clean) if clean else None
    if daily:
        agent_print(f"Got it — ${daily}/day budget. Finding hotels within that.", GREEN)
    prefs["daily_budget_usd"] = daily

    nightly = int(daily*0.5) if daily else int(budget*0.3/max(days,1))
    pref_str = ", ".join(k for k in ["breakfast","room_service","spa","near_transport"] if prefs.get(k)) or "no specific amenities"

    # Build smart city list:
    # - Skip pure transit labels ("Departure", "In Transit" etc.)
    # - For "City A to City B" days, book hotel in City B
    # - For day trips (small country, good transport), reuse base city hotel
    skip_words = {"departure","in transit","transit","flight","airport","return","back"}

    def is_transit_only(loc):
        ll = loc.lower().strip()
        return (ll in skip_words or any(w in ll for w in skip_words)
                or ll.startswith("to "))

    dest_map     = {d["city"]:d["country"] for d in chosen["top_destinations"]}
    country_main = next(iter(dest_map.values()),"")
    hotel_map    = {}   # city -> {name, tel, address}

    # Build ordered list of unique real cities plus how many itinerary days each city appears.
    # Multi-day cities should always get a hotel (not treated as a day trip).
    city_sequence = []
    city_day_counts = {}
    for day in chosen["recommended_itinerary"]:
        loc = day["location"]

        # Pure skip
        if is_transit_only(loc):
            continue

        # Transit "A to B" — real city is B
        dest = extract_transit_destination(loc)
        real_city = dest if dest else loc
        city_day_counts[real_city] = city_day_counts.get(real_city, 0) + 1

        if real_city not in [c for c,_ in city_sequence]:
            city_sequence.append((real_city, True))
        
    # Now for each real city, check if it's a day trip from the previous base
    final_cities = []  # list of (city, book_hotel)
    base_city = None
    for city, _ in city_sequence:
        if base_city is None:
            final_cities.append((city, True))
            base_city = city
            continue

        stay_days = city_day_counts.get(city, 1)
        if stay_days > 1:
            agent_print(
                f"{city} appears on {stay_days} days in your itinerary, "
                f"so I’ll book a hotel there instead of a same-day return.",
                GREEN,
            )
            final_cities.append((city, True))
            base_city = city
            continue

        country = dest_map.get(city, dest_map.get(base_city, country_main))
        day_trip, reason, transport = is_day_trip(client, base_city, city, country)
        if day_trip:
            agent_print(
                f"{city} looks feasible as a same-day trip from {base_city} "
                f"via {transport or 'public transport'}.",
                GREEN)
            if reason:
                agent_print(f"Why this could work: {reason}", DIM)
            while True:
                same_day = input(
                    c(f"  Return to {base_city} the same day to avoid a new hotel? (y/n) > ", BOLD)
                ).strip().lower()
                if same_day in ("y", "yes", "n", "no"):
                    break
                agent_print("Please type y or n.", YELLOW)
            if same_day in ("y", "yes"):
                agent_print(f"Great — I’ll keep your hotel in {base_city} and treat {city} as a day trip.", GREEN)
                final_cities.append((city, False))
            else:
                agent_print(f"Understood — I’ll add a hotel in {city}.", GREEN)
                final_cities.append((city, True))
                base_city = city
        else:
            final_cities.append((city, True))
            base_city = city

    for city, needs_hotel in final_cities:
        country = dest_map.get(city,"")
        if not needs_hotel:
            # Reuse base hotel — find the most recent booked hotel
            base = next((h for c,b in reversed(final_cities[:final_cities.index((city,False))])
                         if b and (h:=hotel_map.get(c)) is not None), None)
            if base:
                hotel_map[city] = base
            continue

        divider(f"HOTELS — {city.upper()}")
        agent_think(f"Finding hotels in {city} matching preferences...")

        raw = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role":"system","content":HOTEL_PROMPT},
                {"role":"user","content":
                 f"3 hotels in {city}, {country}. Preferences: {pref_str}. "
                 f"Max nightly: ${nightly} USD. Budget/mid/best-value options."}
            ],
            temperature=1.0, max_tokens=800,
        ).choices[0].message.content.strip()
        raw = re.sub(r"^```json\s*","",raw); raw = re.sub(r"^```\s*","",raw)
        raw = re.sub(r"\s*```$","",raw)
        try:
            hotels = json.loads(raw).get("hotels",[])
        except:
            hotels = []

        if not hotels:
            hotel_map[city] = {"name":"TBD","tel":"","address":""}
            continue

        print()
        for h in hotels:
            ams = [k for k in ["breakfast_included","room_service","spa","near_transport"] if h.get(k)]
            am_str = "  |  ".join(a.replace("_"," ").title() for a in ams) or "Standard"
            print(f"  {c(str(h['option'])+'.',CYAN)} {c(h['name'],BOLD)}  ({h.get('stars',0)}{chr(9733)})  —  {h['area']}")
            print(f"     ~${h['est_nightly_usd']}/night  |  {am_str}")
            print(f"     {c(h['why'],DIM)}\n")

        while True:
            pick = input(c(f"  Pick hotel for {city} (1/2/3) > ", BOLD+CYAN)).strip()
            if pick in ("1","2","3"):
                h = hotels[int(pick)-1]
                hname = h["name"]
                agent_think(f"Looking up contact for {hname}...")
                contact = fetch_hotel_contact(sk, hname, city)
                tel     = contact.get("phone","") or "To be confirmed"
                address = contact.get("address","") or "To be confirmed"
                if tel != "To be confirmed":
                    agent_print(f"Found: {hname}  {tel}", GREEN)
                else:
                    agent_print(f"Selected: {hname} — phone not found, marked TBC.", YELLOW)
                hotel_map[city] = {"name":hname,"tel":tel,"address":address}
                break
            agent_print("Please type 1, 2, or 3.", YELLOW)

    return hotel_map, prefs

# ── Phase 6: Generate itinerary PDF ──────────────────────────────────────────
def generate_pdf(chosen, all_flight_data, hotel_map, chosen_flight, daily_budget_usd):
    divider("APPLICANT DETAILS")
    agent_print("Last step — just need your details for the itinerary document.", GREEN)
    print()
    applicant_name = input(c("  Full name       > ", BOLD)).strip() or "Traveller"
    passport_no    = input(c("  Passport number > ", BOLD)).strip() or "N/A"

    ts       = chosen["trip_summary"]
    region   = ts["destination_region"]
    days     = ts["duration_days"]
    pid      = chosen["id"]
    outbound = all_flight_data[pid]["out"]

    start_dt = datetime.strptime(outbound,"%Y-%m-%d")
    end_dt   = start_dt + timedelta(days=days-1)
    period   = f"{start_dt.strftime('%B %-d, %Y')} \u2013 {end_dt.strftime('%B %-d, %Y')}"
    slug     = re.sub(r"[^a-z0-9]+","_",region.lower()).strip("_")[:30]
    filename = f"ITINERARY_{slug}_{applicant_name.replace(' ','_').upper()}.pdf"

    DARK=colors.HexColor("#1a1a2e"); ACCENT=colors.HexColor("#0077b6")
    LIGHT=colors.HexColor("#f0f4f8"); WHITE=colors.white; MID=colors.HexColor("#dce3ea")
    GREEN_C=colors.HexColor("#2d6a4f")
    font_regular, font_bold = get_pdf_font_names()

    def sty(name,**kw): return ParagraphStyle(name,**kw)
    title_sty = sty("TI",fontSize=18,textColor=DARK,fontName=font_bold,alignment=TA_CENTER,spaceAfter=2)
    lbl_sty   = sty("LB",fontSize=9,textColor=colors.HexColor("#555"),fontName=font_bold)
    body_sty  = sty("BD",fontSize=9,textColor=DARK,fontName=font_regular,leading=13,wordWrap="CJK",splitLongWords=1)
    small_sty = sty("SM",fontSize=8,textColor=DARK,fontName=font_regular,leading=12,wordWrap="CJK",splitLongWords=1)
    hdr_sty   = sty("HD",fontSize=9,textColor=WHITE,fontName=font_bold,alignment=TA_CENTER)
    note_sty  = sty("NT",fontSize=7,textColor=colors.HexColor("#888"),fontName=font_regular,wordWrap="CJK",splitLongWords=1)

    doc   = SimpleDocTemplate(filename,pagesize=A4,
                              leftMargin=15*mm,rightMargin=15*mm,
                              topMargin=15*mm,bottomMargin=15*mm)
    story = []; W=A4[0]-30*mm

    story.append(Paragraph("TRAVEL ITINERARY",title_sty))
    story.append(HRFlowable(width=W,thickness=2,color=ACCENT,spaceAfter=6))

    first_hotel = next(iter(hotel_map.values()),{}) if hotel_map else {}
    info_data = [
        [Paragraph("Name of Applicant:",lbl_sty), Paragraph(applicant_name,body_sty),
         Paragraph("Passport No:",lbl_sty),        Paragraph(passport_no,body_sty)],
        [Paragraph("Travel Period:",lbl_sty),       Paragraph(period,body_sty),
         Paragraph("Destination:",lbl_sty),         Paragraph(region,body_sty)],
    ]
    if chosen_flight:
        fl_str = (f"{chosen_flight.get('airline','--')}  |  "
                  f"{chosen_flight.get('approx_duration','--')}  |  "
                  f"~${chosen_flight.get('estimated_price_usd',0):,}  |  "
                  f"{chosen_flight.get('stops',0)} stop(s)")
        info_data.append([Paragraph("Flight:",lbl_sty),Paragraph(fl_str,body_sty),
                          Paragraph("",lbl_sty),Paragraph("",body_sty)])
    if daily_budget_usd:
        info_data.append([Paragraph("Daily Budget:",lbl_sty),
                          Paragraph(f"${daily_budget_usd:,} USD per person",body_sty),
                          Paragraph("",lbl_sty),Paragraph("",body_sty)])

    info_tbl = Table(info_data,colWidths=[38*mm,55*mm,30*mm,55*mm])
    info_tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),LIGHT),("TOPPADDING",(0,0),(-1,-1),5),
        ("BOTTOMPADDING",(0,0),(-1,-1),5),("LEFTPADDING",(0,0),(-1,-1),6),
        ("GRID",(0,0),(-1,-1),0.4,MID),
    ]))
    story += [info_tbl, Spacer(1,6*mm)]

    # Build per-city accom map
    skip_words = {"departure","in transit","transit","flight","airport","return","back"}
    def is_transit_loc(loc):
        ll = loc.lower().strip()
        return (ll in skip_words or any(w in ll for w in skip_words)
                or ll.startswith("to "))
    def resolve_city(loc):
        """For transit days like 'Sapa to Hanoi', return the destination city."""
        if " to " in loc:  return loc.split(" to ")[-1].strip()
        if " - " in loc:   return loc.split(" - ")[-1].strip()
        return loc
    accom_map = {}
    for day in chosen["recommended_itinerary"]:
        loc = day["location"]
        if loc not in accom_map:
            real = resolve_city(loc)
            h = hotel_map.get(real, hotel_map.get(loc,{})) if not is_transit_loc(loc) else {}
            if h.get("name") and h["name"] != "TBD":
                txt = h["name"]
                if h.get("tel","") not in ("","To be confirmed","TBD"):
                    txt += f"\n(Tel: {h['tel']})"
                if h.get("address","") not in ("","To be confirmed","TBD"):
                    txt += f"\n{h['address']}"
                accom_map[loc] = txt
            else:
                accom_map[loc] = day.get("accommodation_type","TBD")

    def day_date(n): return (start_dt+timedelta(days=n-1)).strftime("%b %-d")

    col_widths=[18*mm,28*mm,78*mm,54*mm]
    rows=[[Paragraph("Date",hdr_sty),Paragraph("Location",hdr_sty),
           Paragraph("Activity / Schedule",hdr_sty),Paragraph("Accommodation / Contact",hdr_sty)]]

    last_day = chosen["recommended_itinerary"][-1]["day"]
    for day in chosen["recommended_itinerary"]:
        acts = []
        if day.get("morning"):   acts.append(f"\u2022 Morning: {day['morning']}")
        if day.get("afternoon"): acts.append(f"\u2022 Afternoon: {day['afternoon']}")
        if day.get("evening"):   acts.append(f"\u2022 Evening: {day['evening']}")
        loc = day["location"]
        real_loc = resolve_city(loc)
        is_skip  = is_transit_loc(loc) and real_loc == loc  # only skip if no dest extracted
        is_last = day["day"] == last_day
        if is_last:
            accom_cell = "Check-out / Departure"
        elif is_transit_loc(loc) and resolve_city(loc) != loc:
            # Transit day like "Sapa to Hanoi" — show destination hotel
            accom_cell = accom_map.get(resolve_city(loc), "Hotel at destination")
        elif is_skip:
            accom_cell = "In Transit"
        else:
            accom_cell = accom_map.get(loc,"TBD")
        rows.append([Paragraph(day_date(day["day"]),small_sty),
                     Paragraph(loc,small_sty),
                     Paragraph("\n".join(acts),small_sty),
                     Paragraph(accom_cell,small_sty)])

    it = Table(rows,colWidths=col_widths,repeatRows=1)
    it.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),ACCENT),("TEXTCOLOR",(0,0),(-1,0),WHITE),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,0),9),
        ("FONTNAME",(0,1),(-1,-1),"Helvetica"),("FONTSIZE",(0,1),(-1,-1),8),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[LIGHT,WHITE]),
        ("GRID",(0,0),(-1,-1),0.4,MID),("VALIGN",(0,0),(-1,-1),"TOP"),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING",(0,0),(-1,-1),5),
    ]))
    story += [it, Spacer(1,6*mm)]
    story.append(Paragraph(
        "Note: Hotel bookings and tour tickets should be confirmed directly with the provider before travel. "
        "Group tour bookings to be arranged upon arrival. "
        "This document was generated for travel planning purposes.",
        note_sty))
    doc.build(story)
    return filename

# ── Save session history ───────────────────────────────────────────────────────
def save_history(chosen, hotel_map, chosen_flight):
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            history = json.load(f)
    history.append({
        "date": datetime.now().isoformat(),
        "destination": chosen["trip_summary"]["destination_region"],
        "plan": chosen,
        "hotels": hotel_map,
        "flight": chosen_flight,
    })
    with open(HISTORY_FILE,"w") as f:
        json.dump(history,f,indent=2)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print()
    print(c("  =========================================", CYAN))
    print(c("   AGENTIC TRAVEL PLANNER  (DeepSeek V3)", BOLD+CYAN))
    print(c("   Human touchpoints: 3  |  Agent handles: everything else", DIM))
    print(c("  =========================================", CYAN))
    print()

    dk, sk = load_config()
    client = OpenAI(api_key=dk, base_url="https://api.deepseek.com")

    # ── Touchpoint 1: Intent ──────────────────────────────────────────────────
    interview_phase(client)

    # ── Agent runs autonomously ───────────────────────────────────────────────
    plans, all_flight_data, mode = planning_phase(client, sk)
    print()
    present_plans(plans, all_flight_data, mode)

    # ── Touchpoint 2: Approval ────────────────────────────────────────────────
    chosen = approval_phase(plans, mode)

    # ── Agent selects flight + hotels ─────────────────────────────────────────
    chosen_flight        = select_flight(client, chosen, all_flight_data)
    hotel_map, prefs     = select_hotels(client, sk, chosen)
    daily_budget_usd     = prefs.get("daily_budget_usd")

    # ── Touchpoint 3: Passport + PDF ─────────────────────────────────────────
    print()
    agent_think("Generating your itinerary PDF...")
    pdf_file = generate_pdf(chosen, all_flight_data, hotel_map,
                            chosen_flight, daily_budget_usd)
    save_history(chosen, hotel_map, chosen_flight)

    print()
    agent_print(f"All done! Your itinerary is saved as: {c(pdf_file, BOLD+GREEN)}", GREEN)
    agent_print("Safe travels!", GREEN)
    print()

if __name__ == "__main__":
    main()
