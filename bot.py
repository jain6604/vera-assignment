#!/usr/bin/env python3
"""
magicpin AI Challenge — Vera Bot
=================================
A production-grade merchant engagement bot that composes highly specific,
category-appropriate, compulsive WhatsApp messages using Claude.

Architecture:
  - 5 required endpoints (healthz, metadata, context, tick, reply)
  - 4-context composer: category + merchant + trigger + customer
  - Routing layer: trigger kind → prompt variant
  - Auto-reply detection (exits after 3 auto-replies)
  - Intent-transition detection (commits → action mode instantly)
  - Hostile graceful exit
  - Anti-repetition tracking
  - Suppression key deduplication

Run: uvicorn bot:app --host 0.0.0.0 --port 8080
"""

import os
import time
import json
import uuid
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import anthropic

# ─── APP & STATE ──────────────────────────────────────────────────────────────

app = FastAPI(title="Vera Challenge Bot", version="2.0.0")
START_TIME = time.time()

# (scope, context_id) → {version: int, payload: dict}
contexts: Dict[Tuple[str, str], Dict] = {}

# conversation_id → list of turn dicts
conversations: Dict[str, List[Dict]] = {}

# suppression keys already fired — prevents duplicate sends
fired_suppression_keys: set = set()

# merchant_id → list of sent message bodies (for anti-repetition)
merchant_sent_bodies: Dict[str, List[str]] = {}

# Initialize Anthropic client (reads ANTHROPIC_API_KEY from env)
claude = anthropic.Anthropic()
CLAUDE_MODEL = "claude-sonnet-4-20250514"

# ─── REQUEST MODELS ───────────────────────────────────────────────────────────

class ContextBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: Dict[str, Any]
    delivered_at: str

class TickBody(BaseModel):
    now: str
    available_triggers: List[str] = []

class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def get_ctx(scope: str, cid: str) -> Optional[Dict]:
    entry = contexts.get((scope, cid))
    return entry["payload"] if entry else None

def count_contexts() -> Dict[str, int]:
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _) in contexts:
        counts[scope] = counts.get(scope, 0) + 1
    return counts

def find_merchant_category(merchant: Dict) -> Optional[Dict]:
    """Find category context for a merchant."""
    slug = merchant.get("category_slug", "")
    cat = get_ctx("category", slug)
    if cat:
        return cat
    # Fallback: pick any loaded category
    for (scope, cid), entry in contexts.items():
        if scope == "category":
            return entry["payload"]
    return None

def merchants_in_category(slug: str) -> List[Dict]:
    """Return all merchant payloads that belong to a given category slug."""
    result = []
    for (scope, cid), entry in contexts.items():
        if scope == "merchant":
            m = entry["payload"]
            if m.get("category_slug", "") == slug:
                result.append(m)
    return result

# ─── DETECTION LOGIC ──────────────────────────────────────────────────────────

AUTO_REPLY_PHRASES = [
    "thank you for contacting", "thank you for reaching out",
    "we will get back to you", "we will respond shortly",
    "our team will respond", "this is an automated",
    "auto-reply", "automatic reply", "autoresponder",
    "out of office", "we received your message", "we have received",
    "please do not reply to this message",
]

def is_auto_reply(msg: str) -> bool:
    low = msg.lower()
    return any(p in low for p in AUTO_REPLY_PHRASES)

INTENT_COMMIT_PHRASES = [
    "let's do it", "lets do it", "ok let's", "ok lets",
    "okay lets", "okay let's", "yes please", "yes, please",
    "haan bilkul", "bilkul", "zaroor", "kar do", "karo",
    "go ahead", "go for it", "proceed", "confirm",
    "sure, do it", "sure do it", "yes do it",
    "start karo", "shuru karo", "theek hai",
    "what's next", "whats next", "next step", "next steps",
    "done deal", "agreed", "sounds good",
]

def is_intent_commit(msg: str) -> bool:
    low = msg.lower().strip()
    # Strong single-word commits
    strong_singles = {"yes", "haan", "ha", "done", "ok", "okay", "sure", "yep", "yup", "absolutely"}
    if low in strong_singles:
        return True
    return any(p in low for p in INTENT_COMMIT_PHRASES)

HOSTILE_PHRASES = [
    "stop messaging", "stop texting", "stop contacting",
    "don't message", "dont message", "don't contact",
    "this is spam", "useless spam", "not interested",
    "unsubscribe", "remove me", "block", "report",
    "mat karo", "band karo", "chodo", "chhodo",
    "nahi chahiye", "nahin chahiye", "bezkar", "bakwaas",
    "annoying", "irritating", "harassment",
]

def is_hostile(msg: str) -> bool:
    low = msg.lower()
    return any(p in low for p in HOSTILE_PHRASES)

def count_auto_replies_in_conv(conv: List[Dict], from_role: str) -> int:
    return sum(1 for t in conv if t.get("from") == from_role and is_auto_reply(t.get("msg", "")))

# ─── VOICE PROFILES ───────────────────────────────────────────────────────────

VOICE_PROFILES = {
    "peer_clinical": (
        "Write as a peer professional (dentist-to-dentist). "
        "Clinical vocabulary welcome: fluoride varnish, caries, recall, prophylaxis, etc. "
        "Never say 'cure' or 'guaranteed'. Cite sources (journal, page). No hype. Peer tone."
    ),
    "warm_friendly": (
        "Warm, friendly, practical. Like a knowledgeable friend talking salon business. "
        "Use ₹ prices. Relatable, not corporate."
    ),
    "operator_to_operator": (
        "Operator-to-operator. Direct, numbers-first. Like a business consultant to a restaurant owner. "
        "No fluff. Leads with the stat or the opportunity."
    ),
    "coaching": (
        "Motivational, results-focused. Like a gym coach to a gym owner. "
        "Energy without hype. Numbers, wins, actions."
    ),
    "trustworthy_precise": (
        "Trustworthy, precise. Like a pharmacist. Exact numbers. "
        "Never make unverifiable claims. Regulatory awareness."
    ),
}

def get_voice_instruction(tone: str) -> str:
    return VOICE_PROFILES.get(tone, "Professional, direct, WhatsApp-friendly. Short sentences.")

# ─── LANGUAGE DETECTION ───────────────────────────────────────────────────────

def get_lang_instruction(languages: List[str]) -> str:
    langs = [l.lower() for l in languages]
    if "hi" in langs or "hi-en" in langs or "hinglish" in langs:
        return (
            "LANGUAGE: Hindi-English mix (Hinglish). Natural mix like a professional WhatsApp message. "
            "Example: 'Dr. Meera, aapki CTR peer se thodi kam hai — 2.1% vs 3.0% average. "
            "Ek quick photo update se fark pad sakta hai. Post karein?'"
        )
    if "hi" in " ".join(langs):
        return "LANGUAGE: Hindi-English code-mix where natural. Mostly English, sprinkle Hindi for warmth."
    return "LANGUAGE: English. Clear, WhatsApp-friendly. No jargon."

# ─── TRIGGER ROUTING ──────────────────────────────────────────────────────────

TRIGGER_FRAMES = {
    "research_digest": (
        "Frame as 'new research just dropped that's directly relevant to your patients/customers.' "
        "Lead with the specific finding and its magnitude. Offer to pull the abstract + draft patient content."
    ),
    "recall_due": (
        "Frame as 'a specific patient's recall window is open.' "
        "Name the patient, the time since last visit, and the exact slot(s) available. "
        "Offer to send the reminder on their behalf."
    ),
    "perf_spike": (
        "Lead with the exact spike numbers vs baseline. "
        "Ask a curiosity hook: 'want to see what drove it?' or 'want to capitalize before it fades?'"
    ),
    "perf_dip": (
        "Lead with the exact dip (%, absolute). Show loss aversion. "
        "Offer one concrete fix (photo update, post, offer activation). Make it low-effort."
    ),
    "milestone_reached": (
        "Celebrate the milestone with the exact number. "
        "Use social proof: 'You're now in the top X% of [category] in [locality].' "
        "Offer the next milestone nudge."
    ),
    "dormant_with_vera": (
        "Acknowledge the gap, don't guilt-trip. "
        "Lead with one interesting fact or stat about their account they might not know. "
        "Ask a low-commitment question."
    ),
    "festival_upcoming": (
        "Name the festival and exact days remaining. "
        "Give a specific service+price offer that works for this festival. "
        "Social proof: 'X [category] merchants in [locality] are running this already.'"
    ),
    "review_theme_emerged": (
        "Name the specific theme from reviews (e.g., 'wait time'). "
        "Show them the exact count ('3 reviews this week mention...'). "
        "Offer to help draft a response or fix."
    ),
    "competitor_opened": (
        "Be strategic, not alarmist. "
        "Exact distance, exact GBP rating if known. "
        "Offer one proactive move (a post, an offer, a review ask)."
    ),
    "category_trend_movement": (
        "Lead with the search trend number (e.g., '+62% YoY'). "
        "Connect it to their specific local context. "
        "Offer to help capture the demand (post, content, offer)."
    ),
    "scheduled_recurring": (
        "Keep it fresh — a curiosity question or a genuine peer insight. "
        "Not a reminder. Something they'd actually want to read on a Friday morning."
    ),
    "customer_lapsed_soft": (
        "Remind the merchant that a specific customer's recall window opened. "
        "Give the customer name (if available) and time since last visit. "
        "Offer to draft the recall WhatsApp for them."
    ),
    "appointment_tomorrow": (
        "Mention tomorrow's appointment. "
        "Offer pre-visit prep content or a reminder to the customer."
    ),
    "regulation_change": (
        "Name the specific regulation change and its effective date. "
        "Explain the direct impact on their practice. "
        "Offer to help them prepare or inform their patients."
    ),
    "weather_heatwave": (
        "Name the temperature and the relevant service opportunity. "
        "Keep it short and seasonal. Social proof if available."
    ),
    "local_news_event": (
        "Name the event and the direct business opportunity or threat. "
        "Offer one concrete action to take advantage of it."
    ),
}

def get_trigger_frame(kind: str) -> str:
    return TRIGGER_FRAMES.get(kind, (
        "Lead with the most specific, verifiable fact from the trigger payload. "
        "Connect to the merchant's current state. Single CTA at the end."
    ))

# ─── LLM COMPOSER ─────────────────────────────────────────────────────────────

def build_compose_system(category: Dict, merchant: Dict, trigger: Dict,
                          customer: Optional[Dict] = None) -> str:
    slug = category.get("slug", "general")
    voice = category.get("voice", {})
    tone = voice.get("tone", "professional")
    taboos = voice.get("taboos", voice.get("vocab_taboo", []))
    allowed = voice.get("vocab_allowed", [])
    peer_stats = category.get("peer_stats", {})
    offer_catalog = category.get("offer_catalog", [])
    digest = category.get("digest", [])[:3]
    seasonal = category.get("seasonal_beats", [])
    trends = category.get("trend_signals", [])[:2]
    content_lib = category.get("patient_content_library", [])[:2]

    identity = merchant.get("identity", {})
    m_name = identity.get("name", "the merchant")
    owner = identity.get("owner_first_name", "")
    locality = identity.get("locality", "")
    city = identity.get("city", "")
    languages = identity.get("languages", ["en"])
    verified = identity.get("verified", False)

    perf = merchant.get("performance", {})
    views = perf.get("views", "?")
    calls = perf.get("calls", "?")
    ctr = perf.get("ctr", "?")
    v_delta = perf.get("views_delta_7d", perf.get("views_delta", "?"))
    c_delta = perf.get("calls_delta_7d", perf.get("calls_delta", "?"))

    sub = merchant.get("subscription", {})
    offers = merchant.get("offers", [])
    active_offers = [o for o in offers if o.get("status") == "active"]
    expired_offers = [o for o in offers if o.get("status") in ("expired", "paused")]
    signals = merchant.get("signals", [])
    cust_agg = merchant.get("customer_aggregate", {})
    conv_hist = merchant.get("conversation_history", {})
    last_engaged = conv_hist.get("last_merchant_reply_at", "?")

    trg_kind = trigger.get("kind", "")
    trg_payload = trigger.get("payload", {})
    trg_urgency = trigger.get("urgency", 3)
    trg_scope = trigger.get("scope", "merchant")

    voice_inst = get_voice_instruction(tone)
    lang_inst = get_lang_instruction(languages)
    trigger_frame = get_trigger_frame(trg_kind)

    customer_block = ""
    if customer:
        cust_id = customer.get("identity", {})
        lapse = customer.get("lapse_state", {})
        customer_block = f"""
CUSTOMER CONTEXT (message is going TO this customer, from merchant's number):
  Name: {cust_id.get("name", "?")}
  Phone: REDACTED
  Last visit: {lapse.get("last_visit_date", "?")} ({lapse.get("days_since_last_visit", "?")} days ago)
  Lapse state: {lapse.get("state", "?")}
  Language pref: {cust_id.get("languages", ["en"])}
  Appointment history: {customer.get("appointment_history", [])}
  send_as MUST be: merchant_on_behalf
"""

    return f"""You are Vera, magicpin's AI merchant-engagement assistant.
Your job: compose ONE highly specific, compulsive WhatsApp message.

━━━ CATEGORY ━━━
Slug: {slug}
Voice: {voice_inst}
Taboo words (NEVER use): {taboos}
Allowed vocabulary: {allowed}
Peer stats: avg_rating={peer_stats.get("avg_rating", "?")}, avg_reviews={peer_stats.get("avg_reviews", "?")}, avg_ctr={peer_stats.get("avg_ctr", "?")}
Offer catalog: {[o.get("title") for o in offer_catalog]}

━━━ MERCHANT ━━━
Name: {m_name}
Owner: {owner}
Location: {locality}, {city}
Verified: {verified}
Languages: {languages}
Subscription: {sub.get("status", "?")} | Plan: {sub.get("plan", "?")} | {sub.get("days_remaining", "?")} days left

Performance (30d):
  Views: {views} | Calls: {calls} | CTR: {ctr}
  7d views delta: {v_delta} | 7d calls delta: {c_delta}
  Peer median CTR: {peer_stats.get("avg_ctr", "?")}

Active Offers: {[o.get("title") for o in active_offers] or "None"}
Expired/Paused Offers: {[o.get("title") for o in expired_offers[:2]] or "None"}
Account Signals: {signals}
Customer Aggregate: active={cust_agg.get("active_count", "?")}, lapsed={cust_agg.get("lapsed_count", "?")}, retention={cust_agg.get("retention_6mo", "?")}
Last merchant reply to Vera: {last_engaged}

━━━ TRIGGER ━━━
Kind: {trg_kind} (urgency={trg_urgency}/5, scope={trg_scope})
Payload: {json.dumps(trg_payload, ensure_ascii=False)}

━━━ CATEGORY KNOWLEDGE ━━━
Latest digest: {json.dumps(digest, ensure_ascii=False)}
Seasonal beats: {json.dumps(seasonal, ensure_ascii=False)}
Trend signals: {json.dumps(trends, ensure_ascii=False)}
Content library: {json.dumps(content_lib, ensure_ascii=False)}
{customer_block}
━━━ TRIGGER FRAMING GUIDANCE ━━━
{trigger_frame}

━━━ COMPOSITION RULES ━━━
1. Pick THE ONE signal that best combines trigger + merchant state. Don't dump every fact.
2. Use REAL numbers from context ONLY. Never invent data. If uncertain, omit the number.
3. ONE CTA at the very end. Binary (Reply YES / Reply 1 / 2) or open-ended question.
4. Length: 2–4 sentences. WhatsApp-native. Short paragraphs or line breaks OK.
5. NO preamble ("I hope you're doing well...", "Just checking in...").
6. NO re-introduction after first message.
7. For research triggers: cite the source (journal name + issue/page) inline.
8. {lang_inst}
9. Compulsion levers (use 1–2): specificity, loss aversion, social proof, curiosity hook, effort externalization ("I've drafted X — just say go").
10. CTA must be the LAST sentence.

━━━ WHAT THE JUDGE REWARDS ━━━
✓ Specificity: exact numbers, dates, source citations
✓ Category fit: voice matches business type exactly
✓ Merchant fit: references THEIR actual metrics and offers
✓ Decision quality: the right signal for this exact moment
✓ Engagement compulsion: strong reason to reply NOW, low-effort ask

━━━ WHAT THE JUDGE PENALIZES ━━━
✗ Generic offers ("Flat 30% off") — use service+price ("Dental Cleaning @ ₹299")
✗ Multiple CTAs ("Reply YES for X, NO for Y")
✗ Buried CTA (must be last sentence)
✗ Hallucinated data not in context
✗ Long preambles
✗ Taboo words

OUTPUT: Just the message text. No JSON. No explanation. No quotes. Just the message."""


def compose_proactive_message(category: Dict, merchant: Dict, trigger: Dict,
                               customer: Optional[Dict] = None,
                               previous_bodies: Optional[List[str]] = None) -> str:
    """Compose a proactive message for a tick action."""
    system = build_compose_system(category, merchant, trigger, customer)

    trg_kind = trigger.get("kind", "")
    trg_payload = trigger.get("payload", {})
    identity = merchant.get("identity", {})
    m_name = identity.get("name", "merchant")

    prev_context = ""
    if previous_bodies:
        prev_context = (
            "\n\nPREVIOUS MESSAGES YOU ALREADY SENT TO THIS MERCHANT "
            "(do NOT repeat these — vary the content):\n"
            + "\n".join(f"  • {b[:120]}" for b in previous_bodies[-3:])
        )

    user_prompt = (
        f"Compose the next Vera message for {m_name}.\n\n"
        f"Trigger kind: {trg_kind}\n"
        f"Key trigger facts: {json.dumps(trg_payload, ensure_ascii=False)}\n"
        f"{prev_context}\n\n"
        "Write ONE WhatsApp message. Just the text, nothing else."
    )

    resp = claude.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=500,
        temperature=0.3,
        system=system,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return resp.content[0].text.strip()


def compose_reply(category: Optional[Dict], merchant: Optional[Dict],
                  conv_history: List[Dict], incoming: str) -> str:
    """Compose a reply to a merchant's incoming message."""
    identity = (merchant or {}).get("identity", {})
    m_name = identity.get("name", "the merchant")
    languages = identity.get("languages", ["en"])
    lang_inst = get_lang_instruction(languages)

    slug = (category or {}).get("slug", "general")
    voice = (category or {}).get("voice", {})
    tone = voice.get("tone", "professional")
    taboos = voice.get("taboos", voice.get("vocab_taboo", []))
    active_offers = [o for o in (merchant or {}).get("offers", []) if o.get("status") == "active"]
    signals = (merchant or {}).get("signals", [])
    sub = (merchant or {}).get("subscription", {})
    cust_agg = (merchant or {}).get("customer_aggregate", {})

    history_text = ""
    if conv_history:
        history_text = "\nCONVERSATION SO FAR:\n"
        for t in conv_history[-6:]:
            role = "Merchant" if t.get("from") == "merchant" else "Vera"
            history_text += f"  {role}: {t.get('msg', '')}\n"

    system = f"""You are Vera, magicpin's merchant assistant. Reply to this merchant WhatsApp message.

Merchant: {m_name} ({slug})
Subscription: {sub.get("status", "?")} | Plan: {sub.get("plan", "?")}
Active offers: {[o.get("title") for o in active_offers] or "None"}
Signals: {signals}
Customer aggregate: active={cust_agg.get("active_count", "?")}, lapsed={cust_agg.get("lapsed_count", "?")}
{lang_inst}
Taboo words: {taboos}
{history_text}

REPLY RULES:
1. If merchant accepted / said yes → ACTION MODE: say what you're doing RIGHT NOW. No re-qualifying. Use words like "Sending now", "I've drafted", "Scheduling", "Done".
2. If merchant asked a specific question → answer it directly with numbers if available.
3. If merchant wants more info → give the one most useful piece, then a single follow-up question.
4. Length: 1–3 sentences. WhatsApp-native.
5. CTA must be the last sentence.
6. Never re-introduce yourself.
7. Never use taboo words: {taboos}

OUTPUT: Just the reply text. No explanation."""

    resp = claude.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=300,
        temperature=0.2,
        system=system,
        messages=[{"role": "user", "content": f'Merchant says: "{incoming}"\n\nWrite your reply:'}],
    )
    return resp.content[0].text.strip()


def compose_intent_commit_reply(merchant: Optional[Dict], category: Optional[Dict],
                                 conv_history: List[Dict], incoming: str) -> str:
    """Compose an action-mode reply when merchant commits."""
    identity = (merchant or {}).get("identity", {})
    m_name = identity.get("name", "the merchant")
    active_offers = [o for o in (merchant or {}).get("offers", []) if o.get("status") == "active"]
    signals = (merchant or {}).get("signals", [])

    last_vera = next(
        (t["msg"] for t in reversed(conv_history) if t.get("from") not in ("merchant",)),
        ""
    )

    system = f"""You are Vera, magicpin's merchant assistant. The merchant JUST COMMITTED to proceed.

SWITCH TO ACTION MODE IMMEDIATELY. No re-qualifying. No "Great!" filler before the action.

Merchant: {m_name}
Active offers: {[o.get("title") for o in active_offers] or "None"}
Account signals: {signals}
What Vera last said: {last_vera[:300]}

ACTION MODE RULES:
1. First word(s) should be the action: "Sending now —", "Done —", "I've drafted —", "Scheduling —", "Posting —"
2. Name the specific deliverable (e.g. "the fluoride recall post", "your ₹299 cleaning offer", "the patient WhatsApp").
3. Confirm the next concrete step or ask for one tiny piece of info to complete it.
4. Max 2 sentences.

OUTPUT: Just the reply. No explanation."""

    resp = claude.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=200,
        temperature=0.1,
        system=system,
        messages=[{"role": "user", "content": f'Merchant said: "{incoming}"\n\nAction-mode reply:'}],
    )
    return resp.content[0].text.strip()

# ─── TICK LOGIC ───────────────────────────────────────────────────────────────

def resolve_trigger_merchants(trg: Dict) -> List[str]:
    """Given a trigger payload, return list of merchant_ids it targets."""
    # Explicit merchant_id on the trigger
    mid = trg.get("merchant_id")
    if mid:
        return [mid]
    # merchant_id inside payload
    mid = trg.get("payload", {}).get("merchant_id")
    if mid:
        return [mid]
    # Category-wide trigger — target ALL merchants of that category
    slug = trg.get("payload", {}).get("category") or trg.get("category_slug", "")
    if slug:
        mids = [
            payload.get("merchant_id")
            for (scope, cid), entry in contexts.items()
            if scope == "merchant"
            for payload in [entry["payload"]]
            if payload.get("category_slug", "") == slug and payload.get("merchant_id")
        ]
        return mids
    # Fallback: all merchants
    return [
        entry["payload"].get("merchant_id")
        for (scope, cid), entry in contexts.items()
        if scope == "merchant" and entry["payload"].get("merchant_id")
    ]

def build_tick_actions(available_trigger_ids: List[str]) -> List[Dict]:
    """Core tick logic — decide which messages to send."""
    actions = []
    # Track which merchants already got an action this tick (one per merchant)
    merchants_actioned: set = set()

    # Sort triggers by urgency descending
    trigger_items = []
    for tid in available_trigger_ids:
        trg = get_ctx("trigger", tid)
        if trg:
            trigger_items.append((trg.get("urgency", 3), tid, trg))
    trigger_items.sort(key=lambda x: x[0], reverse=True)

    for urgency, tid, trg in trigger_items:
        suppression_key = trg.get("suppression_key", tid)
        if suppression_key in fired_suppression_keys:
            continue

        merchant_ids = resolve_trigger_merchants(trg)

        for mid in merchant_ids:
            if mid in merchants_actioned:
                continue

            merchant = get_ctx("merchant", mid)
            if not merchant:
                continue

            category = find_merchant_category(merchant)
            if not category:
                continue

            # Customer context if scope=customer
            customer = None
            cust_id = trg.get("payload", {}).get("customer_id")
            if cust_id:
                customer = get_ctx("customer", cust_id)

            # Anti-repetition
            prev_sent = merchant_sent_bodies.get(mid, [])

            try:
                body = compose_proactive_message(category, merchant, trg, customer, prev_sent)
            except Exception as e:
                continue

            if not body:
                continue

            # Check for verbatim repetition
            if body in prev_sent:
                continue

            identity = merchant.get("identity", {})
            trg_kind = trg.get("kind", "")
            trg_scope = trg.get("scope", "merchant")

            # CTA type
            if trg_kind in ("recall_due", "appointment_tomorrow", "customer_lapsed_soft"):
                cta = "binary_yes_no"
            else:
                cta = "open_ended"

            # send_as
            send_as = "merchant_on_behalf" if trg_scope == "customer" else "vera"

            rationale = (
                f"Trigger '{trg_kind}' (urgency={urgency}) fired for {identity.get('name', mid)}. "
                f"Signals: {merchant.get('signals', [])[:2]}. "
                f"Category: {category.get('slug', '?')}. "
                f"Chose this trigger as highest urgency unacted item for this merchant."
            )

            conv_id = f"conv_{mid}_{tid}_{uuid.uuid4().hex[:6]}"

            actions.append({
                "conversation_id": conv_id,
                "merchant_id": mid,
                "customer_id": cust_id,
                "send_as": send_as,
                "trigger_id": tid,
                "template_name": f"vera_{trg_kind}_v2",
                "template_params": [identity.get("name", ""), trg_kind, body[:50]],
                "body": body,
                "cta": cta,
                "suppression_key": suppression_key,
                "rationale": rationale,
            })

            # Track
            merchants_actioned.add(mid)
            merchant_sent_bodies.setdefault(mid, []).append(body)
            fired_suppression_keys.add(suppression_key)

    return actions

# ─── ENDPOINTS ────────────────────────────────────────────────────────────────

@app.get("/v1/healthz")
async def healthz():
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": count_contexts(),
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Vera Pro",
        "team_members": ["Challenger"],
        "model": CLAUDE_MODEL,
        "approach": (
            "4-context Claude composer with trigger-kind routing, "
            "auto-reply detection, intent-transition handler, hostile graceful exit, "
            "anti-repetition tracking, suppression key dedup"
        ),
        "contact_email": os.environ.get("CONTACT_EMAIL", "contact@example.com"),
        "version": "2.0.0",
        "submitted_at": datetime.utcnow().isoformat() + "Z",
    }


@app.post("/v1/context")
async def push_context(body: ContextBody):
    valid_scopes = {"category", "merchant", "customer", "trigger"}
    if body.scope not in valid_scopes:
        return JSONResponse(
            status_code=400,
            content={"accepted": False, "reason": "invalid_scope",
                     "details": f"scope must be one of {sorted(valid_scopes)}"}
        )

    key = (body.scope, body.context_id)
    cur = contexts.get(key)
    if cur and cur["version"] >= body.version:
        return JSONResponse(
            status_code=409,
            content={"accepted": False, "reason": "stale_version",
                     "current_version": cur["version"]}
        )

    contexts[key] = {"version": body.version, "payload": body.payload}

    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}_{uuid.uuid4().hex[:6]}",
        "stored_at": datetime.utcnow().isoformat() + "Z",
    }


@app.post("/v1/tick")
async def tick(body: TickBody):
    if not body.available_triggers:
        return {"actions": []}
    try:
        actions = build_tick_actions(body.available_triggers)
    except Exception:
        actions = []
    return {"actions": actions}


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    message = body.message.strip()
    conv_id = body.conversation_id
    merchant_id = body.merchant_id or ""
    customer_id = body.customer_id

    # Store this turn
    conversations.setdefault(conv_id, []).append({
        "from": body.from_role,
        "msg": message,
        "received_at": body.received_at,
        "turn": body.turn_number,
    })
    conv = conversations[conv_id]

    # ── AUTO-REPLY DETECTION ────────────────────────────────────────────────
    if is_auto_reply(message):
        auto_count = count_auto_replies_in_conv(conv, body.from_role)
        if auto_count >= 3:
            return {
                "action": "end",
                "rationale": (
                    f"Detected {auto_count} auto-replies in conv {conv_id}. "
                    "This is an automated WhatsApp Business responder, not a real merchant. "
                    "Exiting to avoid spam loop."
                ),
            }
        return {
            "action": "wait",
            "wait_seconds": 3600,
            "rationale": (
                f"Possible auto-reply detected ({auto_count}/3 threshold). "
                "Backing off 1 hour before retrying."
            ),
        }

    # ── HOSTILE DETECTION ───────────────────────────────────────────────────
    if is_hostile(message):
        return {
            "action": "end",
            "body": (
                "Understood — we'll stop messaging you. "
                "You can reach magicpin support anytime if you need help in the future. 🙏"
            ),
            "rationale": "Merchant expressed hostility or opt-out. Gracefully ending conversation.",
        }

    # Load contexts for reply composition
    merchant = get_ctx("merchant", merchant_id) if merchant_id else None
    category = find_merchant_category(merchant) if merchant else None

    conv_without_current = conv[:-1]

    # ── INTENT COMMIT DETECTION ─────────────────────────────────────────────
    if is_intent_commit(message):
        try:
            reply_body = compose_intent_commit_reply(merchant, category, conv_without_current, message)
        except Exception:
            reply_body = (
                "Sending now — I'll have the draft ready in under 2 minutes. "
                "I'll ping you once it's posted."
            )
        return {
            "action": "send",
            "body": reply_body,
            "cta": "open_ended",
            "rationale": (
                f"Merchant committed with '{message[:60]}'. "
                "Switched to action mode immediately — no re-qualifying."
            ),
        }

    # ── NORMAL REPLY ────────────────────────────────────────────────────────
    try:
        reply_body = compose_reply(category, merchant, conv_without_current, message)
        return {
            "action": "send",
            "body": reply_body,
            "cta": "open_ended",
            "rationale": (
                f"Merchant engaged with genuine message. "
                f"Composed contextual reply for {(merchant or {}).get('identity', {}).get('name', conv_id)}."
            ),
        }
    except Exception:
        return {
            "action": "send",
            "body": "Got it — let me look into that and get back to you shortly.",
            "cta": "open_ended",
            "rationale": "Fallback reply — LLM composition error.",
        }


# Optional teardown endpoint
@app.post("/v1/teardown")
async def teardown():
    contexts.clear()
    conversations.clear()
    fired_suppression_keys.clear()
    merchant_sent_bodies.clear()
    return {"status": "wiped"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("bot:app", host="0.0.0.0", port=port, reload=False)
