#!/usr/bin/env python3
"""
magicpin AI Challenge — Vera Bot (Gemini Edition)
==================================================
Uses Google Gemini API (free tier) for message composition.
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
import urllib.request
import urllib.error

# ─── APP & STATE ──────────────────────────────────────────────────────────────

app = FastAPI(title="Vera Challenge Bot", version="2.0.0")
START_TIME = time.time()

contexts: Dict[Tuple[str, str], Dict] = {}
conversations: Dict[str, List[Dict]] = {}
fired_suppression_keys: set = set()
merchant_sent_bodies: Dict[str, List[str]] = {}

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-1.5-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# ─── GEMINI CALLER ────────────────────────────────────────────────────────────

def call_gemini(prompt: str, max_tokens: int = 500) -> str:
    """Call Gemini API and return text response."""
    url = f"{GEMINI_URL}?key={GEMINI_API_KEY}"
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": max_tokens,
        }
    }).encode("utf-8")

    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        resp = urllib.request.urlopen(req, timeout=25)
        data = json.loads(resp.read().decode("utf-8"))
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        raise RuntimeError(f"Gemini error: {e}")

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
    slug = merchant.get("category_slug", "")
    cat = get_ctx("category", slug)
    if cat:
        return cat
    for (scope, cid), entry in contexts.items():
        if scope == "category":
            return entry["payload"]
    return None

# ─── DETECTION LOGIC ──────────────────────────────────────────────────────────

AUTO_REPLY_PHRASES = [
    "thank you for contacting", "thank you for reaching out",
    "we will get back to you", "we will respond shortly",
    "our team will respond", "this is an automated",
    "auto-reply", "automatic reply", "autoresponder",
    "out of office", "we received your message",
    "please do not reply to this message",
]

def is_auto_reply(msg: str) -> bool:
    low = msg.lower()
    return any(p in low for p in AUTO_REPLY_PHRASES)

INTENT_COMMIT_PHRASES = [
    "let's do it", "lets do it", "ok let's", "ok lets",
    "okay lets", "go ahead", "go for it", "proceed", "confirm",
    "yes please", "haan bilkul", "bilkul", "zaroor", "kar do",
    "start karo", "shuru karo", "what's next", "whats next",
    "done deal", "agreed", "sounds good", "sure do it",
]

def is_intent_commit(msg: str) -> bool:
    low = msg.lower().strip()
    strong_singles = {"yes", "haan", "ha", "done", "ok", "okay", "sure", "yep", "yup"}
    if low in strong_singles:
        return True
    return any(p in low for p in INTENT_COMMIT_PHRASES)

HOSTILE_PHRASES = [
    "stop messaging", "stop texting", "stop contacting",
    "don't message", "dont message", "this is spam",
    "useless spam", "not interested", "unsubscribe",
    "remove me", "mat karo", "band karo", "nahi chahiye",
    "annoying", "irritating", "harassment", "bakwaas",
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
        "Clinical vocabulary welcome: fluoride varnish, caries, recall, prophylaxis. "
        "Never say 'cure' or 'guaranteed'. Cite sources (journal, page). Peer tone, no hype."
    ),
    "warm_friendly": "Warm, friendly, practical. Like a knowledgeable friend talking salon business. Use ₹ prices.",
    "operator_to_operator": "Direct, numbers-first. Like a business consultant to a restaurant owner. No fluff.",
    "coaching": "Motivational, results-focused. Like a gym coach to a gym owner. Numbers, wins, actions.",
    "trustworthy_precise": "Trustworthy, precise. Exact numbers. Never make unverifiable claims.",
}

def get_voice_instruction(tone: str) -> str:
    return VOICE_PROFILES.get(tone, "Professional, direct, WhatsApp-friendly. Short sentences.")

def get_lang_instruction(languages: List[str]) -> str:
    langs = [l.lower() for l in languages]
    if "hi" in langs or "hi-en" in langs or "hinglish" in langs:
        return "LANGUAGE: Hindi-English mix (Hinglish). Natural mix like a professional WhatsApp message."
    return "LANGUAGE: English. Clear, WhatsApp-friendly."

# ─── TRIGGER ROUTING ──────────────────────────────────────────────────────────

TRIGGER_FRAMES = {
    "research_digest": "Frame as 'new research just dropped relevant to your patients.' Lead with specific finding and magnitude. Offer to pull abstract + draft patient content.",
    "recall_due": "Frame as 'a specific patient's recall window is open.' Name patient, time since last visit, exact slots. Offer to send reminder on their behalf.",
    "perf_spike": "Lead with exact spike numbers vs baseline. Curiosity hook: 'want to see what drove it?'",
    "perf_dip": "Lead with exact dip (%, absolute). Show loss aversion. Offer one concrete fix. Make it low-effort.",
    "milestone_reached": "Celebrate with exact number. Social proof: 'You're now in top X% of [category] in [locality].'",
    "dormant_with_vera": "Acknowledge gap, don't guilt-trip. Lead with interesting stat about their account.",
    "festival_upcoming": "Name festival and exact days remaining. Give specific service+price offer. Social proof.",
    "review_theme_emerged": "Name the specific theme from reviews. Show exact count. Offer to help draft response.",
    "competitor_opened": "Strategic not alarmist. Exact distance, GBP rating. Offer one proactive move.",
    "category_trend_movement": "Lead with search trend number. Connect to local context. Offer to help capture demand.",
    "scheduled_recurring": "Fresh curiosity question or genuine peer insight. Not a reminder.",
    "customer_lapsed_soft": "Remind merchant specific customer recall window opened. Time since last visit. Offer to draft recall WhatsApp.",
    "regulation_change": "Name the specific regulation and effective date. Direct impact on practice. Offer to help prepare.",
    "weather_heatwave": "Name temperature and relevant service opportunity. Short and seasonal.",
}

def get_trigger_frame(kind: str) -> str:
    return TRIGGER_FRAMES.get(kind, "Lead with most specific verifiable fact from trigger. Single CTA at end.")

# ─── LLM COMPOSER ─────────────────────────────────────────────────────────────

def build_prompt(category: Dict, merchant: Dict, trigger: Dict,
                 customer: Optional[Dict] = None,
                 previous_bodies: Optional[List[str]] = None,
                 mode: str = "proactive") -> str:

    slug = category.get("slug", "general")
    voice = category.get("voice", {})
    tone = voice.get("tone", "professional")
    taboos = voice.get("taboos", voice.get("vocab_taboo", []))
    peer_stats = category.get("peer_stats", {})
    offer_catalog = category.get("offer_catalog", [])
    digest = category.get("digest", [])[:3]
    seasonal = category.get("seasonal_beats", [])
    trends = category.get("trend_signals", [])[:2]

    identity = merchant.get("identity", {})
    m_name = identity.get("name", "the merchant")
    owner = identity.get("owner_first_name", "")
    locality = identity.get("locality", "")
    city = identity.get("city", "")
    languages = identity.get("languages", ["en"])

    perf = merchant.get("performance", {})
    views = perf.get("views", "?")
    calls = perf.get("calls", "?")
    ctr = perf.get("ctr", "?")

    sub = merchant.get("subscription", {})
    offers = merchant.get("offers", [])
    active_offers = [o for o in offers if o.get("status") == "active"]
    signals = merchant.get("signals", [])
    cust_agg = merchant.get("customer_aggregate", {})

    trg_kind = trigger.get("kind", "")
    trg_payload = trigger.get("payload", {})
    trg_urgency = trigger.get("urgency", 3)

    voice_inst = get_voice_instruction(tone)
    lang_inst = get_lang_instruction(languages)
    trigger_frame = get_trigger_frame(trg_kind)

    customer_block = ""
    if customer:
        cust_id = customer.get("identity", {})
        lapse = customer.get("lapse_state", {})
        customer_block = f"""
CUSTOMER CONTEXT:
  Name: {cust_id.get("name", "?")}
  Last visit: {lapse.get("last_visit_date", "?")} ({lapse.get("days_since_last_visit", "?")} days ago)
  Lapse state: {lapse.get("state", "?")}
  send_as MUST be: merchant_on_behalf
"""

    prev_context = ""
    if previous_bodies:
        prev_context = "\nPREVIOUS MESSAGES ALREADY SENT (do NOT repeat):\n" + \
                       "\n".join(f"  • {b[:120]}" for b in previous_bodies[-3:])

    return f"""You are Vera, magicpin's AI merchant-engagement assistant.
Compose ONE highly specific, compulsive WhatsApp message.

━━━ CATEGORY ━━━
Slug: {slug}
Voice: {voice_inst}
Taboo words (NEVER use): {taboos}
Peer stats: avg_rating={peer_stats.get("avg_rating","?")}, avg_ctr={peer_stats.get("avg_ctr","?")}
Offer catalog: {[o.get("title") for o in offer_catalog]}
Latest digest: {json.dumps(digest, ensure_ascii=False)}
Seasonal: {json.dumps(seasonal, ensure_ascii=False)}
Trends: {json.dumps(trends, ensure_ascii=False)}

━━━ MERCHANT ━━━
Name: {m_name} | Owner: {owner}
Location: {locality}, {city}
Languages: {languages}
Subscription: {sub.get("status","?")} | Plan: {sub.get("plan","?")} | {sub.get("days_remaining","?")} days left
Performance: views={views}, calls={calls}, ctr={ctr}, peer_median_ctr={peer_stats.get("avg_ctr","?")}
Active Offers: {[o.get("title") for o in active_offers] or "None"}
Signals: {signals}
Customer Aggregate: active={cust_agg.get("active_count","?")}, lapsed={cust_agg.get("lapsed_count","?")}
{customer_block}
━━━ TRIGGER ━━━
Kind: {trg_kind} (urgency={trg_urgency}/5)
Payload: {json.dumps(trg_payload, ensure_ascii=False)}

━━━ TRIGGER FRAMING ━━━
{trigger_frame}
{prev_context}

━━━ RULES ━━━
1. Pick THE ONE best signal combining trigger + merchant state. Don't dump every fact.
2. Use REAL numbers from context ONLY. Never invent data.
3. ONE CTA at the very end. Binary (Reply YES/1/2) or open-ended question.
4. Length: 2-4 sentences. WhatsApp-native.
5. NO preamble ("I hope you're doing well...").
6. NO re-introduction after first message.
7. For research triggers: cite source inline (journal + page).
8. {lang_inst}
9. Compulsion levers (use 1-2): specificity, loss aversion, social proof, curiosity, effort externalization.
10. CTA must be the LAST sentence.

PENALTIES (avoid these):
- Generic offers ("Flat 30% off") → use service+price ("Dental Cleaning @ ₹299")
- Multiple CTAs
- Buried CTA
- Hallucinated data
- Taboo words: {taboos}

OUTPUT: Just the message text. Nothing else. No quotes. No explanation."""


def compose_proactive_message(category: Dict, merchant: Dict, trigger: Dict,
                               customer: Optional[Dict] = None,
                               previous_bodies: Optional[List[str]] = None) -> str:
    prompt = build_prompt(category, merchant, trigger, customer, previous_bodies)
    return call_gemini(prompt, max_tokens=500)


def compose_reply(category: Optional[Dict], merchant: Optional[Dict],
                  conv_history: List[Dict], incoming: str) -> str:
    identity = (merchant or {}).get("identity", {})
    m_name = identity.get("name", "the merchant")
    languages = identity.get("languages", ["en"])
    lang_inst = get_lang_instruction(languages)
    active_offers = [o for o in (merchant or {}).get("offers", []) if o.get("status") == "active"]
    signals = (merchant or {}).get("signals", [])
    sub = (merchant or {}).get("subscription", {})
    slug = (category or {}).get("slug", "general")
    taboos = (category or {}).get("voice", {}).get("taboos", [])
    cust_agg = (merchant or {}).get("customer_aggregate", {})

    history_text = ""
    if conv_history:
        history_text = "\nCONVERSATION SO FAR:\n"
        for t in conv_history[-6:]:
            role = "Merchant" if t.get("from") == "merchant" else "Vera"
            history_text += f"  {role}: {t.get('msg', '')}\n"

    prompt = f"""You are Vera, magicpin's merchant assistant. Reply to this merchant WhatsApp message.

Merchant: {m_name} ({slug})
Subscription: {sub.get("status","?")} | Plan: {sub.get("plan","?")}
Active offers: {[o.get("title") for o in active_offers] or "None"}
Signals: {signals}
Customer aggregate: active={cust_agg.get("active_count","?")}, lapsed={cust_agg.get("lapsed_count","?")}
{lang_inst}
Taboo words: {taboos}
{history_text}

Merchant says: "{incoming}"

REPLY RULES:
1. If merchant accepted/said yes → ACTION MODE: say what you're doing RIGHT NOW. Use "Sending now", "I've drafted", "Scheduling", "Done".
2. If merchant asked a question → answer directly with numbers if available.
3. Length: 1-3 sentences. WhatsApp-native.
4. CTA must be last sentence.
5. Never re-introduce yourself.
6. Never use taboo words: {taboos}

OUTPUT: Just the reply text. No explanation."""

    return call_gemini(prompt, max_tokens=300)


def compose_intent_commit_reply(merchant: Optional[Dict], category: Optional[Dict],
                                 conv_history: List[Dict], incoming: str) -> str:
    identity = (merchant or {}).get("identity", {})
    m_name = identity.get("name", "the merchant")
    active_offers = [o for o in (merchant or {}).get("offers", []) if o.get("status") == "active"]
    signals = (merchant or {}).get("signals", [])

    last_vera = next(
        (t["msg"] for t in reversed(conv_history) if t.get("from") not in ("merchant",)), ""
    )

    prompt = f"""You are Vera, magicpin's merchant assistant. The merchant JUST COMMITTED to proceed.

SWITCH TO ACTION MODE IMMEDIATELY. No re-qualifying.

Merchant: {m_name}
Active offers: {[o.get("title") for o in active_offers] or "None"}
Signals: {signals}
What Vera last said: {last_vera[:300]}
Merchant committed: "{incoming}"

ACTION MODE RULES:
1. First words should be the action: "Sending now —", "Done —", "I've drafted —", "Scheduling —"
2. Name the specific deliverable.
3. Confirm next concrete step or ask one tiny piece of info.
4. Max 2 sentences.

OUTPUT: Just the reply. No explanation."""

    return call_gemini(prompt, max_tokens=200)

# ─── TICK LOGIC ───────────────────────────────────────────────────────────────

def resolve_trigger_merchants(trg: Dict) -> List[str]:
    mid = trg.get("merchant_id")
    if mid:
        return [mid]
    mid = trg.get("payload", {}).get("merchant_id")
    if mid:
        return [mid]
    slug = trg.get("payload", {}).get("category") or trg.get("category_slug", "")
    if slug:
        return [
            entry["payload"].get("merchant_id")
            for (scope, cid), entry in contexts.items()
            if scope == "merchant"
            and entry["payload"].get("category_slug", "") == slug
            and entry["payload"].get("merchant_id")
        ]
    return [
        entry["payload"].get("merchant_id")
        for (scope, cid), entry in contexts.items()
        if scope == "merchant" and entry["payload"].get("merchant_id")
    ]


def build_tick_actions(available_trigger_ids: List[str]) -> List[Dict]:
    actions = []
    merchants_actioned: set = set()

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
            if not mid or mid in merchants_actioned:
                continue

            merchant = get_ctx("merchant", mid)
            if not merchant:
                continue

            category = find_merchant_category(merchant)
            if not category:
                continue

            customer = None
            cust_id = trg.get("payload", {}).get("customer_id")
            if cust_id:
                customer = get_ctx("customer", cust_id)

            prev_sent = merchant_sent_bodies.get(mid, [])

            try:
                body = compose_proactive_message(category, merchant, trg, customer, prev_sent)
            except Exception:
                continue

            if not body or body in prev_sent:
                continue

            identity = merchant.get("identity", {})
            trg_kind = trg.get("kind", "")
            trg_scope = trg.get("scope", "merchant")

            cta = "binary_yes_no" if trg_kind in ("recall_due", "appointment_tomorrow", "customer_lapsed_soft") else "open_ended"
            send_as = "merchant_on_behalf" if trg_scope == "customer" else "vera"

            rationale = (
                f"Trigger '{trg_kind}' (urgency={urgency}) for {identity.get('name', mid)}. "
                f"Signals: {merchant.get('signals', [])[:2]}. "
                f"Category: {category.get('slug', '?')}."
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
        "model": f"google/{GEMINI_MODEL}",
        "approach": (
            "4-context Gemini composer with trigger-kind routing, "
            "auto-reply detection, intent-transition handler, "
            "hostile graceful exit, anti-repetition, suppression key dedup"
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

    conversations.setdefault(conv_id, []).append({
        "from": body.from_role,
        "msg": message,
        "received_at": body.received_at,
        "turn": body.turn_number,
    })
    conv = conversations[conv_id]

    # Auto-reply detection
    if is_auto_reply(message):
        auto_count = count_auto_replies_in_conv(conv, body.from_role)
        if auto_count >= 3:
            return {
                "action": "end",
                "rationale": f"Detected {auto_count} auto-replies. Exiting to avoid spam loop.",
            }
        return {
            "action": "wait",
            "wait_seconds": 3600,
            "rationale": f"Possible auto-reply ({auto_count}/3). Backing off 1 hour.",
        }

    # Hostile detection
    if is_hostile(message):
        return {
            "action": "end",
            "body": "Understood — we'll stop messaging you. You can reach magicpin support anytime if needed. 🙏",
            "rationale": "Merchant expressed hostility. Gracefully ending.",
        }

    merchant = get_ctx("merchant", merchant_id) if merchant_id else None
    category = find_merchant_category(merchant) if merchant else None
    conv_without_current = conv[:-1]

    # Intent commit detection
    if is_intent_commit(message):
        try:
            reply_body = compose_intent_commit_reply(merchant, category, conv_without_current, message)
        except Exception:
            reply_body = "Sending now — I'll have the draft ready in under 2 minutes. I'll ping you once it's posted."
        return {
            "action": "send",
            "body": reply_body,
            "cta": "open_ended",
            "rationale": f"Merchant committed with '{message[:60]}'. Switched to action mode immediately.",
        }

    # Normal reply
    try:
        reply_body = compose_reply(category, merchant, conv_without_current, message)
        return {
            "action": "send",
            "body": reply_body,
            "cta": "open_ended",
            "rationale": f"Merchant engaged genuinely. Composed contextual reply.",
        }
    except Exception:
        return {
            "action": "send",
            "body": "Got it — let me look into that and get back to you shortly.",
            "cta": "open_ended",
            "rationale": "Fallback reply.",
        }


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
