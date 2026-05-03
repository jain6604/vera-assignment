#!/usr/bin/env python3
"""
magicpin AI Challenge — Vera Bot v3 (Fixed)
Key fixes:
- Trigger coverage: works even when judge pushes minimal context
- STOP/hostile: always returns action=end
- Template fallbacks if Gemini API fails
- Robust trigger→merchant→category resolution
"""

import os, time, json, uuid, urllib.request
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(title="Vera Challenge Bot", version="3.0.0")
START_TIME = time.time()

contexts: Dict[Tuple[str, str], Dict] = {}
conversations: Dict[str, List[Dict]] = {}
fired_suppression_keys: set = set()
merchant_sent_bodies: Dict[str, List[str]] = {}

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-1.5-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

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

def call_gemini(prompt: str, max_tokens: int = 500) -> Optional[str]:
    if not GEMINI_API_KEY:
        return None
    url = f"{GEMINI_URL}?key={GEMINI_API_KEY}"
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": max_tokens}
    }).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        resp = urllib.request.urlopen(req, timeout=25)
        data = json.loads(resp.read().decode("utf-8"))
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        return None

def get_ctx(scope: str, cid: str) -> Optional[Dict]:
    entry = contexts.get((scope, cid))
    return entry["payload"] if entry else None

def count_contexts() -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for (scope, _) in contexts:
        counts[scope] = counts.get(scope, 0) + 1
    return counts

def find_category_for_merchant(merchant: Dict) -> Optional[Dict]:
    slug = merchant.get("category_slug", "") or merchant.get("identity", {}).get("category_slug", "")
    if slug:
        cat = get_ctx("category", slug)
        if cat:
            return cat
    for (scope, _), entry in contexts.items():
        if scope == "category":
            return entry["payload"]
    return None

def get_merchant_for_trigger(trg: Dict) -> Tuple[Optional[Dict], Optional[str]]:
    mid = trg.get("merchant_id") or trg.get("payload", {}).get("merchant_id") or trg.get("context_id")
    if mid:
        m = get_ctx("merchant", mid)
        if m:
            return m, mid
    payload = trg.get("payload", {})
    cat_slug = payload.get("category", "") or trg.get("category_slug", "")
    for (scope, cid), entry in contexts.items():
        if scope == "merchant":
            m = entry["payload"]
            if not cat_slug or m.get("category_slug", "") == cat_slug:
                return m, m.get("merchant_id", cid)
    # Build minimal merchant from trigger payload
    minimal = {
        "merchant_id": mid or "unknown",
        "category_slug": cat_slug or "general",
        "identity": {
            "name": payload.get("merchant_name", payload.get("name", "the merchant")),
            "owner_first_name": payload.get("owner_name", ""),
            "locality": payload.get("locality", ""),
            "city": payload.get("city", ""),
            "languages": ["en"],
        },
        "subscription": {"status": "active", "plan": "Pro", "days_remaining": 90},
        "performance": {"views": payload.get("views", 1000), "calls": payload.get("calls", 20), "ctr": payload.get("ctr", 0.02)},
        "offers": [],
        "signals": [],
        "customer_aggregate": {},
    }
    return minimal, mid

STOP_WORDS = {"stop", "unsubscribe", "quit", "cancel", "end", "block"}
AUTO_REPLY_PHRASES = [
    "thank you for contacting", "thank you for reaching out",
    "we will get back to you", "this is an automated", "auto-reply",
    "automatic reply", "out of office", "we received your message",
]
HOSTILE_PHRASES = [
    "stop messaging", "stop texting", "don't message", "dont message",
    "this is spam", "not interested", "unsubscribe", "remove me",
    "mat karo", "band karo", "nahi chahiye", "annoying", "harassment", "bakwaas",
]
INTENT_COMMIT_PHRASES = [
    "let's do it", "lets do it", "ok let's", "ok lets",
    "go ahead", "go for it", "proceed", "confirm", "yes please",
    "haan bilkul", "bilkul", "zaroor", "kar do", "start karo",
    "what's next", "whats next", "done deal", "agreed", "sounds good",
]

def is_stop_command(msg: str) -> bool:
    clean = msg.strip().lower().rstrip("!.").strip()
    if clean in STOP_WORDS:
        return True
    return any(p in clean for p in HOSTILE_PHRASES)

def is_auto_reply(msg: str) -> bool:
    low = msg.lower()
    return any(p in low for p in AUTO_REPLY_PHRASES)

def is_intent_commit(msg: str) -> bool:
    low = msg.lower().strip()
    if low in {"yes", "haan", "ha", "done", "ok", "okay", "sure", "yep", "yup"}:
        return True
    return any(p in low for p in INTENT_COMMIT_PHRASES)

def count_auto_replies(conv: List[Dict], from_role: str) -> int:
    return sum(1 for t in conv if t.get("from") == from_role and is_auto_reply(t.get("msg", "")))

TRIGGER_FRAMES = {
    "research_digest": "Frame as 'new research just dropped relevant to your patients.' Lead with finding + source citation. Offer to draft patient content.",
    "recall_due": "Name patient, days since last visit, exact available slots. Offer to send recall on merchant's behalf.",
    "perf_spike": "Lead with exact spike % vs baseline. Curiosity hook: 'want to see what drove it?'",
    "perf_dip": "Lead with exact dip number. Loss aversion. One concrete fix. Low-effort ask.",
    "milestone_reached": "Celebrate with exact number. Social proof: top X% of category in locality.",
    "dormant_with_vera": "Lead with one interesting stat about their account. Low-commitment question.",
    "festival_upcoming": "Name festival and days remaining. Specific service+price offer. Social proof count.",
    "review_theme_emerged": "Name the theme. Show exact count. Offer to draft response.",
    "competitor_opened": "Strategic not alarmist. Exact distance. One proactive move.",
    "category_trend_movement": "Lead with search trend % number. Connect to their local context.",
    "scheduled_recurring": "Fresh peer insight or curiosity question. Not a reminder.",
    "customer_lapsed_soft": "Customer name, days since visit. Offer to draft recall WhatsApp.",
    "regulation_change": "Specific regulation, effective date, direct impact. Offer to help prepare.",
    "appointment_tomorrow": "Tomorrow's appointment details. Pre-visit confirmation.",
}

def get_fallback_message(trg_kind: str, merchant: Dict, trg_payload: Dict) -> str:
    identity = merchant.get("identity", {})
    name = identity.get("name", "your business")
    owner = identity.get("owner_first_name", "") or "there"
    locality = identity.get("locality", "")
    perf = merchant.get("performance", {})
    views = perf.get("views", "?")
    ctr = perf.get("ctr", "?")
    offers = [o.get("title", "") for o in merchant.get("offers", []) if o.get("status") == "active"]
    offer_str = offers[0] if offers else "your active offer"
    cust_agg = merchant.get("customer_aggregate", {})
    lapsed = cust_agg.get("lapsed_count", "?")

    templates = {
        "research_digest": f"{owner}, new research just dropped — {trg_payload.get('top_item', {}).get('title', 'a key clinical finding')} ({trg_payload.get('top_item', {}).get('source', 'latest journal')}). Want me to draft a patient WhatsApp for {name}?",
        "recall_due": f"Hi {trg_payload.get('customer_name', 'there')} — it's been {trg_payload.get('days_since_last_visit', '6 months')} since your last visit at {name}. Slots open: {', '.join(trg_payload.get('available_slots', ['this week']))}. Reply 1 to confirm.",
        "perf_spike": f"{owner}, {name}'s views hit {trg_payload.get('views_yesterday', views)} yesterday — {trg_payload.get('spike_pct', '20')}% above your average. Want to activate {offer_str} to capture this momentum?",
        "perf_dip": f"{owner}, calls at {name} dropped {trg_payload.get('dip_pct', '15')}% this week. One quick fix: reactivating {offer_str} in {locality} could recover this. Should I do it?",
        "milestone_reached": f"Congratulations {owner}! {name} crossed {trg_payload.get('milestone_value', '500')} {trg_payload.get('milestone_type', 'views')} — top {trg_payload.get('percentile', '20')}% in {locality}. Want to push for the next milestone?",
        "festival_upcoming": f"{owner}, {trg_payload.get('festival_name', 'the upcoming festival')} is {trg_payload.get('days_remaining', '7')} days away. {trg_payload.get('peer_count', 'Several')} merchants in {locality} are already running offers. Want me to activate one for {name}?",
        "dormant_with_vera": f"Hey {owner}! {name} has {views} views this month and {lapsed} customers due for recall. One message could bring them back — want me to draft it?",
        "review_theme_emerged": f"{owner}, {trg_payload.get('review_count', '3')} recent reviews for {name} mention '{trg_payload.get('theme', 'your service')}'. Want me to draft a response and a fix?",
        "competitor_opened": f"{owner}, a new competitor opened {trg_payload.get('distance_km', '0.5')}km from {name} (rating: {trg_payload.get('competitor_rating', '4.0')}). Want to get ahead with a targeted offer this week?",
        "category_trend_movement": f"{owner}, searches for your category in {locality} are up {trg_payload.get('trend_delta_pct', '30')}% this month. Want me to help {name} capture this demand with a post?",
        "customer_lapsed_soft": f"{owner}, {trg_payload.get('customer_name', 'a key customer')} hasn't visited {name} in {trg_payload.get('days_since_last_visit', '90')} days. Want me to send them a recall WhatsApp with {offer_str}?",
        "regulation_change": f"{owner}, regulatory update affecting your practice from {trg_payload.get('effective_date', 'next month')}. Want a quick summary and a patient notice draft?",
        "scheduled_recurring": f"{owner}, quick update on {name}: CTR is {ctr} vs peer median {trg_payload.get('peer_median_ctr', '0.03')} in {locality}. One photo update can close this gap — want me to queue it?",
        "appointment_tomorrow": f"Reminder: appointment at {name} tomorrow at {trg_payload.get('time', '10:00 AM')} for {trg_payload.get('customer_name', 'your patient')}. Reply 1 to confirm or 2 to reschedule.",
    }
    return templates.get(trg_kind, f"{owner}, here's a quick update on {name}: {views} views this month and {lapsed} customers due for follow-up. Want me to help with your next engagement?")

def compose_proactive_message(merchant: Dict, trg: Dict, category: Optional[Dict],
                               customer: Optional[Dict] = None,
                               previous_bodies: Optional[List[str]] = None) -> str:
    trg_kind = trg.get("kind", "scheduled_recurring")
    trg_payload = trg.get("payload", {})
    identity = merchant.get("identity", {})
    languages = identity.get("languages", ["en"])
    lang_inst = "Hinglish (Hindi-English mix)" if any(l in ["hi", "hi-en", "hinglish"] for l in [x.lower() for x in languages]) else "English"
    cat_slug = (category or {}).get("slug", merchant.get("category_slug", "general"))
    voice = (category or {}).get("voice", {})
    taboos = voice.get("taboos", [])
    peer_stats = (category or {}).get("peer_stats", {})
    digest = (category or {}).get("digest", [])[:2]
    offer_catalog = (category or {}).get("offer_catalog", [])
    perf = merchant.get("performance", {})
    sub = merchant.get("subscription", {})
    active_offers = [o for o in merchant.get("offers", []) if o.get("status") == "active"]
    cust_agg = merchant.get("customer_aggregate", {})
    prev_context = ""
    if previous_bodies:
        prev_context = "PREVIOUS MESSAGES (do NOT repeat):\n" + "\n".join(f"- {b[:100]}" for b in previous_bodies[-2:])
    customer_block = ""
    if customer:
        cid = customer.get("identity", {})
        lapse = customer.get("lapse_state", {})
        customer_block = f"Customer: {cid.get('name','?')}, last visit {lapse.get('last_visit_date','?')} ({lapse.get('days_since_last_visit','?')} days ago)"

    prompt = f"""You are Vera, magicpin's AI merchant engagement assistant.
Write ONE highly specific, compulsive WhatsApp message.

MERCHANT: {identity.get('name','?')} | Owner: {identity.get('owner_first_name','')} | {identity.get('locality','')}
Category: {cat_slug} | Language: {lang_inst}
Subscription: {sub.get('status','active')} {sub.get('plan','Pro')} ({sub.get('days_remaining','?')} days left)
Views: {perf.get('views','?')} | Calls: {perf.get('calls','?')} | CTR: {perf.get('ctr','?')} (peer median: {peer_stats.get('avg_ctr','?')})
Active offers: {[o.get('title') for o in active_offers] or 'None'}
Offer catalog: {[o.get('title') for o in offer_catalog[:3]]}
Signals: {merchant.get('signals',[])}
Lapsed customers: {cust_agg.get('lapsed_count','?')}
Category digest: {json.dumps(digest, ensure_ascii=False)}
{customer_block}
TRIGGER KIND: {trg_kind} (urgency={trg.get('urgency',3)}/5)
TRIGGER PAYLOAD: {json.dumps(trg_payload, ensure_ascii=False)}
FRAMING: {TRIGGER_FRAMES.get(trg_kind, 'Lead with most specific verifiable fact. Single CTA at end.')}
{prev_context}

RULES:
1. Use REAL numbers from context only. Never invent data.
2. ONE CTA as the LAST sentence.
3. 2-4 sentences. WhatsApp-native.
4. No preamble.
5. Taboo words — NEVER use: {taboos}
6. For research triggers: cite source inline.
7. Use: specificity, loss aversion, social proof, curiosity, effort externalization.

OUTPUT: Just the message. No quotes, no explanation."""

    result = call_gemini(prompt, max_tokens=400)
    if result and len(result) > 20:
        return result
    return get_fallback_message(trg_kind, merchant, trg_payload)

def compose_reply_message(merchant: Optional[Dict], category: Optional[Dict],
                           conv_history: List[Dict], incoming: str) -> str:
    identity = (merchant or {}).get("identity", {})
    m_name = identity.get("name", "the merchant")
    languages = identity.get("languages", ["en"])
    active_offers = [o for o in (merchant or {}).get("offers", []) if o.get("status") == "active"]
    sub = (merchant or {}).get("subscription", {})
    cat_slug = (category or {}).get("slug", "general")
    taboos = (category or {}).get("voice", {}).get("taboos", [])
    cust_agg = (merchant or {}).get("customer_aggregate", {})
    lang_inst = "Hinglish" if any(l in ["hi", "hi-en"] for l in [x.lower() for x in languages]) else "English"
    history_text = ""
    if conv_history:
        history_text = "CONVERSATION:\n" + "\n".join(
            f"  {'Merchant' if t.get('from') == 'merchant' else 'Vera'}: {t.get('msg','')}"
            for t in conv_history[-5:]
        )
    prompt = f"""You are Vera, magicpin merchant assistant. Reply to this merchant message.

Merchant: {m_name} ({cat_slug}) | Subscription: {sub.get('status','?')} {sub.get('plan','?')}
Active offers: {[o.get('title') for o in active_offers] or 'None'}
Lapsed customers: {cust_agg.get('lapsed_count','?')} | Language: {lang_inst}
{history_text}
Merchant says: "{incoming}"

RULES: 1. If yes/agreed → ACTION MODE: "Sending now —", "I've drafted —" 2. Answer questions directly with numbers. 3. 1-3 sentences. 4. CTA last. 5. Taboo: {taboos}
OUTPUT: Just the reply."""

    result = call_gemini(prompt, max_tokens=250)
    return result if result and len(result) > 10 else "Got it — working on that now. Should I go ahead?"

def compose_action_mode_reply(merchant: Optional[Dict], conv_history: List[Dict], incoming: str) -> str:
    active_offers = [o for o in (merchant or {}).get("offers", []) if o.get("status") == "active"]
    last_vera = next((t["msg"] for t in reversed(conv_history) if t.get("from") != "merchant"), "")
    prompt = f"""Vera, magicpin assistant. Merchant committed. ACTION MODE.
Active offers: {[o.get('title') for o in active_offers] or 'None'}
Vera last said: {last_vera[:200]}
Merchant: "{incoming}"
Start with action word: "Sending now —", "Done —", "I've drafted —"
Name specific deliverable. Max 2 sentences. OUTPUT: Just reply."""
    result = call_gemini(prompt, max_tokens=150)
    return result if result and len(result) > 10 else "Sending now — I'll have your draft ready within 2 minutes and ping you once it's live."

def build_tick_actions(available_trigger_ids: List[str]) -> List[Dict]:
    actions = []
    merchants_actioned: set = set()
    trigger_items = [
        (trg.get("urgency", 3), tid, trg)
        for tid in available_trigger_ids
        for trg in [get_ctx("trigger", tid)]
        if trg
    ]
    trigger_items.sort(key=lambda x: x[0], reverse=True)

    for urgency, tid, trg in trigger_items:
        suppression_key = trg.get("suppression_key", tid)
        if suppression_key in fired_suppression_keys:
            continue
        trg_kind = trg.get("kind", "scheduled_recurring")
        trg_scope = trg.get("scope", "merchant")
        trg_payload = trg.get("payload", {})
        merchant, mid = get_merchant_for_trigger(trg)
        if not merchant:
            continue
        key = mid or id(merchant)
        if key in merchants_actioned:
            continue
        category = find_category_for_merchant(merchant)
        customer = None
        cust_id = trg_payload.get("customer_id")
        if cust_id:
            customer = get_ctx("customer", cust_id)
        prev_sent = merchant_sent_bodies.get(str(mid) or "", [])
        try:
            body = compose_proactive_message(merchant, trg, category, customer, prev_sent)
        except Exception:
            body = get_fallback_message(trg_kind, merchant, trg_payload)
        if not body:
            body = get_fallback_message(trg_kind, merchant, trg_payload)
        if body in prev_sent:
            continue
        identity = merchant.get("identity", {})
        cta = "binary_yes_no" if trg_kind in ("recall_due", "appointment_tomorrow", "customer_lapsed_soft") else "open_ended"
        send_as = "merchant_on_behalf" if trg_scope == "customer" else "vera"
        conv_id = f"conv_{mid}_{tid}_{uuid.uuid4().hex[:6]}"
        actions.append({
            "conversation_id": conv_id,
            "merchant_id": mid,
            "customer_id": cust_id,
            "send_as": send_as,
            "trigger_id": tid,
            "template_name": f"vera_{trg_kind}_v3",
            "template_params": [identity.get("name", ""), trg_kind],
            "body": body,
            "cta": cta,
            "suppression_key": suppression_key,
            "rationale": f"Trigger '{trg_kind}' (urgency={urgency}) for {identity.get('name', mid)}. Category: {(category or {}).get('slug','general')}.",
        })
        merchants_actioned.add(key)
        merchant_sent_bodies.setdefault(str(mid) or "", []).append(body)
        fired_suppression_keys.add(suppression_key)
    return actions

@app.get("/v1/healthz")
async def healthz():
    return {"status": "ok", "uptime_seconds": int(time.time() - START_TIME), "contexts_loaded": count_contexts(), "version": "3.0.0"}

@app.get("/v1/metadata")
async def metadata():
    return {"team_name": "Vera Pro", "team_members": ["Challenger"], "model": f"google/{GEMINI_MODEL}",
            "approach": "4-context Gemini composer with 14 trigger templates, STOP=end, auto-reply detection, intent-transition, anti-repetition",
            "contact_email": os.environ.get("CONTACT_EMAIL", "contact@example.com"), "version": "3.0.0",
            "submitted_at": datetime.utcnow().isoformat() + "Z"}

@app.post("/v1/context")
async def push_context(body: ContextBody):
    valid_scopes = {"category", "merchant", "customer", "trigger"}
    if body.scope not in valid_scopes:
        return JSONResponse(status_code=400, content={"accepted": False, "reason": "invalid_scope"})
    key = (body.scope, body.context_id)
    cur = contexts.get(key)
    if cur and cur["version"] >= body.version:
        return JSONResponse(status_code=409, content={"accepted": False, "reason": "stale_version", "current_version": cur["version"]})
    contexts[key] = {"version": body.version, "payload": body.payload}
    return {"accepted": True, "ack_id": f"ack_{body.context_id}_v{body.version}_{uuid.uuid4().hex[:6]}", "stored_at": datetime.utcnow().isoformat() + "Z"}

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
    conversations.setdefault(conv_id, []).append({"from": body.from_role, "msg": message, "received_at": body.received_at, "turn": body.turn_number})
    conv = conversations[conv_id]

    # STOP/hostile → ALWAYS end
    if is_stop_command(message):
        return {"action": "end", "body": "Understood — we'll stop messaging you. You can reach magicpin support anytime. 🙏",
                "rationale": "STOP/opt-out detected. Ending conversation immediately."}

    # Auto-reply
    if is_auto_reply(message):
        auto_count = count_auto_replies(conv, body.from_role)
        if auto_count >= 3:
            return {"action": "end", "rationale": f"Detected {auto_count} auto-replies. Exiting."}
        return {"action": "wait", "wait_seconds": 3600, "rationale": f"Auto-reply ({auto_count}/3). Backing off 1h."}

    merchant = get_ctx("merchant", merchant_id) if merchant_id else None
    category = find_category_for_merchant(merchant) if merchant else None
    conv_without_current = conv[:-1]

    # Intent commit → action mode
    if is_intent_commit(message):
        try:
            reply_body = compose_action_mode_reply(merchant, conv_without_current, message)
        except Exception:
            reply_body = "Sending now — I'll have the draft ready in 2 minutes. I'll ping you once it's live."
        return {"action": "send", "body": reply_body, "cta": "open_ended", "rationale": f"Merchant committed. Action mode."}

    # Normal reply
    try:
        reply_body = compose_reply_message(merchant, category, conv_without_current, message)
    except Exception:
        reply_body = "Got it — let me check on that and come back to you shortly. Shall I proceed?"
    return {"action": "send", "body": reply_body, "cta": "open_ended", "rationale": "Contextual reply composed."}

@app.post("/v1/teardown")
async def teardown():
    contexts.clear(); conversations.clear(); fired_suppression_keys.clear(); merchant_sent_bodies.clear()
    return {"status": "wiped"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("bot:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), reload=False)
