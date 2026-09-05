# Razorpay Dispute Agent

Receives Razorpay dispute webhooks, works out what evidence the card network
actually requires for that reason code, checks what the dispute already carries,
and asks an LLM for a recommendation the merchant can act on — all printed to
the terminal as the webhook lands.

```
webhook POST ──▶ verify ──▶ normalize ──▶ investigate ──▶ agent ──▶ report + case file
                signature    reason code    Razorpay API    verdict     (terminal + JSON)
                             → rulebook
```

## The problem

A merchant on Razorpay gets a dispute webhook and is given a deadline — often a
few days — to respond. To answer it they have to know three things that the
webhook does not tell them:

1. **What this reason code actually means.** `10.4` is not self-explanatory, and
   the rules differ per card network. Visa, Mastercard, Amex and RuPay each
   publish their own codes and their own list of what counts as proof.
2. **What evidence that code requires.** Razorpay accepts evidence in fixed
   slots (`shipping_proof`, `access_activity_log`, `billing_proof`, …). Which
   slots matter depends entirely on why the customer disputed.
3. **What they already have versus what they must go and find.** The dispute
   object carries some evidence already; the rest has to come from the
   merchant's own systems before the clock runs out.

In practice this is a manual lookup against a rulebook, per dispute, under time
pressure — so small merchants either contest blindly with the wrong documents,
or accept disputes they could have won.

This project tries to do that lookup automatically the moment the webhook lands,
and hand the merchant a specific answer: contest or accept, what is already on
file, what is missing, and how to get it.

## What works, and what does not

Being honest about the state of this, because the demo does not tell the whole
story.

**Working end to end:**

- Webhook receipt, HMAC verification, event de-duplication, sub-5s ack, and
  off-thread investigation.
- Payload extraction across every envelope shape we hit, including a
  double-nested one that silently dropped events until we archived raw bodies
  and found it.
- Normalization: reason code → network rulebook → required evidence slots →
  weighted coverage of what the dispute already carries. This is deterministic
  code, not the model, and it is the part that actually earns its keep.
- The agent loop: tool use, a structured verdict against a fixed JSON schema, a
  terminal report, per-case JSON files, and a `chat` REPL to interrogate a case.

**Where it falls down:**

- **The LLM step is quota-blocked.** On the Gemini free tier one investigation
  costs 2–5 requests and hits `429` almost immediately, so most runs end in the
  `MANUAL_REVIEW` fallback rather than a real verdict. The plumbing is verified;
  the model's own analysis has not been exercised at any volume. This is the
  single biggest gap between "it runs" and "it works".
- **The agent has no merchant-side tools.** `const.py` instructs it to search
  order records, fulfilment, support tickets and access logs — but `tools.py`
  exposes exactly one tool, `fetch_dispute_expanded_payment`. So evidence can
  never move from *merchant must supply* to *found*; the agent can only ever
  tell you what to go and fetch by hand. Closing this is the highest-value next
  step.
- **It never actually contests.** There is no submit-evidence call. The output
  is a recommendation and a draft, not an action.
- **Never validated against a real dispute.** Every test used doc samples or
  hand-made ids, which 404 against the live API. The `SKIP_DISPUTE_FETCH` demo
  mode exists precisely because of this. The expanded-payment path has never
  returned a real payment object.
- **The reason-code coverage is partial.** `reasoncode.json` holds the card
  network codes; Razorpay's own codes (e.g. `processed_invalid_expired_card`)
  are not in it, so those disputes resolve to zero requirements and 0% coverage.
  Correct behaviour — the code refuses to guess — but useless output.
- **The document → slot mapping is a keyword heuristic.** `DOCUMENT_RULES` in
  `tools.py` folds ~300 free-text phrases onto eleven slots by substring match,
  first rule wins. It has never been measured against labelled disputes, so its
  error rate is unknown.
- **No evaluation exists.** `eval.py` is empty, `gen.py` (synthetic dispute
  generation) is entirely commented out, and `patch.py` imports a `core` module
  that is not in the repo. So there is no answer to "is the recommendation any
  good" — only "did it produce one".
- **State is in-memory and single-process.** The work queue and the
  de-duplication LRU do not survive a restart, and there is no retry for a
  dispute whose investigation failed. The Flask dev server, no auth on `/cases`.

Read the reports accordingly: the requirement list and coverage number are
trustworthy, the recommendation prose is unvalidated.

## What it does

1. **Verifies** the `X-Razorpay-Signature` HMAC before trusting a byte of the body.
2. **De-duplicates** on `x-razorpay-event-id` — Razorpay redelivers on any non-2xx.
3. **Acks in under 5s** and investigates on a worker thread, so Razorpay never retries.
4. **Normalizes** the payload: `reason_code` → `reasoncode.json` → the Razorpay
   evidence slots that rulebook requires → weighted coverage of what the dispute
   already has.
5. **Enriches** with `GET /disputes/:id?expand[]=payment`.
6. **Asks the agent** for a structured verdict, then prints a merchant-readable
   report and writes `cases/<dispute_id>.json`.

Nothing here fabricates dispute data. A field the API did not return stays
`null`, and an unrecognised reason code produces an empty requirement list
rather than a guessed one.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` in the project root:

```ini
RAZORPAY_KEY_ID=rzp_test_xxxxxxxx
RAZORPAY_KEY_SECRET=xxxxxxxx
RAZORPAY_WEBHOOK_SECRET=the-secret-you-typed-into-the-razorpay-webhook-form
GEMINI_API_KEY=xxxxxxxx

# optional
PORT=5050
GEMINI_MODEL=gemini-3.8-flash
SKIP_SIGNATURE_CHECK=1     # accept unsigned posts (local curl only)
SKIP_DISPUTE_FETCH=1       # demo mode: trust the webhook body, skip the API call
NO_COLOR=1                 # plain output
```

`.env` is gitignored. Never commit real keys.

## Running

```bash
python main.py                 # webhook server; reports print here
python main.py run <disp_id>   # investigate a dispute now, no webhook needed
python main.py show <disp_id>  # re-print a case already investigated
python main.py chat <disp_id>  # ask the agent questions about a case
python main.py list            # one line per case with its recommendation
```

Expose it to Razorpay with ngrok and register
`https://<your-tunnel>/webhook/razorpay` in the dashboard:

```bash
ngrok http 5050
```

### Local test

```bash
SKIP_SIGNATURE_CHECK=1 SKIP_DISPUTE_FETCH=1 python main.py

curl -X POST http://127.0.0.1:5050/webhook/razorpay \
  -H "Content-Type: application/json" \
  -H "x-razorpay-event-id: evt_test_001" \
  -d '{
    "event": "payment.dispute.created",
    "payload": { "dispute": { "entity": {
      "id": "disp_demo123", "status": "open", "phase": "chargeback",
      "reason_code": "10.4", "amount": 6000000, "currency": "INR",
      "respond_by": 1778500000,
      "evidence": { "shipping_proof": ["doc_1"] }
    }}}
  }'
```

The server terminal shows progress, then the report:

```
investigating disp_demo123
  done skipped the api fetch (SKIP_DISPUTE_FETCH=1), using the webhook entity
  done agent finished - 2 rounds, 1 tool calls

==============================================================================
 DISPUTE disp_demo123  |  payment.dispute.created
==============================================================================
  Reason    10.4  Other Fraud – Card-Absent Environment
  Network   Visa / Fraud
  Amount    Rs 60000.0   Status open / chargeback
  Deadline  112.4 hours to respond
  Evidence  43% of required slots covered (3/7 weighted)

  REQUEST_MORE_EVIDENCE   confidence: medium
  ...
  Talk to it       python main.py chat disp_demo123
==============================================================================
```

## HTTP endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/webhook/razorpay` | the URL to register in the Razorpay dashboard |
| GET | `/health` | liveness and config sanity, no secrets |
| GET | `/cases` | every case, with its recommendation |
| GET | `/cases/<dispute_id>` | one full case file |
| POST | `/replay/<dispute_id>` | run the pipeline by hand, no webhook needed |

## Files

| File | Role |
|---|---|
| `main.py` | Flask server, normalization, worker, terminal report, CLI |
| `tools.py` | Razorpay API calls + the reason-code → evidence-slot normalizer |
| `agent.py` | the LLM loop (tool use + structured verdict) and chat |
| `const.py` | system prompt and the verdict JSON schema |
| `reasoncode.json` | per-network reason codes and the documents each requires |
| `cases/` | one JSON case file per dispute, plus `events.jsonl` and `raw/` |

## Webhook payload shapes

`dispute_entity()` finds the dispute wherever it is — the documented
`payload.dispute.entity`, a double-nested envelope, a flattened `payload.dispute`,
a bare entity as the body, or a payload that only carries the dispute id (it
fetches the rest from the API). A payment-only event is correctly ignored.

Every accepted body is archived verbatim to `cases/raw/<event-id>.json`, so a
payload that misbehaves is on disk rather than only in a terminal scrollback.

## The verdict

The agent must answer in the schema in `const.py`:

| Field | Meaning |
|---|---|
| `recommendation` | `CONTEST` / `ACCEPT` / `REQUEST_MORE_EVIDENCE` / `MANUAL_REVIEW` |
| `confidence` | `high` / `medium` / `low` |
| `headline` | one sentence for the notification |
| `reasoning` | why, referencing the reason code and the evidence actually present |
| `evidence_found` | each item tagged with where it came from |
| `missing_evidence` | each with *why it matters* and *how to obtain it* |
| `risks` | contradictions, deadline pressure, anything weakening a contest |
| `merchant_actions` | ordered steps with priorities |
| `proposed_response` | draft rebuttal the merchant can edit |
| `submission_ready` | true only when every required slot is covered |

If the model or the network fails, the verdict degrades to a real-shaped
`MANUAL_REVIEW` carrying the error — a webhook worker always ends up with
something to write down.

## Troubleshooting

**`api fetch failed (404)`** — the dispute id does not exist for your API keys.
Expected for hand-made demo ids; the agent falls back to the webhook's own
entity. On a real id, check you are not mixing `rzp_test_*` keys with live data.

**`invalid signature` (401)** — `RAZORPAY_WEBHOOK_SECRET` does not match the
secret in the dashboard, or you are posting by hand. Use `SKIP_SIGNATURE_CHECK=1`
locally.

**`429 ... generate_content_free_tier_requests`** — Gemini free-tier quota. One
investigation costs 2–5 requests. Enable billing on the key, or set
`GEMINI_MODEL` to a model with free-tier room.

**Edits do not take effect** — `app.run` has no reloader. Restart the server.

**Reason code resolves to nothing / 0% coverage** — that code is not in
`reasoncode.json`, so no requirement list is produced. This is deliberate: the
code never guesses a rulebook it does not have.
