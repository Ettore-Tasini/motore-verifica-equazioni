"""
Web server: espone il motore di verifica (verify_engine.py) come endpoint
HTTP che il foglio HTML può chiamare. Implementa esattamente il contratto
descritto in contratto-api.md. Fa solo da "involucro": tutta la logica vera
sta in verify_engine.process_sheet, già testata separatamente.

Avvio in locale (per provarlo sul tuo computer prima di pubblicarlo):
    pip install -r requirements.txt
    uvicorn app:app --reload --port 8000

Poi il foglio HTML può chiamare http://localhost:8000/verify
"""
import json
import os
from typing import Optional

import redis as redis_lib
import verify_engine as engine
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Motore di verifica equazioni")

# ---------- CORS ----------
# ALLOWED_ORIGINS va impostato come variabile d'ambiente su Render, es:
#   ALLOWED_ORIGINS=https://tuosito.netlify.app,http://localhost:8000
# "null" serve per quando apri il foglio HTML direttamente come file locale
# (file://): il browser manda Origin: null in quel caso.
_origins_env = os.environ.get("ALLOWED_ORIGINS", "null")
ALLOWED_ORIGINS = [o.strip() for o in _origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type", "X-Api-Key"],
)

# ---------- Chiave condivisa ----------
# Va impostata come variabile d'ambiente su Render: API_SHARED_KEY=qualcosa
# Se non è impostata, il server rifiuta ogni richiesta (fail-safe: meglio
# rotto-e-visibile che aperto-per-sbaglio).
API_SHARED_KEY = os.environ.get("API_SHARED_KEY")


def check_api_key(x_api_key: Optional[str]):
    if not API_SHARED_KEY:
        raise HTTPException(status_code=500, detail="Server non configurato: manca API_SHARED_KEY.")
    if x_api_key != API_SHARED_KEY:
        raise HTTPException(status_code=401, detail="Chiave API mancante o non valida.")


# ---------- Libreria di test condivisa (foglio -> Claude) ----------
# Cassetta postale temporanea, non un archivio permanente: il foglio scrive
# qui i test nuovi/modificati, Claude li legge e li salva davvero (nel
# progetto), poi li conferma con /tests/ack per svuotare la cassetta.
# Nessuna persistenza sul piano gratuito del Key Value: va bene, è un
# passaggio transitorio, non l'unica copia dei dati (quella resta sul
# dispositivo dello studente finché non viene letta).
TEST_LIBRARY_REDIS_URL = os.environ.get("TEST_LIBRARY_REDIS_URL")
_redis_client = redis_lib.from_url(TEST_LIBRARY_REDIS_URL) if TEST_LIBRARY_REDIS_URL else None
TEST_KEY_PREFIX = "testlib:"


def _require_redis():
    if _redis_client is None:
        raise HTTPException(status_code=500, detail="Server non configurato: manca TEST_LIBRARY_REDIS_URL.")
    return _redis_client


class TestSubmission(BaseModel):
    id: str
    name: str
    righe: list[str]


class SubmitTestsRequest(BaseModel):
    tests: list[TestSubmission]


class AckTestsRequest(BaseModel):
    ids: list[str]


# ---------- Modelli della richiesta/risposta (rispecchiano il contratto) ----------
class RowIn(BaseModel):
    index: int
    plain: str
    latex: Optional[str] = ""
    role_hint: Optional[str] = "equation"


class VerifyRequest(BaseModel):
    variable_hint: Optional[str] = None
    rows: list[RowIn]


class StepOut(BaseModel):
    index: int
    status: str  # first | ok | error | unreadable | skip
    relation: Optional[str] = None
    note: Optional[str] = None
    diagnosis: Optional[str] = None


class FinalCheckOut(BaseModel):
    status: str  # ok | incomplete | invalid_solution_present | no_final_solution | unknown
    message: str
    correct_solutions: Optional[list[str]] = None
    solutions_are_equivalent_forms: Optional[bool] = False


class VerifyResponse(BaseModel):
    steps: list[StepOut]
    final_check: FinalCheckOut


@app.get("/")
def health():
    return {"status": "ok", "message": "Motore di verifica attivo."}


@app.post("/verify", response_model=VerifyResponse)
def verify(payload: VerifyRequest, x_api_key: Optional[str] = Header(default=None)):
    check_api_key(x_api_key)
    rows = [r.model_dump() for r in payload.rows]
    result = engine.process_sheet(rows, variable_hint=payload.variable_hint)
    return result


@app.post("/tests/submit")
def submit_tests(payload: SubmitTestsRequest, x_api_key: Optional[str] = Header(default=None)):
    check_api_key(x_api_key)
    client = _require_redis()
    for t in payload.tests:
        client.set(TEST_KEY_PREFIX + t.id, json.dumps({"id": t.id, "name": t.name, "righe": t.righe}))
    return {"status": "ok", "received": len(payload.tests)}


@app.get("/tests/pending")
def pending_tests(x_api_key: Optional[str] = Header(default=None)):
    check_api_key(x_api_key)
    client = _require_redis()
    keys = client.keys(TEST_KEY_PREFIX + "*")
    tests = [json.loads(client.get(k)) for k in keys]
    return {"tests": tests}


@app.post("/tests/ack")
def ack_tests(payload: AckTestsRequest, x_api_key: Optional[str] = Header(default=None)):
    check_api_key(x_api_key)
    client = _require_redis()
    if payload.ids:
        client.delete(*[TEST_KEY_PREFIX + i for i in payload.ids])
    return {"status": "ok", "deleted": len(payload.ids)}
