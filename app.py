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
import os
from typing import List, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import verify_engine as engine

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


# ---------- Modelli della richiesta/risposta (rispecchiano il contratto) ----------
class RowIn(BaseModel):
    index: int
    plain: str
    latex: Optional[str] = ""
    role_hint: Optional[str] = "equation"


class VerifyRequest(BaseModel):
    variable_hint: Optional[str] = None
    rows: List[RowIn]


class StepOut(BaseModel):
    index: int
    status: str  # first | ok | error | unreadable | skip
    relation: Optional[str] = None
    note: Optional[str] = None
    diagnosis: Optional[str] = None


class FinalCheckOut(BaseModel):
    status: str  # ok | incomplete | invalid_solution_present | no_final_solution | unknown
    message: str
    correct_solutions: Optional[List[str]] = None
    solutions_are_equivalent_forms: Optional[bool] = False


class VerifyResponse(BaseModel):
    steps: List[StepOut]
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
