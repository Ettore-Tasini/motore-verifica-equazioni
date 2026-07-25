"""
Suite di regressione sulla QUALITA' DEI MESSAGGI, non solo sugli stati.

Perche' esiste: per un po' i test hanno controllato solo che ogni riga
ricevesse lo stato giusto (ok/error/unreadable). Cosi' facendo sono passati
inosservati messaggi tecnicamente "giusti" ma inutili o fuorvianti per uno
studente ("qui non torna quasi nulla", "troppo complesso" su un banale errore
di segno, un "2x - 6" che lo studente non aveva mai scritto). Lo stato giusto
con la spiegazione sbagliata, per questo progetto, e' comunque un fallimento:
il valore del quaderno sta tutto nella spiegazione.

Quindi qui ogni caso dichiara COSA deve comparire nel messaggio e cosa NON
deve comparire, ed e' quello il test.

Uso:
    python test_diagnostica.py
Non serve avviare il server: chiama process_sheet direttamente.

Se aggiungi un caso nuovo, prima leggi il messaggio prodotto e chiediti "uno
studente di terza media capirebbe cosa ha sbagliato e come rimediare?".
Se la risposta e' no, il codice va sistemato, non il test.
"""
import sys

import verify_engine as engine

# (nome, righe, [(indice_riga, stato_atteso, [frammenti_richiesti], [frammenti_vietati])])
CASI = [
    (
        "raccoglimento di x^2 quando un termine non lo contiene",
        ["2*x^2+1=3*x+4", "2*x^2-3*x=3", "x^2*(2-3)=3"],
        [(2, "error", ["Non puoi raccogliere x^2", "-3x", "non è un multiplo"],
          ["moltiplicando per", "troppo complesso"])],
    ),
    (
        "raccoglimento di x quando resta un termine noto",
        ["x^2 + 2*x + 3 = 4*x", "x^2 - 2*x + 3 = 0", "x*(x - 2 + 3) = 0", "x = 0 & x = -1"],
        [(2, "error", ["Non puoi raccogliere x", "non è un multiplo"], ["troppo complesso"]),
         (3, "skip", [], [])],
    ),
    (
        "raccoglimento lecito ma parentesi sbagliata",
        ["x^2 + 3*x = 0", "x*(x+5) = 0"],
        [(1, "error", ["Raccogliendo x", "x(x + 3)", "non x(x + 5)"], ["troppo complesso"])],
    ),
    (
        "distribuzione incompleta di un fattore numerico",
        ["2*(x-3)=4", "2*x-3=4", "x=7/2"],
        [(1, "error", ["2(x - 3)", "2x - 6", "OGNI termine"], ["raccogliere"])],
    ),
    (
        "cambio di tutti i segni con un termine sbagliato",
        ["-x^2 + 2*x - 1 = 0", "x^2 - 2*x + 4 = 0", "-2 ± sqrt(9 - 4) => x = -2"],
        [(1, "error", ["cambiando tutti i segni", "+1", "+4"], ["non torna quasi nulla"]),
         (2, "skip", [], [])],
    ),
    (
        "trasporto senza cambiare segno, con frazione intatta",
        ["(x+3)/2 + 4/(3*x-4) = 3*x^2+1",
         "((x+3)*(3*x-4)+8)/(6*x-8) = 3*x^2+1",
         "((x+3)*(3*x-4)+8)/(6*x-8) - 3*x^2+1 = 0"],
        [(2, "error", ["senza cambiargli segno", "deve diventare"], ["troppo complesso"])],
    ),
    (
        "denominatore eliminato solo sulla frazione",
        ["((x+3)*(3*x-4)+8)/(6*x-8) - 3*x^2+1 = 0",
         "(x+3)*(3*x-4) + 8 - 3*x^2+1 = 0"],
        [(1, "error", ["OGNI termine", "6x - 8"], ["troppo complesso"])],
    ),
    (
        "mcm sbagliato combinando due frazioni",
        ["1/(1+x) + 1/(1-x) = 3*x^2 + 2*x", "2*x/((1+x)*(1-x)) = x*(3*x+2)"],
        [(1, "error", ["combinato le frazioni", "minimo comune denominatore"], ["troppo complesso"])],
    ),
    (
        "risposta finale dichiarata impossibile dopo una freccia",
        ["x^2 + 2*x - 6 = 3*x^2 - 3", "x^2 - 6 = -2*x + 3*x^2 - 3",
         "-2*x^2 - 3 = -2*x", "2*x^2 - 2*x + 3 = 0",
         "2 = sqrt(4 - 24) => IMPOSSIBILE"],
        [(4, "skip", [], [])],
    ),
    (
        "disequazione: fuori portata, ma va detto chiaramente",
        ["2*x^2+3*x<-4", "-2*x^2-3*x<4"],
        [(0, "unreadable", ["disequazione"], ["attesa esattamente una"])],
    ),
    # --- casi che devono restare OK: guardia contro i falsi allarmi ---
    (
        "raccoglimento corretto: nessun errore",
        ["x^2 + 3*x = 0", "x*(x+3) = 0"],
        [(1, "ok", [], [])],
    ),
    (
        "eliminazione corretta del denominatore: nessun errore",
        ["1/(x-1) + 1/(x+1) = 1", "2*x = x**2 - 1"],
        [(1, "ok", [], [])],
    ),
    (
        "trasporto e ricombinazione insieme: nessun errore",
        ["1/(x-1) + 1/(x+1) = 1", "1/(x-1) + 1/(x+1) - 1 = 0"],
        [(1, "ok", [], [])],
    ),
    (
        "divisione per un coefficiente: nessun errore",
        ["2*x + 4 = 6", "x + 2 = 3", "x = 1"],
        [(1, "ok", [], []), (2, "ok", [], [])],
    ),
]


def testo_riga(step):
    return (step.get("diagnosis") or "") + " " + (step.get("note") or "")


def main():
    falliti = []
    for nome, righe, attese in CASI:
        rows = [{"index": i, "plain": r} for i, r in enumerate(righe)]
        try:
            risultato = engine.process_sheet(rows)
        except Exception as e:
            falliti.append(f"[{nome}] ECCEZIONE: {type(e).__name__}: {e}")
            continue
        per_indice = {s["index"]: s for s in risultato["steps"]}
        for idx, stato, richiesti, vietati in attese:
            step = per_indice.get(idx)
            if step is None:
                falliti.append(f"[{nome}] riga {idx}: nessun verdetto restituito")
                continue
            if step["status"] != stato:
                falliti.append(
                    f"[{nome}] riga {idx}: stato '{step['status']}', atteso '{stato}'\n"
                    f"    messaggio: {testo_riga(step).strip()}")
                continue
            messaggio = testo_riga(step)
            for frammento in richiesti:
                if frammento not in messaggio:
                    falliti.append(
                        f"[{nome}] riga {idx}: manca '{frammento}' nel messaggio\n"
                        f"    messaggio: {messaggio.strip()}")
            for frammento in vietati:
                if frammento in messaggio:
                    falliti.append(
                        f"[{nome}] riga {idx}: il messaggio contiene '{frammento}', che qui è fuorviante\n"
                        f"    messaggio: {messaggio.strip()}")

    if falliti:
        print(f"FALLITI: {len(falliti)}\n")
        for f in falliti:
            print(" - " + f)
        return 1
    print(f"Tutti i {len(CASI)} casi passano (stati e messaggi).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
