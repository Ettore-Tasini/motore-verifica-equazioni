"""
Core verification logic: parsing, relation classification, and the final
check against the true solution set. Pure Python + SymPy, no web framework
dependency, so it can be tested and reasoned about in isolation.

Scope of this first version (per the agreed incremental roadmap): single
polynomial equations, degree 1 or 2, one variable, no parameters. Radicals,
fractions, logs/exponentials, trig, and systems come in later phases and
will currently fall back to an honest "not supported yet" result rather
than a wrong answer.
"""
import re

from sympy import E as _E
from sympy import (
    Add,
    Eq,
    FiniteSet,
    Mul,
    Poly,
    Pow,
    S,
    Symbol,
    cancel,
    expand,
    lcm,
    simplify,
    together,
)
from sympy import Float as _Float
from sympy import Integer as _Integer
from sympy import Rational as _Rational
from sympy import cos as _cos
from sympy import exp as _exp
from sympy import log as _log
from sympy import pi as _pi
from sympy import sin as _sin
from sympy import sqrt as _sqrt
from sympy import tan as _tan
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

# Only these names are exposed to the parser — deliberately restrictive,
# since this endpoint will be reachable on the internet (even behind a
# shared key). No arbitrary code execution should ever be possible here.
# Integer/Float/Symbol are required by SymPy's own auto_number/auto_symbol
# transformations (they build numbers and variable names out of these), not
# an extra surface — everything else here is plain math.
SAFE_GLOBALS = {
    'sqrt': _sqrt, 'log': _log, 'sin': _sin, 'cos': _cos, 'tan': _tan,
    'exp': _exp, 'pi': _pi, 'E': _E, 'Rational': _Rational,
    'Integer': _Integer, 'Float': _Float, 'Symbol': Symbol,
}
TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application, convert_xor)


def safe_parse(text):
    """Parses a math expression from student/OCR text into a SymPy expression.
    Restricted namespace: only the names in SAFE_GLOBALS are available, plus
    auto-created single-letter symbols for the variable(s)."""
    text = text.strip()
    return parse_expr(text, transformations=TRANSFORMATIONS, global_dict=dict(SAFE_GLOBALS), evaluate=True)


# Con evaluate=False il parser di SymPy genera codice che costruisce l'albero
# a mano, quindi ha bisogno dei costruttori (Mul/Add/Pow) nel namespace:
# senza, solleva NameError e il parsing grezzo non funziona MAI. Sono
# costruttori di espressioni, stessa categoria di Integer/Float/Symbol gia'
# presenti: non allargano la superficie di rischio.
RAW_GLOBALS = dict(SAFE_GLOBALS, Mul=Mul, Add=Add, Pow=Pow)


def safe_parse_raw(text):
    """Come safe_parse ma SENZA valutare: conserva la struttura esattamente
    come l'ha scritta lo studente. Serve per le diagnosi strutturali (es.
    riconoscere un raccoglimento) che la valutazione distruggerebbe: SymPy
    normalmente collassa 'x^2*(2-3)' in '-x^2', cancellando ogni traccia del
    fatto che lo studente aveva raccolto x^2. Da NON usare per i confronti
    matematici (un albero non valutato non e' affidabile per quelli)."""
    return parse_expr(text.strip(), transformations=TRANSFORMATIONS,
                      global_dict=dict(RAW_GLOBALS), evaluate=False)


def split_equation(text, raw=False):
    """'lhs = rhs' (textual) -> (lhs_expr, rhs_expr). Raises ValueError with a
    human-readable reason on anything that isn't exactly one '='. Con
    raw=True usa safe_parse_raw (struttura conservata, vedi sopra)."""
    if text.count('=') != 1:
        raise ValueError(f"attesa esattamente una '=', trovate {text.count('=')}")
    left_txt, right_txt = text.split('=')
    parser = safe_parse_raw if raw else safe_parse
    return parser(left_txt), parser(right_txt)


def detect_variable(*exprs, hint=None):
    free = set()
    for e in exprs:
        free |= e.free_symbols
    if hint:
        for s in free:
            if s.name == hint:
                return s
    if free:
        return min(free, key=lambda s: s.name)
    return Symbol('x')


def poly_coeffs(expr, var):
    """Expands expr and returns {0: c0, 1: c1, 2: c2} as EXACT SymPy values,
    or None if expr isn't a degree<=2 polynomial in `var` alone (e.g. it has
    another free symbol/parameter, or a degree we don't support yet)."""
    expr = expand(expr)
    if expr.free_symbols - {var}:
        return None  # extra symbols/parameters: out of scope for now
    try:
        p = Poly(expr, var)
    except Exception:
        return None
    if p.degree() > 2:
        return None
    coeffs = {0: S(0), 1: S(0), 2: S(0)}
    for monom, coeff in zip(p.monoms(), p.coeffs()):
        coeffs[monom[0]] = coeff
    return coeffs


def check_relation(prev_lhs, prev_rhs, cur_lhs, cur_rhs, var):
    """Exact analogue of the JS engine's proportionality check, but symbolic
    (no numeric sampling, no floating-point tolerance needed)."""
    p = poly_coeffs(prev_lhs - prev_rhs, var)
    c = poly_coeffs(cur_lhs - cur_rhs, var)
    if p is None or c is None:
        return {'supported': False}
    k = None
    for deg in (2, 1, 0):
        if simplify(p[deg]) != 0:
            k = simplify(c[deg] / p[deg])
            break
    if k is None:
        # prev side was identically 0=0 (degenerate) — equivalent only if cur is too
        equivalent = all(simplify(c[d]) == 0 for d in (0, 1, 2))
        return {'supported': True, 'equivalent': equivalent, 'k': None, 'p': p, 'c': c}
    equivalent = (k != 0) and all(simplify(c[deg] - k * p[deg]) == 0 for deg in (0, 1, 2))
    return {'supported': True, 'equivalent': equivalent, 'k': k, 'p': p, 'c': c}


def side_denominator(expr, var):
    """Denominatore di expr dopo averlo ricondotto a una frazione unica
    (together); Integer(1) se expr non ha alcuna frazione. Usa together (non
    cancel) apposta: cancel semplificherebbe via i fattori comuni tra
    numeratore e denominatore, nascondendo un valore che andrebbe escluso dal
    dominio (es. (x+3)/(x**2-9) ha comunque x=3 da escludere)."""
    _, denom = together(expr).as_numer_denom()
    return denom


def row_has_fraction(lhs, rhs, var):
    """True se lhs o rhs contiene, dopo together, un denominatore che
    dipende da var — cioè questa riga è (o contiene) una frazione algebrica
    in var."""
    return var in side_denominator(lhs, var).free_symbols or var in side_denominator(rhs, var).free_symbols


def common_denominator(lhs, rhs, var):
    """mcm dei denominatori var-dipendenti di lhs e rhs, calcolato SOLO sui
    lati passati (mai mischiando prev e cur: moltiplicare due volte una riga
    già cancellata falserebbe il confronto). Integer(1) se nessuno dei due
    lati ha un denominatore che dipende da var."""
    dl, dr = side_denominator(lhs, var), side_denominator(rhs, var)
    dl_has_var, dr_has_var = var in dl.free_symbols, var in dr.free_symbols
    if dl_has_var and dr_has_var:
        return lcm(dl, dr)
    if dl_has_var:
        return dl
    if dr_has_var:
        return dr
    return S(1)


def clear_denominator(lhs, rhs, D):
    """(lhs*D, rhs*D) espansi dopo cancel. Il risultato può non essere
    polinomiale se D non è il denominatore giusto per questi lati: il
    chiamante verifica il risultato via poly_coeffs/check_relation, questa
    funzione non lo garantisce da sola."""
    return expand(cancel(lhs * D)), expand(cancel(rhs * D))


def rational_ratio(prev_lhs, prev_rhs, cur_lhs, cur_rhs, var):
    """Classificatore primario per transizioni tra righe con frazioni:
    calcola ratio = cancel((cur_lhs-cur_rhs)/(prev_lhs-prev_rhs)). Se il
    rapporto risulta 'pulito' (var non resta al denominatore del rapporto
    stesso) il passaggio è valido — equivalenza se il rapporto è una costante
    pura, implicazione se dipende da var (moltiplicazione per un'espressione
    che può annullarsi). Se il rapporto resta 'sporco' non è classificabile
    con sicurezza: {'supported': False} — non è MAI usato per dedurre un
    errore, solo equivalence/implication."""
    diff_p = cancel(prev_lhs - prev_rhs)
    diff_c = cancel(cur_lhs - cur_rhs)
    if diff_p == 0:
        return {'supported': True, 'equivalent': diff_c == 0, 'relation_kind': 'equivalence'}

    ratio = cancel(diff_c / diff_p)
    _, ratio_denom = ratio.as_numer_denom()
    if var in ratio_denom.free_symbols:
        return {'supported': False}
    if var in ratio.free_symbols:
        return {'supported': True, 'equivalent': True, 'relation_kind': 'implication'}
    return {'supported': True, 'equivalent': ratio != 0, 'relation_kind': 'equivalence'}


def check_fraction_relation(prev_lhs, prev_rhs, cur_lhs, cur_rhs, var):
    """Layer sopra check_relation per transizioni con frazioni: prova prima
    rational_ratio (decide ok/implication in un colpo solo, anche per casi
    insoliti come moltiplicare per il denominatore più un fattore extra). Se
    il rapporto è sporco e la riga precedente ha un denominatore var-
    dipendente e la riga corrente è già polinomiale pulita, calcola il
    denominatore comune di prev e riusa check_relation (e quindi
    diagnose_step, tramite diagnose_fraction_step) per vedere se è un errore
    di cancellazione riconoscibile. Ritorna un dict compatibile con
    check_relation ({'supported', 'equivalent'}) più 'relation_kind', e
    quando equivalent=False anche 'expected_lhs'/'expected_rhs' per la
    diagnosi. Se nessuno dei due percorsi si applica: {'supported': False}
    (fallback onesto, mai un errore inventato)."""
    ratio_result = rational_ratio(prev_lhs, prev_rhs, cur_lhs, cur_rhs, var)
    if ratio_result['supported']:
        return ratio_result

    if not row_has_fraction(prev_lhs, prev_rhs, var):
        return {'supported': False}

    # Prima di tutto: la frazione e' rimasta identica e sono cambiati solo i
    # termini polinomiali? (es. "porto tutto a primo membro"). E' il caso
    # piu' semplice e va riconosciuto per primo, altrimenti finisce nel ramo
    # "combina frazioni" o nel fallback, con spiegazioni che non c'entrano.
    transport = check_fraction_transport(prev_lhs, prev_rhs, cur_lhs, cur_rhs, var)
    if transport is not None:
        return transport

    partial = check_partial_denominator_clearing(prev_lhs, prev_rhs, cur_lhs, cur_rhs, var)
    if partial is not None:
        return partial

    if row_has_fraction(cur_lhs, cur_rhs, var):
        # La riga corrente ha ancora una frazione: non è un tentativo di
        # eliminare il denominatore (quello produce un polinomio pulito),
        # ma di raccogliere più frazioni dello stesso lato in una sola.
        # Il confronto giusto è contro prev ricombinato (together), non
        # contro prev moltiplicato per D.
        expected_lhs, expected_rhs = together(prev_lhs), together(prev_rhs)
        diag = diagnose_combine_step(prev_lhs, prev_rhs, expected_lhs, expected_rhs, cur_lhs, cur_rhs, var)
        if diag is None:
            return {'supported': False}
        return {'supported': True, 'equivalent': False, 'relation_kind': 'equivalence', 'combine_diagnosis': diag}

    D = common_denominator(prev_lhs, prev_rhs, var)
    expected_lhs, expected_rhs = clear_denominator(prev_lhs, prev_rhs, D)

    cleared = check_relation(expected_lhs, expected_rhs, cur_lhs, cur_rhs, var)
    if not cleared['supported']:
        return {'supported': False}

    result = {'supported': True, 'equivalent': cleared['equivalent'], 'relation_kind': 'implication'}
    if not cleared['equivalent']:
        result['expected_lhs'] = expected_lhs
        result['expected_rhs'] = expected_rhs
    return result


def fmt_num(v):
    v = simplify(v)
    return str(v)


def fmt_term(coeff, degree, var):
    coeff = simplify(coeff)
    if coeff == 0:
        return '0'
    sign = '-' if coeff.could_extract_minus_sign() else '+'
    absval = simplify(-coeff if coeff.could_extract_minus_sign() else coeff)
    numstr = fmt_num(absval)
    if degree == 0:
        return f"{sign}{numstr}"
    varpart = str(var) if degree == 1 else f"{var}^{degree}"
    return f"{sign}{varpart}" if numstr == '1' else f"{sign}{numstr}{varpart}"


def term_label(degree, var):
    if degree == 0:
        return 'termine noto'
    if degree == 1:
        return f"termine con {var}"
    return f"termine con {var}^{degree}"


def side_has_content(pSide, cSide):
    """True if this side is ever anything other than plain 0, before or after."""
    return any(simplify(pSide[d]) != 0 for d in (0, 1, 2)) or any(simplify(cSide[d]) != 0 for d in (0, 1, 2))


def _house_style(s):
    s = s.replace('**', '^')
    s = re.sub(r'(?<=[0-9)])\*(?=[a-zA-Z(])', '', s)
    s = s.replace('*', '')
    return s


def format_expr(expr):
    """Renders a SymPy expression in the same house style used elsewhere
    (2x not 2*x, x^2 not x**2) instead of SymPy's own str() conventions."""
    return _house_style(str(expand(expr)))


def format_expr_raw(expr):
    """Same house style as format_expr, but WITHOUT expand() first — for
    grounded diagnosis on expressions with a fraction, where expand() would
    distribute the numerator over the denominator (e.g. (x+1)/(x-2) becomes
    x/(x-2)+1/(x-2)), no longer matching what the student actually wrote."""
    return _house_style(str(expr))


def _pulled_factor(expr, var):
    """Se expr ha la forma F*(...) — un unico fattore Add (la parentesi
    raccolta) e il resto un monomio puro c*var**k, k>=1 — ritorna (F, k).
    None se questa struttura non è individuabile (nessun raccoglimento
    esplicito, o la parentesi contiene altri simboli). Lavora sull'espressione
    COSÌ COM'È STATA SCRITTA, prima di qualunque expand(): expand
    distribuirebbe subito F dentro la parentesi, cancellando l'unico indizio
    che lo studente ha provato a raccogliere qualcosa."""
    factors = Mul.make_args(expr)
    add_factors = [f for f in factors if f.is_Add]
    if len(add_factors) != 1:
        return None
    rest = [f for f in factors if f is not add_factors[0]]
    F = Mul(*rest) if rest else S(1)
    if F == 1 or F.free_symbols != {var}:
        return None
    try:
        p = Poly(F, var)
    except Exception:
        return None
    monoms = p.monoms()
    if len(monoms) != 1 or monoms[0][0] < 1:
        return None
    return F, monoms[0][0]


def _bad_factor_diagnosis(prev_lhs, prev_rhs, cur_lhs, cur_rhs, var,
                          raw_cur_lhs=None, raw_cur_rhs=None):
    """Riconosce un raccoglimento impossibile: lo studente ha raccolto un
    fattore var**k su un lato (es. x^2) ma quel lato, nel passaggio
    precedente, aveva un termine di grado inferiore a k (es. 3x) — quel
    termine non è un multiplo del fattore raccolto, quindi il raccoglimento
    non è valido a prescindere da cosa scrive dentro la parentesi. None se il
    pattern non si applica (nessun raccoglimento esplicito individuato, o il
    grado raccolto è comunque valido — allora l'errore è altrove, tipicamente
    un pezzo dimenticato dentro la parentesi).

    raw_cur_lhs/raw_cur_rhs sono le stesse righe parsate SENZA valutazione:
    quando ci sono si cerca il raccoglimento li' dentro, perche' la versione
    valutata puo' aver gia' cancellato la parentesi (es. 'x^2*(2-3)' diventa
    '-x^2': senza la forma grezza il raccoglimento sarebbe invisibile e lo
    studente si vedrebbe un messaggio generico su tutt'altro)."""
    sides = (
        (prev_lhs, cur_lhs, raw_cur_lhs if raw_cur_lhs is not None else cur_lhs, "a sinistra"),
        (prev_rhs, cur_rhs, raw_cur_rhs if raw_cur_rhs is not None else cur_rhs, "a destra"),
    )
    for prev_side, _cur_side, search_side, label in sides:
        # Il raccoglimento potrebbe non coprire l'intero lato (es. resta un
        # termine noto fuori dalla parentesi: "x^2*(x+3) - 4"): va cercato in
        # ciascun addendo, non solo nel lato preso per intero.
        terms = search_side.args if search_side.is_Add else (search_side,)
        for term in terms:
            pulled = _pulled_factor(term, var)
            if pulled is None:
                continue
            F, k = pulled
            prev_coeffs = poly_coeffs(prev_side, var)
            if prev_coeffs is None:
                continue
            offending = [d for d in range(k) if simplify(prev_coeffs[d]) != 0]
            if not offending:
                continue
            d = max(offending)
            term_str = fmt_term(prev_coeffs[d], d, var)
            if term_str.startswith('+'):
                term_str = term_str[1:]
            factor_str = format_expr_raw(F)
            return (f"Non puoi raccogliere {factor_str} {label}: il termine {term_str} "
                    f"non è un multiplo di {factor_str}. Per raccogliere {factor_str} "
                    f"ogni termine deve contenerlo.")
    return None


def _factor_and_paren(expr):
    """Da un'espressione della forma F*(...) ritorna (F, parentesi) con
    ENTRAMBI i pezzi valutati, oppure None. Valutare i pezzi e' obbligatorio
    se expr viene da safe_parse_raw: la forma grezza di '2*(x-3)' e'
    '2*(x - 1*3)' e stamparla cosi' com'e' produrrebbe '2(x - 13)' (il
    formattatore toglie gli asterischi), cioe' un messaggio palesemente
    sbagliato sotto gli occhi dello studente."""
    factors = Mul.make_args(expr)
    add_factors = [f for f in factors if f.is_Add]
    if len(add_factors) != 1:
        return None
    paren = add_factors[0].doit(deep=True)
    rest = [f for f in factors if f is not add_factors[0]]
    F = (Mul(*rest) if rest else S(1)).doit(deep=True)
    if F == 1 or not paren.is_Add:
        return None
    return F, paren


def _distribution_mismatch_diagnosis(prev_side, cur_side, var, label, raw_prev_side=None):
    """Riconosce una distribuzione incompleta: la riga precedente aveva la
    forma F*(...) e la riga corrente sembra un tentativo di distribuirla che
    non corrisponde all'espansione corretta — nomina il termine dentro la
    parentesi che non è stato moltiplicato. None se quella forma non c'è
    (allora la causa dell'errore è un'altra e non va inventata qui).

    raw_prev_side è la riga precedente parsata senza valutazione, ed è
    quella che conta davvero: SymPy espande da solo i fattori numerici in
    fase di parsing ('2*(x-3)' diventa subito '2x - 6'), quindi senza la
    forma grezza il caso più comune di tutti — dimenticare di moltiplicare
    il numero per il secondo termine della parentesi — sarebbe invisibile, e
    lo studente si vedrebbe citare un '2x - 6' che non ha mai scritto."""
    source = raw_prev_side if raw_prev_side is not None else prev_side
    parts = _factor_and_paren(source)
    if parts is None:
        return None
    F, paren = parts
    expected = expand(F * paren)
    # Coerenza fra forma grezza e forma valutata: se non combaciano, la
    # struttura letta non è davvero quella della riga precedente e qualsiasi
    # spiegazione costruita sopra sarebbe inventata.
    if simplify(cancel(expected - expand(prev_side))) != 0:
        return None
    if simplify(cancel(cur_side - expected)) == 0:
        return None
    expected_coeffs = poly_coeffs(expected, var)
    cur_coeffs = poly_coeffs(cur_side, var)
    if expected_coeffs is None or cur_coeffs is None:
        return None
    diffs = [d for d in (0, 1, 2) if simplify(expected_coeffs[d] - cur_coeffs[d]) != 0]
    if not diffs:
        return None
    d = max(diffs)
    f_str, paren_str = format_expr_raw(F), format_expr_raw(paren)
    return (f"Nel membro {label} hai moltiplicato {f_str} solo per una parte della parentesi: "
            f"{f_str}({paren_str}) fa {format_expr(expected)}, non {format_expr(cur_side)}. "
            f"Controlla il {term_label(d, var)}: {f_str} va moltiplicato per OGNI termine "
            f"dentro la parentesi.")


def diagnose_multiply_step(prev_lhs, prev_rhs, cur_lhs, cur_rhs, var, k):
    """Diagnosi per un passaggio che moltiplica/divide entrambi i membri per
    una costante k (k già noto da check_relation, dal coefficiente di grado
    più alto). Confronta OGNI termine di OGNI lato contro k*prev — un
    confronto moltiplicativo, non additivo come fa diagnose_step — molto più
    mirato per questo tipo di passaggio (es. dividere per -1 e sbagliare il
    segno di un solo termine). None se non si trova un numero ridotto di
    mismatch puntuali: se anche con questo confronto le cose non tornano in
    troppi punti, non è plausibile che l'operazione fosse davvero "molt. per
    k ovunque tranne qui", meglio lasciare il fallback esistente."""
    pL = poly_coeffs(prev_lhs, var); pR = poly_coeffs(prev_rhs, var)
    cL = poly_coeffs(cur_lhs, var); cR = poly_coeffs(cur_rhs, var)
    if None in (pL, pR, cL, cR):
        return None
    mism = []
    for side_label, p, c in (("sinistra", pL, cL), ("destra", pR, cR)):
        if all(simplify(p[d] - c[d]) == 0 for d in (0, 1, 2)):
            continue  # lato mai toccato: non ha senso pretendere che sia stato moltiplicato per k
        for d in (0, 1, 2):
            if simplify(c[d] - k * p[d]) != 0:
                mism.append((side_label, d, p[d], c[d]))
    if not mism or len(mism) > 2:
        return None
    # k = -1 e' il caso "ho cambiato tutti i segni": chiamarlo
    # "moltiplicando per -1" e' corretto ma non e' come lo pensa lo studente.
    operazione = "cambiando tutti i segni" if k == -1 else f"moltiplicando per {fmt_num(k)}"
    parts = []
    for side_label, d, pval, cval in mism:
        expected_val = simplify(k * pval)
        parts.append(f"nel {term_label(d, var)} a {side_label} {operazione} "
                      f"ci si aspetta {fmt_term(expected_val, d, var)}, tu hai scritto {fmt_term(cval, d, var)}")
    text = '; '.join(parts) + '.'
    return text[0].upper() + text[1:]


def _rewrite_message(label, prev_side, cur_side):
    """Ultimo messaggio disponibile quando si sa che UN membro è stato
    riscritto male ma non quale operazione lo studente stesse tentando:
    mostra il prima e il dopo e si ferma lì. Deliberatamente NON parla di
    raccoglimento o distribuzione: quando l'operazione è riconosciuta ci
    pensano le funzioni dedicate, e appiccicare qui un consiglio sul
    'raccogliere' finiva per darlo anche a chi stava facendo tutt'altro."""
    return (f"Hai riscritto il membro {label} in modo scorretto: prima era "
            f"{format_expr(prev_side)}, ora è {format_expr(cur_side)} — non sono la stessa "
            f"espressione. Ricontrolla i segni e i numeri di questo membro, uno per uno.")


def _factoring_mismatch_diagnosis(prev_side, cur_side, var, label, raw_cur_side=None):
    """Il caso speculare di _distribution_mismatch_diagnosis: la parentesi
    sta nella riga NUOVA perché lo studente ha raccolto un fattore comune,
    e il fattore raccolto è legittimo (divide davvero tutti i termini) ma
    quello che ha lasciato dentro la parentesi è sbagliato.

    Qui si può dire esattamente cosa doveva restarci dentro — che è
    l'informazione utile — invece del generico 'non sono la stessa
    espressione'. Se invece il fattore NON divide tutti i termini l'errore è
    un altro (raccoglimento impossibile) e lo spiega _bad_factor_diagnosis."""
    source = raw_cur_side if raw_cur_side is not None else cur_side
    parts = _factor_and_paren(source)
    if parts is None:
        return None
    F, paren = parts
    if F == 0:
        return None
    atteso = cancel(prev_side / F)
    if poly_coeffs(atteso, var) is None:
        return None  # il fattore non divide in modo pulito: non e' questo il caso
    if simplify(cancel(paren - atteso)) == 0:
        return None  # la parentesi e' giusta: l'errore, se c'e', sta altrove
    if simplify(cancel(expand(F * paren) - expand(cur_side))) != 0:
        return None  # struttura letta non coerente con la riga scritta
    f_str = format_expr_raw(F)
    return (f"Raccogliendo {f_str} nel membro {label} hai sbagliato quello che resta dentro la "
            f"parentesi: {format_expr_raw(prev_side)} diviso {f_str} fa {format_expr(atteso)}, "
            f"quindi va scritto {f_str}({format_expr(atteso)}), non {f_str}({format_expr_raw(paren)}).")


def diagnose_step(prev_lhs, prev_rhs, cur_lhs, cur_rhs, var,
                  raw_prev_lhs=None, raw_prev_rhs=None,
                  raw_cur_lhs=None, raw_cur_rhs=None):
    """Per-side (left vs right) coefficient comparison — same style as the
    JS diagnosis, but exact/symbolic instead of numeric-fitted."""
    pL = poly_coeffs(prev_lhs, var); pR = poly_coeffs(prev_rhs, var)
    cL = poly_coeffs(cur_lhs, var); cR = poly_coeffs(cur_rhs, var)
    if None in (pL, pR, cL, cR):
        return "Il passaggio è troppo complesso per essere analizzato con precisione automaticamente: ricontrolla con calma ogni numero e segno."

    relevant = [d for d in (0, 1, 2) if any(simplify(x[d]) != 0 for x in (pL, pR, cL, cR))]
    mism = [d for d in relevant if simplify((cL[d] - pL[d]) - (cR[d] - pR[d])) != 0]

    if len(mism) == 0:
        return None  # shouldn't happen if caller already knows it's an error
    if len(mism) >= 3 or (len(relevant) > 1 and len(mism) >= len(relevant)):
        return "Qui non torna quasi nulla: ricontrolla questo passaggio con calma, sembra che più cose insieme non quadrino."

    # Is every mismatch confined to ONE side, while the OTHER side never has
    # any content at all (not just "unchanged at this degree", but plainly
    # always 0 everywhere)? That's not "you forgot to do it on both sides" —
    # there's nothing on the other side to do it to. It's a bad rewrite
    # WITHIN one side (wrong distribution/factoring), and needs a completely
    # different message: compare the whole side before/after, not term by term.
    right_is_trivial = not side_has_content(pR, cR)
    left_is_trivial = not side_has_content(pL, cL)
    # "L'altro lato non serve toccarlo" vale anche quando resta semplicemente
    # invariato (non solo quando è sempre zero) — ma solo per un mismatch a
    # un grado solo: con 2+ gradi coinvolti il confronto termine-per-termine
    # qui sotto è già informativo di suo, non serve allargare la condizione.
    right_unchanged_all = all(simplify(pR[d] - cR[d]) == 0 for d in (0, 1, 2))
    left_unchanged_all = all(simplify(pL[d] - cL[d]) == 0 for d in (0, 1, 2))
    mism_all_on_left = all(simplify(pR[d] - cR[d]) == 0 for d in mism)
    mism_all_on_right = all(simplify(pL[d] - cL[d]) == 0 for d in mism)

    if mism_all_on_left and (right_is_trivial or (right_unchanged_all and len(mism) == 1)):
        bad_factor = _bad_factor_diagnosis(prev_lhs, prev_rhs, cur_lhs, cur_rhs, var)
        if bad_factor is not None:
            return bad_factor
        dist_msg = _distribution_mismatch_diagnosis(prev_lhs, cur_lhs, var, "sinistro", raw_prev_lhs)
        if dist_msg is not None:
            return dist_msg
        fact_msg = _factoring_mismatch_diagnosis(prev_lhs, cur_lhs, var, "sinistro", raw_cur_lhs)
        if fact_msg is not None:
            return fact_msg
        return _rewrite_message("sinistro", prev_lhs, cur_lhs)
    if mism_all_on_right and (left_is_trivial or (left_unchanged_all and len(mism) == 1)):
        bad_factor = _bad_factor_diagnosis(prev_lhs, prev_rhs, cur_lhs, cur_rhs, var)
        if bad_factor is not None:
            return bad_factor
        dist_msg = _distribution_mismatch_diagnosis(prev_rhs, cur_rhs, var, "destro", raw_prev_rhs)
        if dist_msg is not None:
            return dist_msg
        fact_msg = _factoring_mismatch_diagnosis(prev_rhs, cur_rhs, var, "destro", raw_cur_rhs)
        if fact_msg is not None:
            return fact_msg
        return _rewrite_message("destro", prev_rhs, cur_rhs)

    # Trasporto senza cambio di segno: il termine sparisce da un lato e
    # ricompare dall'altro IDENTICO invece che col segno invertito. E' uno
    # degli errori piu' comuni in assoluto e merita di essere chiamato col
    # suo nome, invece di finire nel confronto generico termine-per-termine
    # (che direbbe solo "i due lati devono cambiare allo stesso modo",
    # lasciando allo studente il lavoro di capire che si trattava del segno).
    for d in sorted(mism, reverse=True):
        moved_from_right = (simplify(cR[d]) == 0 and simplify(pR[d]) != 0
                            and simplify((cL[d] - pL[d]) - pR[d]) == 0)
        moved_from_left = (simplify(cL[d]) == 0 and simplify(pL[d]) != 0
                           and simplify((cR[d] - pR[d]) - pL[d]) == 0)
        if moved_from_right or moved_from_left:
            origin, dest = ("destra", "sinistra") if moved_from_right else ("sinistra", "destra")
            moved_val = pR[d] if moved_from_right else pL[d]
            written = fmt_term(moved_val, d, var)
            expected = fmt_term(simplify(-moved_val), d, var)
            return (f"Hai spostato il {term_label(d, var)} da {origin} a {dest} "
                    f"senza cambiargli segno: passando dall'altra parte dell'uguale "
                    f"{written} deve diventare {expected}.")

    parts = []
    for d in sorted(mism, reverse=True)[:2]:
        label = term_label(d, var)
        left_unchanged = simplify(pL[d] - cL[d]) == 0
        right_unchanged = simplify(pR[d] - cR[d]) == 0
        if left_unchanged and not right_unchanged:
            parts.append(f"nel {label} a sinistra sei rimasto a {fmt_term(pL[d], d, var)} (invariato), ma a destra sei passato da {fmt_term(pR[d], d, var)} a {fmt_term(cR[d], d, var)}: l'operazione va applicata su entrambi i lati")
        elif right_unchanged and not left_unchanged:
            parts.append(f"nel {label} a destra sei rimasto a {fmt_term(pR[d], d, var)} (invariato), ma a sinistra sei passato da {fmt_term(pL[d], d, var)} a {fmt_term(cL[d], d, var)}: l'operazione va applicata su entrambi i lati")
        else:
            parts.append(f"nel {label} a sinistra sei passato da {fmt_term(pL[d], d, var)} a {fmt_term(cL[d], d, var)}, a destra invece da {fmt_term(pR[d], d, var)} a {fmt_term(cR[d], d, var)}: i due lati devono cambiare allo stesso modo")
    text = ' ; '.join(parts) + '.'
    return text[0].upper() + text[1:]


FRACTION_IMPLICATION_NOTE = (
    "Hai moltiplicato per un'espressione che contiene la incognita: da qui in "
    "poi potrebbero comparire soluzioni che non vanno bene nell'equazione di "
    "partenza — verrà controllato alla fine."
)


def diagnose_fraction_step(prev_lhs, prev_rhs, expected_lhs, expected_rhs, cur_lhs, cur_rhs, var):
    """Diagnosi per un errore nella cancellazione di un denominatore: una
    frase-ponte mostra la riga originale con la frazione (nella notazione
    dello studente) e cosa ci si aspetta dopo averla cancellata
    correttamente, poi riusa il confronto coefficiente-per-coefficiente di
    diagnose_step tra la forma attesa e quella scritta dallo studente."""
    prev_s = f"{format_expr_raw(prev_lhs)} = {format_expr_raw(prev_rhs)}"
    expected_s = f"{format_expr(expected_lhs)} = {format_expr(expected_rhs)}"
    detail = diagnose_step(expected_lhs, expected_rhs, cur_lhs, cur_rhs, var)
    bridge = f"Eliminando il denominatore da {prev_s} ci si aspetta {expected_s}. "
    return bridge + (detail or "")


def split_fraction_poly(expr, var):
    """Separa expr in (parte con denominatori che dipendono da var, resto
    polinomiale). Serve per i passaggi in cui la frazione viene lasciata
    intatta e si lavora solo sui termini polinomiali (tipico: portare tutto
    a primo membro): li' il confronto giusto e' fra le sole parti
    polinomiali, mentre trattare la riga come 'un'unica frazione da
    ricombinare' non spiega niente allo studente."""
    terms = expr.args if expr.is_Add else (expr,)
    frac = S(0)
    poly = S(0)
    for t in terms:
        if var in side_denominator(t, var).free_symbols:
            frac += t
        else:
            poly += t
    return frac, poly


def check_fraction_transport(prev_lhs, prev_rhs, cur_lhs, cur_rhs, var):
    """Passaggio fra due righe con frazioni in cui la PARTE FRAZIONARIA e'
    rimasta identica (nessuna ricombinazione, nessuna moltiplicazione) e a
    cambiare sono solo i termini polinomiali: allora l'errore, se c'e', sta
    tutto li' e va diagnosticato confrontando le sole parti polinomiali.

    Perche' e' matematicamente lecito concludere 'errore' qui: se le parti
    frazionarie si annullano nella differenza e resta un polinomio P != 0,
    allora cur_diff = prev_diff + P. Un passaggio valido richiederebbe
    cur_diff = k*prev_diff; ma (k-1)*prev_diff = P con prev_diff che ha
    davvero var al denominatore e P polinomiale forza k = 1 e quindi P = 0.
    Quindi P != 0 implica passaggio sbagliato, senza rischio di falso
    allarme. Ritorna None se il pattern non si applica."""
    fp_l, pp_l = split_fraction_poly(prev_lhs, var)
    fp_r, pp_r = split_fraction_poly(prev_rhs, var)
    fc_l, pc_l = split_fraction_poly(cur_lhs, var)
    fc_r, pc_r = split_fraction_poly(cur_rhs, var)

    prev_frac_diff = cancel(fp_l - fp_r)
    cur_frac_diff = cancel(fc_l - fc_r)
    if prev_frac_diff == 0 or simplify(cancel(cur_frac_diff - prev_frac_diff)) != 0:
        return None  # la parte frazionaria e' cambiata: non e' questo il caso

    residuo = cancel((pc_l - pc_r) - (pp_l - pp_r))
    if simplify(residuo) == 0:
        return {'supported': True, 'equivalent': True, 'relation_kind': 'equivalence'}
    if poly_coeffs(residuo, var) is None:
        return None  # residuo non polinomiale/di grado troppo alto: non concludere
    diag = diagnose_step(pp_l, pp_r, pc_l, pc_r, var)
    if diag is None:
        return None
    return {'supported': True, 'equivalent': False, 'relation_kind': 'equivalence',
            'combine_diagnosis': diag}


def check_partial_denominator_clearing(prev_lhs, prev_rhs, cur_lhs, cur_rhs, var):
    """Errore classico nell'eliminare un denominatore: lo studente
    moltiplica per il denominatore SOLO la frazione (che diventa il suo
    numeratore) e lascia gli altri termini come stavano, invece di
    moltiplicare tutto.

    Riconosciuto confrontando la riga scritta con esattamente quel gesto
    sbagliato (numeratore + resto invariato): se combacia, la causa è certa
    e si può nominare. Vale anche quando il risultato CORRETTO sarebbe di
    grado troppo alto per questo motore — ed è il caso che conta, perché lì
    ogni altra strada si fermerebbe a 'troppo complesso' pur essendo un
    errore semplicissimo da spiegare. None se non combacia."""
    prev_diff = prev_lhs - prev_rhs
    frac, poly = split_fraction_poly(prev_diff, var)
    if frac == 0 or simplify(poly) == 0:
        return None
    numer, denom = together(frac).as_numer_denom()
    if var not in denom.free_symbols:
        return None
    cur_diff = cancel(cur_lhs - cur_rhs)
    if simplify(cancel(cur_diff - (numer + poly))) != 0:
        return None
    denom_s = format_expr_raw(denom)
    poly_s = format_expr_raw(poly)
    diag = (f"Per eliminare il denominatore hai moltiplicato per ({denom_s}) solo la frazione: "
            f"va moltiplicato per ({denom_s}) OGNI termine dei due membri. "
            f"{poly_s} è rimasto com'era, invece di diventare ({poly_s})({denom_s}).")
    return {'supported': True, 'equivalent': False, 'relation_kind': 'implication',
            'combine_diagnosis': diag}


def _is_fraction_sum(side, var):
    """True se side è una somma di 2+ termini, almeno due dei quali hanno
    individualmente un denominatore che dipende da var — cioè c'era
    davvero più di una frazione da combinare. Se il lato è UNA sola
    frazione (o una frazione sola più roba senza frazioni), 'combinare le
    frazioni' non è l'operazione plausibile: l'errore è altrove (tipico
    caso: un trasporto che lascia la frazione intatta e sbaglia il segno
    di un altro termine) e va lasciato al fallback onesto invece di
    inventare una spiegazione sul mcm che non c'entra."""
    terms = side.args if side.is_Add else (side,)
    frac_terms = [t for t in terms if var in side_denominator(t, var).free_symbols]
    return len(frac_terms) >= 2


def diagnose_combine_step(prev_lhs, prev_rhs, expected_lhs, expected_rhs, cur_lhs, cur_rhs, var):
    """Diagnosi per un errore nel raccogliere più frazioni in una sola sullo
    STESSO lato (mcm sbagliato, numeratore calcolato male), senza che
    l'equazione sia stata moltiplicata per nulla — quindi expected_lhs/rhs
    sono semplicemente prev ricombinato (together), non una forma
    polinomiale: il confronto è per cancellazione dell'intera frazione, non
    coefficiente per coefficiente (diagnose_step non si applica, richiede
    lati polinomiali). Usa format_expr_raw ovunque (mai format_expr, che
    farebbe expand e romperebbe la frazione unica in una somma di frazioni,
    non più fedele a quanto scritto)."""
    mism_left = simplify(cancel(cur_lhs - expected_lhs)) != 0
    mism_right = simplify(cancel(cur_rhs - expected_rhs)) != 0
    if mism_left and not _is_fraction_sum(prev_lhs, var):
        mism_left = False
    if mism_right and not _is_fraction_sum(prev_rhs, var):
        mism_right = False
    if not mism_left and not mism_right:
        return None

    def side_msg(label, prev_s, expected_s, cur_s):
        return (f"hai combinato le frazioni {label} in modo scorretto: partendo da {prev_s} "
                f"ci si aspetta {expected_s}, tu hai scritto {cur_s}. Controlla il minimo comune "
                f"denominatore: forse manca un fattore, o il numeratore non è stato calcolato bene")

    parts = []
    if mism_left:
        parts.append(side_msg("a sinistra", format_expr_raw(prev_lhs), format_expr_raw(expected_lhs), format_expr_raw(cur_lhs)))
    if mism_right:
        parts.append(side_msg("a destra", format_expr_raw(prev_rhs), format_expr_raw(expected_rhs), format_expr_raw(cur_rhs)))
    text = '; '.join(parts) + '.'
    return text[0].upper() + text[1:]


def solve_original(lhs, rhs, var):
    """Solves the ORIGINAL equation (first row) for the true solution set,
    used by the final check. Real solutions only, matching the scope of
    these grades (no complex roots expected/taught here)."""
    from sympy import solveset
    result = solveset(Eq(lhs, rhs), var, domain=S.Reals)
    if result == S.EmptySet and simplify(cancel(lhs - rhs)) == 0:
        # equazione razionale che è in realtà un'identità (vera per ogni x
        # tranne i poli): solveset può collassare erroneamente a EmptySet
        # invece di Reals meno i poli — non fidarsi, dichiarare 'unknown'.
        return None
    return result


_SPLIT_RE = re.compile(r'\bor\b|\bo\b|∨|;|,|&')
_ARROW_RE = re.compile(r'=>|⇒|∴')


def strip_reasoning_prefix(text):
    """Se la riga contiene una freccia di implicazione (es. 'discriminante
    negativo => impossibile', o '-2 ± sqrt(9-4) => x = -2'), tiene solo la
    conclusione dopo l'ULTIMA freccia, scartando i conti/il ragionamento
    prima — altrimenti quel testo aggiunge un altro '=' che rompe il parsing
    (split_equation/parse_declared_solutions contano gli '=' per capire cosa
    stanno leggendo) e la dichiarazione finale non viene capita per niente."""
    parts = _ARROW_RE.split(text)
    return parts[-1].strip() if len(parts) > 1 else text


def parse_declared_solutions(text, var):
    """Parses a student's final-answer line into a list of SymPy values.
    Accepts 'x=3', 'x=2 or x=3', 'x=2, x=3', 'x=2 & x=3', etc."""
    text = strip_reasoning_prefix(text)
    parts = _SPLIT_RE.split(text)
    values = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        rhs_txt = part.split('=', 1)[1] if '=' in part else part
        try:
            values.append(simplify(safe_parse(rhs_txt)))
        except Exception:
            continue
    return values


_EMPTY_PHRASES = {
    'impossibile', 'nessunasoluzione', 'nessunesoluzioni', 'nonesistesoluzione',
    'nonesistonosoluzioni', 'insiemevuoto', 's=∅', '∅', 'nessuna', 'nosolution',
    'impossible', 's={}', 's=vuoto', 'equazioneimpossibile',
}


def is_declared_empty(text):
    """Riconosce risposte come 'impossibile', 'nessuna soluzione', 'S = ∅', ecc.
    come dichiarazione esplicita che l'equazione non ha soluzioni — da NON
    confondere con un testo che semplicemente non si riesce a interpretare."""
    text = strip_reasoning_prefix(text)
    t = text.strip().lower().replace(' ', '').rstrip('.')
    return t in _EMPTY_PHRASES


def compare_solutions(true_set, declared_values, declared_is_empty_claim=False):
    if true_set == S.EmptySet:
        true_vals = []
    elif isinstance(true_set, FiniteSet):
        true_vals = list(true_set)
    else:
        return {'status': 'unknown'}

    if declared_is_empty_claim:
        if len(true_vals) == 0:
            return {'status': 'ok', 'correct': []}
        return {'status': 'wrongly_claimed_impossible', 'missing': true_vals, 'correct': true_vals}

    matched = set()
    invalid = []
    for d in declared_values:
        found = False
        for i, t in enumerate(true_vals):
            if i in matched:
                continue
            if simplify(d - t) == 0:
                matched.add(i)
                found = True
                break
        if not found:
            invalid.append(d)
    missing = [t for i, t in enumerate(true_vals) if i not in matched]
    if invalid:
        return {'status': 'invalid_solution_present', 'invalid': invalid, 'missing': missing, 'correct': true_vals}
    if missing:
        return {'status': 'incomplete', 'missing': missing, 'correct': true_vals}
    return {'status': 'ok', 'correct': true_vals}


def looks_like_solution_row(text, var):
    """True se questa riga sembra la DICHIARAZIONE DELLA RISPOSTA FINALE e
    non un passaggio dell'esercizio. Usata solo per l'ultima riga scritta, e
    solo quando quella riga non e' interpretabile come equazione singola:
    tipicamente perche' contiene piu' di un '=' ('x = 0 & x = -1') o una
    freccia di conclusione ('... => x = -2', '... => IMPOSSIBILE').

    Senza questo controllo righe del genere venivano marcate 'unreadable'
    con un '?' rosso e la nota "non sono riuscito a interpretare questa riga
    come equazione": per lo studente sembra un errore suo, mentre la riga e'
    perfettamente sensata — semplicemente non e' un passaggio. Il verdetto
    su quella riga lo da' gia' il controllo finale.

    Volutamente restrittivo: pretende che i valori dichiarati siano NUMERI.
    Cosi' una riga che non c'entra (es. una disequazione, che questo motore
    non sa ancora gestire) non viene silenziosamente nascosta come se fosse
    una risposta finale, ma resta segnalata come non interpretata."""
    if is_declared_empty(text):
        return True
    try:
        values = parse_declared_solutions(text, var)
    except Exception:
        return False
    return bool(values) and all(getattr(v, 'is_number', False) for v in values)


# Ruoli che questa prima versione sa gestire per il confronto passo-passo.
# Gli altri (domain, substitution, side_calc, case_split) sono riconosciuti
# dal contratto ma non ancora implementati: lo diciamo onestamente invece
# di ignorarli o fingere di capirli.
SUPPORTED_STEP_ROLES = {"equation", None, "equazione", ""}


def process_sheet(rows, variable_hint=None):
    """Framework-independent request handler: rows is a list of plain dicts
    with keys index/plain/role_hint (latex is accepted but unused server-side).
    Returns a plain dict matching the steps/final_check contract exactly —
    no FastAPI/Pydantic dependency, so this can be unit-tested directly."""
    steps = []
    prev_parsed = None
    first_equation = None
    solution_row = None

    # Indice (nella lista rows) dell'ultima riga con del testo: e' l'unica
    # che ha senso interpretare come dichiarazione della risposta finale.
    last_written = None
    for i, row in enumerate(rows):
        if (row.get('plain') or '').strip():
            last_written = i

    for row_position, row in enumerate(rows):
        idx = row.get('index')
        plain = row.get('plain', '') or ''
        role = (row.get('role_hint') or 'equation').lower()

        if role in ('solution', 'soluzione'):
            solution_row = row
            steps.append({'index': idx, 'status': 'skip', 'relation': 'solution'})
            continue

        if role not in SUPPORTED_STEP_ROLES:
            steps.append({
                'index': idx, 'status': 'unreadable', 'relation': role,
                'note': f"Le righe di tipo '{role}' non sono ancora supportate in questa versione: "
                        f"per ora vengono ignorate nel controllo, non contano né come errore né come ok."
            })
            continue

        if not plain.strip():
            steps.append({'index': idx, 'status': 'skip'})
            continue

        # Disequazione: riga scritta benissimo, semplicemente fuori da quello
        # che questo motore sa ancora fare. Va detto con parole sue, non con
        # un messaggio sul numero di '=' trovati, che sembrerebbe un errore
        # dello studente quando non lo e'.
        # Le frecce di conclusione ('=>', '⇒') vanno tolte PRIMA di cercare i
        # segni di disuguaglianza: contengono un '>' e farebbero scambiare
        # per disequazione una normale riga di risposta finale.
        plain_senza_frecce = _ARROW_RE.sub(' ', plain)
        if any(sign in plain_senza_frecce for sign in ('<', '>', '≤', '≥', '⩽', '⩾')):
            steps.append({
                'index': idx, 'status': 'unreadable', 'relation': None,
                'note': "Questa è una disequazione: per ora il motore controlla solo le equazioni "
                        "(le disequazioni richiedono di seguire anche il verso, che non è ancora gestito). "
                        "La riga non viene controllata: non conta né come errore né come ok."
            })
            continue

        try:
            lhs, rhs = split_equation(plain)
            var = detect_variable(lhs, rhs, hint=variable_hint)
        except Exception as e:
            # L'ultima riga scritta puo' benissimo non essere un'equazione:
            # e' la risposta finale ("x = 0 & x = -1", "... => IMPOSSIBILE").
            # In quel caso non e' un errore dello studente e non va marcata
            # come illeggibile: la valuta il controllo finale.
            hint_var = first_equation[2] if first_equation else Symbol(variable_hint or 'x')
            if row_position == last_written and looks_like_solution_row(plain, hint_var):
                solution_row = row
                steps.append({'index': idx, 'status': 'skip', 'relation': 'solution'})
                continue
            steps.append({'index': idx, 'status': 'unreadable',
                          'note': f"Non sono riuscito a interpretare questa riga come equazione: {e}"})
            continue

        if first_equation is None:
            first_equation = (lhs, rhs, var)

        # Versione non valutata della riga: serve solo alle diagnosi
        # strutturali (vedi safe_parse_raw). Se il parsing grezzo fallisce si
        # prosegue senza: e' un di piu' per spiegare meglio, non un requisito.
        try:
            raw_lhs, raw_rhs = split_equation(plain, raw=True)
        except Exception:
            raw_lhs, raw_rhs = None, None

        if prev_parsed is None:
            steps.append({'index': idx, 'status': 'first'})
            prev_parsed = (lhs, rhs, var, raw_lhs, raw_rhs)
            continue

        p_lhs, p_rhs, _p_var, p_raw_lhs, p_raw_rhs = prev_parsed
        rel = check_relation(p_lhs, p_rhs, lhs, rhs, var)
        relation_label = 'equivalence' if rel['supported'] else None

        if not rel['supported'] and (row_has_fraction(p_lhs, p_rhs, var) or row_has_fraction(lhs, rhs, var)):
            rel = check_fraction_relation(p_lhs, p_rhs, lhs, rhs, var)
            relation_label = rel.get('relation_kind')

        if not rel['supported']:
            bad_factor_diag = _bad_factor_diagnosis(p_lhs, p_rhs, lhs, rhs, var, raw_lhs, raw_rhs)
            if bad_factor_diag is not None:
                steps.append({'index': idx, 'status': 'error', 'relation': None, 'diagnosis': bad_factor_diag})
            else:
                steps.append({
                    'index': idx, 'status': 'unreadable', 'relation': None,
                    'note': "Questo passaggio è troppo complesso per questa versione del motore "
                            "(per ora gestisce solo equazioni di 1°/2° grado, anche con frazioni algebriche, "
                            "in una sola variabile, senza parametri)."
                })
        elif rel['equivalent']:
            note = FRACTION_IMPLICATION_NOTE if relation_label == 'implication' else None
            steps.append({'index': idx, 'status': 'ok', 'relation': relation_label, 'note': note})
        else:
            # Ordine deliberato: dalla diagnosi piu' specifica (che nomina
            # l'operazione che lo studente stava facendo) alla piu' generica.
            # 1. Raccoglimento impossibile: la struttura scritta dice da sola
            #    cosa stava provando a fare, ed e' sbagliato a monte.
            diag = _bad_factor_diagnosis(p_lhs, p_rhs, lhs, rhs, var, raw_lhs, raw_rhs)
            # 2. Diagnosi gia' prodotte dal classificatore delle frazioni.
            if diag is None and 'combine_diagnosis' in rel:
                diag = rel['combine_diagnosis']
            if diag is None and relation_label == 'implication' and 'expected_lhs' in rel:
                diag = diagnose_fraction_step(p_lhs, p_rhs, rel['expected_lhs'], rel['expected_rhs'], lhs, rhs, var)
            # 3. Moltiplicazione/divisione per una costante: confronto
            #    moltiplicativo mirato invece di quello additivo generico.
            if diag is None:
                k = rel.get('k')
                if k is not None and k != 1:
                    diag = diagnose_multiply_step(p_lhs, p_rhs, lhs, rhs, var, k)
            # 4. Confronto generico termine per termine.
            if diag is None:
                diag = diagnose_step(p_lhs, p_rhs, lhs, rhs, var,
                                     p_raw_lhs, p_raw_rhs, raw_lhs, raw_rhs)
            steps.append({'index': idx, 'status': 'error', 'relation': relation_label, 'diagnosis': diag})

        prev_parsed = (lhs, rhs, var, raw_lhs, raw_rhs)

    # ---------- Controllo finale ----------
    if solution_row is None and rows:
        solution_row = rows[-1]

    if first_equation is None or solution_row is None or not (solution_row.get('plain') or '').strip():
        final = {'status': 'no_final_solution',
                 'message': 'Non ho trovato una riga di soluzione finale da controllare.'}
    else:
        f_lhs, f_rhs, f_var = first_equation
        try:
            true_set = solve_original(f_lhs, f_rhs, f_var)
            if true_set is None:
                cmp = {'status': 'unknown'}
            else:
                plain_text = solution_row['plain']
                if is_declared_empty(plain_text):
                    cmp = compare_solutions(true_set, [], declared_is_empty_claim=True)
                else:
                    declared = parse_declared_solutions(plain_text, f_var)
                    if not declared and plain_text.strip():
                        # Testo presente ma non interpretato come nessun
                        # valore: NON trattarlo come "nessuna soluzione
                        # dichiarata", altrimenti confrontare vuoto-contro-
                        # vuoto (quando anche l'equazione vera non ha
                        # soluzioni) darebbe un falso "tutto corretto" anche
                        # se lo studente ha scritto una risposta sbagliata
                        # che non siamo riusciti a leggere.
                        cmp = {'status': 'unknown'}
                    else:
                        cmp = compare_solutions(true_set, declared)
        except Exception:
            cmp = {'status': 'unknown'}

        if cmp['status'] == 'ok':
            final = {'status': 'ok', 'message': 'Hai trovato tutte le soluzioni corrette.'}
        elif cmp['status'] == 'wrongly_claimed_impossible':
            correct = [str(v) for v in cmp['correct']]
            final = {
                'status': 'incomplete',
                'message': "In realtà l'equazione ha soluzione: hai detto che era impossibile, ma non lo è. "
                           f"{'La soluzione corretta è' if len(correct) == 1 else 'Le soluzioni corrette sono'}: {', '.join(correct)}.",
                'correct_solutions': correct,
            }
        elif cmp['status'] == 'incomplete':
            missing = [str(v) for v in cmp['missing']]
            final = {
                'status': 'incomplete',
                'message': "Le soluzioni che hai scritto sono corrette, ma non sono tutte: "
                           f"ti manca {'la soluzione' if len(missing) == 1 else 'le soluzioni'} {', '.join(missing)}.",
                'correct_solutions': [str(v) for v in cmp['correct']],
            }
        elif cmp['status'] == 'invalid_solution_present':
            invalid = [str(v) for v in cmp['invalid']]
            if not cmp['correct']:
                final = {
                    'status': 'invalid_solution_present',
                    'message': f"{'Il valore' if len(invalid) == 1 else 'I valori'} {', '.join(invalid)} "
                               f"che hai scritto non {'è una soluzione valida' if len(invalid) == 1 else 'sono soluzioni valide'}: "
                               "questa equazione in realtà non ha soluzioni reali (è impossibile).",
                    'correct_solutions': [],
                }
            else:
                final = {
                    'status': 'invalid_solution_present',
                    'message': f"{'La soluzione' if len(invalid) == 1 else 'Le soluzioni'} {', '.join(invalid)} "
                               f"non {'è valida' if len(invalid) == 1 else 'sono valide'} per l'equazione di partenza. "
                               f"Le soluzioni corrette sono: {', '.join(str(v) for v in cmp['correct'])}.",
                    'correct_solutions': [str(v) for v in cmp['correct']],
                }
        else:
            final = {
                'status': 'unknown',
                'message': "Non sono riuscito a determinare con certezza la soluzione di questo esercizio "
                           "(potrebbe essere troppo complesso per questa versione del motore)."
            }

    return {'steps': steps, 'final_check': final}
