"""Synthetic people and every surface form of their identifiers. Everything here is drawn from a seeded RNG."""
from __future__ import annotations

import random
import string
import unicodedata
from dataclasses import dataclass, field

from veilbench import ids
from veilbench.corpus import names as N

DOMAINS = ["example.com", "example.org", "mail.example", "corp.example", "inbox.test", "post.example.net"]
TITLES_EN = ["Mr.", "Ms.", "Mrs.", "Dr."]
TITLES_ES = ["Sr.", "Sra.", "Dr.", "Dra."]
TITLES_TL = ["Ginoong", "Binibining", "G.", "Bb."]


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


@dataclass
class Person:
    culture: str
    given: list[str]          # given name tokens as displayed
    family: list[str]         # family name tokens as displayed
    country: str              # PH, ES, US, SG
    email_local: str
    domain: str
    phone_digits: str         # national significant digits with country code, no plus
    phone_cc: str
    address: str
    address_country: str
    national_id: str
    national_id_kind: str
    card: str
    iban: str
    dob: tuple[int, int, int]
    ip: str
    uid: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def full(self) -> str:
        return " ".join(self.given + self.family)

    @property
    def first(self) -> str:
        return self.given[0]

    @property
    def last(self) -> str:
        return " ".join(self.family)

    @property
    def email(self) -> str:
        return f"{self.email_local}@{self.domain}"


def _pick_name(rng: random.Random, culture: str) -> tuple[list[str], list[str]]:
    pool = N.POOLS[culture]
    accents = pool.get("accented", {})

    def disp(tok: str) -> str:
        return accents[tok] if tok in accents and rng.random() < 0.6 else tok

    if culture == "ph":
        if rng.random() < 0.18:
            given = rng.choice(pool["compound"]).split()
        else:
            given = [rng.choice(pool["first"])]
            if rng.random() < 0.2:
                given.append(rng.choice(pool["first"]))
        family = rng.choice(pool["particled"]).split() if rng.random() < 0.16 else [rng.choice(pool["last"])]
        return given, family
    if culture == "es":
        given = [disp(rng.choice(pool["first"]))]
        if rng.random() < 0.3:
            given.append(disp(rng.choice(pool["first"])))
        family = [disp(rng.choice(pool["last"]))]
        if rng.random() < 0.7:
            family.append(disp(rng.choice(pool["last"])))
        return given, family
    if culture in ("zh", "vn"):
        # family name first in the native order, but displayed Western order in our records
        return [rng.choice(pool["first"])], [rng.choice(pool["last"])]
    given = [disp(rng.choice(pool["first"]))]
    if culture == "en" and rng.random() < 0.12:
        given.append(rng.choice(string.ascii_uppercase) + ".")
    return given, [disp(rng.choice(pool["last"]))]


def _email_local(rng: random.Random, given: list[str], family: list[str]) -> str:
    g = strip_accents(given[0]).lower().replace(".", "")
    f = strip_accents("".join(family)).lower().replace(" ", "").replace(".", "")
    style = rng.choice(["dot", "dot", "underscore", "initial", "concat", "reverse"])
    local = {"dot": f"{g}.{f}", "underscore": f"{g}_{f}", "initial": f"{g[0]}{f}", "concat": f"{g}{f}", "reverse": f"{f}.{g}"}[style]
    if rng.random() < 0.3:
        local += str(rng.randint(1, 99))
    return local


def _address(rng: random.Random, country: str) -> str:
    if country == "PH":
        city, zipc = rng.choice(N.CITIES_PH)
        street = rng.choice(N.STREETS_PH)
        num = rng.randint(1, 980)
        form = rng.choice([0, 1, 2])
        if form == 0:
            return f"{num} {street} St., {city} {zipc}"
        if form == 1:
            return f"Blk {rng.randint(1, 30)} Lot {rng.randint(1, 40)} {street} Street, Brgy. {rng.choice(N.STREETS_PH)}, {city}"
        return f"{num} {street} Street, Barangay {rng.choice(N.STREETS_PH)}, {city}, Philippines {zipc}"
    if country == "ES":
        city, zipc = rng.choice(N.CITIES_ES)
        street = rng.choice(N.STREETS_ES)
        num = rng.randint(1, 140)
        if rng.random() < 0.5:
            return f"Calle de {street} {num}, {zipc} {city}"
        return f"C/ {street}, {num}, {rng.randint(1, 6)}º {rng.choice('ABCD')}, {zipc} {city}"
    city, st, zipc = rng.choice(N.CITIES_US)
    suffix = rng.choice(["St", "Ave", "Blvd", "Rd", "Street", "Avenue", "Lane"])
    street = rng.choice(N.STREETS_EN)
    num = rng.randint(10, 9999)
    if rng.random() < 0.3:
        return f"{num} {street} {suffix}, Apt {rng.randint(1, 40)}, {city}, {st} {zipc}"
    return f"{num} {street} {suffix}, {city}, {st} {zipc}"


def _national_id(rng: random.Random, country: str) -> tuple[str, str]:
    if country == "PH":
        kind = rng.choice(["SSS", "PHILSYS", "TIN"])
        if kind == "SSS":
            return f"{rng.randint(10, 99)}-{rng.randint(1000000, 9999999)}-{rng.randint(0, 9)}", "SSS"
        if kind == "PHILSYS":
            return "-".join(f"{rng.randint(0, 9999):04d}" for _ in range(4)), "PHILSYS"
        return f"{rng.randint(100, 999)}-{rng.randint(100, 999)}-{rng.randint(100, 999)}-{rng.randint(0, 999):03d}", "TIN"
    if country == "ES":
        n = rng.randint(10_000_000, 99_999_999)
        return f"{n}{ids.dni_letter(n)}", "DNI"
    if country == "SG":
        pre = rng.choice("ST")
        d = f"{rng.randint(0, 9999999):07d}"
        return f"{pre}{d}{ids.nric_letter(pre, d)}", "NRIC"
    while True:
        area, group, serial = f"{rng.randint(1, 899):03d}", f"{rng.randint(1, 99):02d}", f"{rng.randint(1, 9999):04d}"
        if ids.ssn_valid(area, group, serial):
            return f"{area}-{group}-{serial}", "SSN"


def _card(rng: random.Random) -> str:
    kind = rng.choice(["visa", "mc", "amex"])
    if kind == "visa":
        return ids.luhn_complete("4" + "".join(rng.choice(string.digits) for _ in range(5)), 16)
    if kind == "mc":
        return ids.luhn_complete(str(rng.randint(51, 55)) + "".join(rng.choice(string.digits) for _ in range(4)), 16)
    return ids.luhn_complete(rng.choice(["34", "37"]) + "".join(rng.choice(string.digits) for _ in range(4)), 15)


def _iban(rng: random.Random) -> str:
    c = rng.choice(["ES", "DE", "GB"])
    if c == "ES":
        return ids.iban_make("ES", "".join(rng.choice(string.digits) for _ in range(20)))
    if c == "DE":
        return ids.iban_make("DE", "".join(rng.choice(string.digits) for _ in range(18)))
    return ids.iban_make("GB", "".join(rng.choice(string.ascii_uppercase) for _ in range(4)) + "".join(rng.choice(string.digits) for _ in range(14)))


def _phone(rng: random.Random, country: str) -> tuple[str, str]:
    if country == "PH":
        return "63", "9" + "".join(rng.choice(string.digits) for _ in range(9))
    if country == "ES":
        return "34", rng.choice("67") + "".join(rng.choice(string.digits) for _ in range(8))
    if country == "SG":
        return "65", rng.choice("89") + "".join(rng.choice(string.digits) for _ in range(7))
    return "1", f"{rng.randint(201, 989)}{rng.randint(200, 999)}{rng.randint(0, 9999):04d}"


def make_person(rng: random.Random, cultures: list[str], uid: int = 0) -> Person:
    culture = rng.choice(cultures)
    given, family = _pick_name(rng, culture)
    country = {"ph": rng.choice(["PH", "PH", "PH", "SG"]), "es": rng.choice(["ES", "ES", "ES", "PH"]), "en": rng.choice(["US", "US", "US", "SG"])}.get(
        culture, rng.choice(["PH", "ES", "US", "SG"]))
    cc, digits = _phone(rng, country)
    addr_country = country if country in ("PH", "ES", "US") else "PH"
    nid, nkind = _national_id(rng, country)
    return Person(
        culture=culture, given=given, family=family, country=country,
        email_local=_email_local(rng, given, family), domain=rng.choice(DOMAINS),
        phone_digits=digits, phone_cc=cc, address=_address(rng, addr_country), address_country=addr_country,
        national_id=nid, national_id_kind=nkind, card=_card(rng), iban=_iban(rng),
        dob=(rng.randint(1950, 2005), rng.randint(1, 12), rng.randint(1, 28)),
        ip=".".join(str(rng.randint(11, 223)) for _ in range(4)), uid=uid,
    )


# ---------------------------------------------------------------- surface forms

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
DIGIT_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def phone_surface(rng: random.Random, p: Person, obfuscate: bool = False) -> tuple[str, str]:
    """(surface text, canonical digits)."""
    d, cc = p.phone_digits, p.phone_cc
    canon = d  # national significant digits: the part that survives a dropped country code
    if obfuscate:
        style = rng.choice(["spaced", "words", "spaced"])
        if style == "spaced":
            return " ".join(d), canon
        return " ".join(DIGIT_WORDS[int(c)] for c in d), canon
    if cc == "63":
        forms = [f"+63 {d[:3]} {d[3:6]} {d[6:]}", f"0{d[:3]}-{d[3:6]}-{d[6:]}", f"+63{d}", f"0{d[:3]} {d[3:6]} {d[6:]}"]
    elif cc == "34":
        forms = [f"+34 {d[:3]} {d[3:5]} {d[5:7]} {d[7:]}", f"+34{d}", f"{d[:3]} {d[3:5]} {d[5:7]} {d[7:]}"]
    elif cc == "65":
        forms = [f"+65 {d[:4]} {d[4:]}", f"+65{d}", f"{d[:4]}-{d[4:]}"]
    else:
        forms = [f"+1 ({d[:3]}) {d[3:6]}-{d[6:]}", f"({d[:3]}) {d[3:6]}-{d[6:]}", f"{d[:3]}-{d[3:6]}-{d[6:]}", f"+1 {d[:3]} {d[3:6]} {d[6:]}", f"{d[:3]}.{d[3:6]}.{d[6:]}"]
    return rng.choice(forms), canon


def email_surface(rng: random.Random, p: Person, obfuscate: bool = False) -> tuple[str, str]:
    local, dom = p.email_local, p.domain
    canon = p.email
    if obfuscate:
        style = rng.choice(["words", "bracket", "paren", "words"])
        if style == "words":
            return f"{local.replace('.', ' dot ').replace('_', ' underscore ')} at {dom.replace('.', ' dot ')}", canon
        if style == "bracket":
            return f"{local}[at]{dom.replace('.', '[dot]')}", canon
        return f"{local}(at){dom}", canon
    return canon, canon


def card_surface(rng: random.Random, p: Person, obfuscate: bool = False) -> tuple[str, str]:
    c = p.card
    if obfuscate:
        return " ".join(c), c
    if len(c) == 15:
        return rng.choice([f"{c[:4]} {c[4:10]} {c[10:]}", c]), c
    groups = [c[i:i + 4] for i in range(0, 16, 4)]
    return rng.choice([" ".join(groups), "-".join(groups), c]), c


def id_surface(rng: random.Random, p: Person, obfuscate: bool = False) -> tuple[str, str]:
    v = p.national_id
    canon = v
    if obfuscate:
        return " ".join(v.replace("-", "")), canon
    if p.national_id_kind == "DNI" and rng.random() < 0.4:
        return f"{v[:-1]}-{v[-1]}", canon
    return v, canon


def dob_surface(rng: random.Random, p: Person, lang: str = "en") -> tuple[str, str]:
    y, m, d = p.dob
    canon = f"{y:04d}-{m:02d}-{d:02d}"
    forms = [canon, f"{d:02d}/{m:02d}/{y}", f"{MONTHS[m - 1]} {d}, {y}", f"{d} {MONTHS[m - 1]} {y}", f"{m}/{d}/{y}"]
    return rng.choice(forms), canon


def person_surface(rng: random.Random, p: Person, form: str | None = None, lang: str = "en") -> str:
    form = form or rng.choice(["full", "full", "full", "first", "last_first", "title_last", "upper"])
    if form == "full":
        return p.full
    if form == "first":
        return p.first
    if form == "last_first":
        return f"{p.last}, {' '.join(p.given)}"
    if form == "upper":
        return p.full.upper()
    titles = TITLES_ES if lang == "es" else TITLES_TL if lang == "tl" else TITLES_EN
    return f"{rng.choice(titles)} {p.last}"


def _alnum(rng: random.Random, n: int, alphabet: str) -> str:
    return "".join(rng.choice(alphabet) for _ in range(n))


B62 = string.ascii_letters + string.digits
B64URL = string.ascii_letters + string.digits + "-_"


def make_secret(rng: random.Random, kind: str | None = None) -> tuple[str, str]:
    """(label, value). Values are assembled from parts at run time so no vendor shaped key is ever a literal in this repo."""
    kind = kind or rng.choice(["stripe", "aws", "github", "slack", "google", "openai", "gitlab", "sendgrid", "hf", "generic", "jwt", "jwt"])
    if kind == "stripe":
        return "API_KEY", "sk_" + "live" + "_" + _alnum(rng, 24, B62)
    if kind == "aws":
        return "API_KEY", "AK" + "IA" + _alnum(rng, 16, string.ascii_uppercase + string.digits)
    if kind == "github":
        return "API_KEY", "gh" + "p_" + _alnum(rng, 36, B62)
    if kind == "slack":
        return "API_KEY", "xo" + "xb-" + _alnum(rng, 11, string.digits) + "-" + _alnum(rng, 12, string.digits) + "-" + _alnum(rng, 24, B62)
    if kind == "google":
        return "API_KEY", "AI" + "za" + _alnum(rng, 35, B64URL)
    if kind == "openai":
        return "API_KEY", "sk-" + "proj-" + _alnum(rng, 40, B64URL)
    if kind == "gitlab":
        return "API_KEY", "gl" + "pat-" + _alnum(rng, 20, B64URL)
    if kind == "sendgrid":
        return "API_KEY", "S" + "G." + _alnum(rng, 22, B64URL) + "." + _alnum(rng, 43, B64URL)
    if kind == "hf":
        return "API_KEY", "h" + "f_" + _alnum(rng, 34, string.ascii_letters)
    if kind == "jwt":
        import base64
        import json
        head = base64.urlsafe_b64encode(json.dumps({"alg": rng.choice(["HS256", "RS256"]), "typ": "JWT"}).encode()).decode().rstrip("=")
        body = base64.urlsafe_b64encode(json.dumps({"sub": str(rng.randint(10000, 99999)), "iat": rng.randint(1_700_000_000, 1_790_000_000), "scope": "read"}).encode()).decode().rstrip("=")
        return "JWT", f"{head}.{body}.{_alnum(rng, 43, B64URL)}"
    return "API_KEY", _alnum(rng, rng.choice([32, 40]), B62)


def make_password(rng: random.Random) -> str:
    words = ["Sunrise", "Blue", "Tiger", "Falcon", "Harbor", "Maple", "Copper", "Lantern", "Orchid", "Granite", "Velvet", "Meteor"]
    return rng.choice(words) + rng.choice(["!", "#", "@", "$", "_"]) + str(rng.randint(10, 9999)) + rng.choice(words).lower()[:3]


# hard negatives: strings that look like identifiers but are not
def make_negative(rng: random.Random, kind: str) -> str:
    if kind == "order":
        return rng.choice(["ORD", "INV", "PO", "SO"]) + "-" + str(rng.randint(100000, 999999))
    if kind == "version":
        return f"v{rng.randint(0, 9)}.{rng.randint(0, 30)}.{rng.randint(0, 20)}"
    if kind == "sha":
        return "".join(rng.choice("0123456789abcdef") for _ in range(40))
    if kind == "uuid":
        h = "".join(rng.choice("0123456789abcdef") for _ in range(32))
        return f"{h[:8]}-{h[8:12]}-4{h[13:16]}-a{h[17:20]}-{h[20:]}"
    if kind == "sku":
        return f"SKU-{rng.randint(1000, 9999)}-{rng.randint(10, 99)}"
    if kind == "epoch":
        return str(rng.randint(1_690_000_000, 1_790_000_000))
    if kind == "ipver":
        return f"{rng.randint(1, 9)}.{rng.randint(0, 20)}.{rng.randint(0, 20)}.{rng.randint(0, 200)}"
    if kind == "long_num":
        while True:
            s = str(rng.randint(10**15, 10**16 - 1))
            if not ids.luhn_valid(s):
                return s
    if kind == "ticket":
        return f"TCK-{rng.randint(10000, 99999)}"
    if kind == "date":
        return f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    if kind == "org":
        return rng.choice(["Acme Logistics", "Globe Telecom", "Northwind Traders", "Contoso Retail", "Initech Systems", "Orbit Freight"])
    if kind == "city":
        return rng.choice(["Cebu City branch", "Madrid warehouse", "Austin office", "Manila hub"])
    if kind == "ref":
        return f"REF{rng.randint(10_000_000, 99_999_999)}"
    return "n/a"
