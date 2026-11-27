"""Record formats the statistical detector is never trained on: a quoted e-mail thread, meeting notes and a chat export.

They use the same identifier generators as the main corpus, so ground truth stays exact by construction, but the
surrounding text, the layout and the cues are new. They exist to show how each detector behaves outside the generator's
training distribution.
"""
from __future__ import annotations

from veilbench.corpus import people as P
from veilbench.corpus.generate import Ctx, strip

OOD_KINDS = [("email_thread", 0.4), ("meeting_notes", 0.35), ("chat_log", 0.25)]


def email_thread(ctx: Ctx) -> None:
    rng, tb, p = ctx.rng, ctx.tb, ctx.p
    tb.t("On ")
    tb.n(P.make_negative(rng, "date"))
    tb.t(", ")
    tb.e("PERSON", p.full)
    tb.t(" <")
    tb.e("EMAIL", p.email)
    tb.t("> wrote:\n")
    for line in rng.sample(["> Could you confirm the delivery to {ADDRESS}?", "> My direct line is {PHONE}.", "> The invoice {NEG_ORDER} is still open.",
                            "> Payment was made with {CARD}.", "> Please use {PERSON2} as the second contact."], 3):
        ctx.fill(line)
        tb.t("\n")
    tb.t("\n--\n")
    ctx.slot("FULL")
    tb.t("\n" + rng.choice(["Procurement Lead", "Customer Success", "Operations Analyst"]) + " | ")
    ctx.slot("PHONE")
    tb.t("\n")
    ctx.slot("ADDRESS")
    tb.t("\n")


def meeting_notes(ctx: Ctx) -> None:
    rng, tb = ctx.rng, ctx.tb
    tb.t("## Notes (")
    tb.n(P.make_negative(rng, "date"))
    tb.t(")\nAttendees: ")
    ctx.slot("FULL")
    tb.t(", ")
    ctx.slot("PERSON2")
    tb.t("\n")
    for line in rng.sample(["- Action: {FIRST} to call {PERSON2} on {PHONE}", "- Billing question on card {CARD}, ref {NEG_REF}", "- Ship the replacement to {ADDRESS}",
                            "- Verify identity with ID {ID} and DOB {DOB}", "- Reset login for {EMAIL}, temporary password {PASSWORD}",
                            "- Rotate the key {KEY} used on build {NEG_SHA}"], 4):
        ctx.fill(line)
        tb.t("\n")


def chat_log(ctx: Ctx) -> None:
    rng, tb, p = ctx.rng, ctx.tb, ctx.p
    handle = strip(p.first).lower() + "." + strip(p.last).lower().replace(" ", "")
    for _ in range(rng.randint(3, 5)):
        tb.t(f"[{rng.randint(8, 18):02d}:{rng.randint(0, 59):02d}] ")
        form = rng.randint(0, 3)
        if form == 0:
            tb.e("PERSON", handle, p.full)
            tb.t(": pushed the fix for ")
            tb.n(P.make_negative(rng, "ticket"))
            tb.t(", build ")
            tb.n(P.make_negative(rng, "sha")[:10])
        elif form == 1:
            tb.t("@")
            tb.e("PERSON", handle, p.full)
            tb.t(" can you resend the credentials? ")
            label, value = P.make_secret(rng, rng.choice(["stripe", "github", "generic"]))
            tb.e(label, value)
        elif form == 2:
            tb.t("bot: new signup ")
            ctx.slot("EMAIL")
            tb.t(" from ")
            ctx.slot("IP")
        else:
            tb.t("bot: shipping label printed for ")
            ctx.slot("FULL")
            tb.t(", ")
            ctx.slot("ADDRESS")
        tb.t("\n")


BUILDERS = {"email_thread": email_thread, "meeting_notes": meeting_notes, "chat_log": chat_log}
