# Security

Read this before putting facegate on a real door. Some of what follows is
uncomfortable, and deliberately so.

## The short version

facegate is a single-board system on a trusted LAN. It has **no authentication
on its web interface**, and its **anti-spoofing is not validated against real
attacks**. It is suitable for a home, a hobby project, or a door where a
determined attacker is not the threat model. It is not a replacement for a
commercial access controller.

## What it gets right

**Fails closed.** An unrecognised face is `DENY`, not a grant
(`access.require_match: true`). If the database is empty, everyone is denied.

**The embedding cache is gated on identity, not position.** A cached embedding
is only reused when a 16x16 appearance descriptor matches, so a different person
in the same spot does not inherit the previous one's identity. See
[ARCHITECTURE.md](ARCHITECTURE.md).

**Liveness runs only on faces that would be granted.** A stranger at the door
never costs the extra pass, which keeps the cost of the check where it matters.

**Small faces are refused.** Below `recognizer.min_face_size` (60 px) a face is
ignored rather than matched — a tiny face gives an unreliable embedding, and
that is the main way a false accept happens.

## The web interface has no login

Anyone who can reach port 8080 can enrol people, delete them, and change the
security thresholds. This is deliberate: a home or office LAN where that is
acceptable, and nothing more.

If the board is reachable from anywhere else:

- put it behind a reverse proxy with authentication, or
- bind `web.host` to `127.0.0.1` and use an SSH tunnel.

Do not expose the port directly to the internet.

## Anti-spoofing is the weakest part

MiniFASNet V2 SE is a real model (98.2% on its own benchmark), but **I have not
validated it against real print or screen attacks here** — I had no physical
printed photo or phone screen to test with, and the public anti-spoof datasets
are large and gated.

What I could do was test software-synthesised attacks, and the honest result is
that they are not representative:

| Test | Score | Verdict |
|---|---|---|
| Genuine face photos | 0.50 - 0.90 | live (correct) |
| Low-contrast "print" | 0.02 | spoof (correct) |
| Over-bright "screen" | 0.47 | spoof (correct) |
| Moiré screen replay | 0.86 | **live — attack missed** |
| Low-quality JPEG | 0.94 | **live — attack missed** |
| Upscaled/resampled | 0.63 | **live — attack missed** |

Realistic digital spoofs sail through.

### The threshold has no comfortable value

Measured against a real face at the camera (92 granted frames), the genuine-face
score has a long low tail:

```
min 0.244 | p1 0.289 | p5 0.389 | median 0.822 | max 0.985
```

which collides with the spoof scores above:

| Threshold | False `DENY_SPOOF` on a genuine face |
|---|---|
| 0.5 | 8.7% of frames |
| 0.4 | 5.4% |
| **0.3 (default)** | **2.2%** |
| 0.25 | 1.1% |
| 0.2 | 0.0% |

`0.2` would give a real person a clean pass, but it sits *below* the one real
spoof measured here (`0.47`). **This is a limitation of the model on a cheap
webcam, not a number you can tune your way out of.**

The default is 0.3 because a false `DENY_SPOOF` locks a real person out of their
own door, which is the worse failure for a single-person installation. If you
enrol several people, or the door is unattended, raise it back toward 0.5 and
accept the false denials.

### To test it yourself

Print a photo of an enrolled person, hold it to the camera, and read the score
off the `PHOTO/SPOOF` box in the live view. If it stays above your threshold,
the model is not catching it and you need something stronger.

## If you need real security

In rough order of how much they help:

1. **Add a second factor** — `access.webhook` POSTs every decision to a system
   that can require a password, a badge, or a second approval.
2. **Require a challenge-response gesture** — turn your head, blink. Much
   stronger than a single frame, and needs a tracking implementation.
3. **Add depth or a second camera at an angle.** Geometry beats heuristics: a
   photograph has no depth and no parallax. This is the only measure that
   fundamentally defeats the replay attack rather than raising the bar.
4. **Re-enrol after changing lighting or camera position.** Recognition accuracy
   depends heavily on the angle at the door; a 45-degree off-axis view degrades
   cosine similarity substantially. Mount the camera at face height, straight on.

## Operational notes

- `data/events.jsonl` is a plain-text access log containing names, timestamps
  and bounding boxes. Treat it as personal data.
- `data/facedb.json` contains biometric embeddings. It is in `.gitignore` for
  a reason — **never commit it**.
- Enrolment photos in `faces/` are images of people. Also gitignored.
- Rotating embeddings is not supported. To invalidate someone's enrolment,
  remove them: the `×` in the People tab, or `--remove NAME`.
