# F1 Anatomy

Five interactive isometric figures that explain how an F1 car works: chassis and safety, power
unit, aerodynamics, transmission and steering, suspension, brakes and tyres. Each has live
controls, readouts driven by a small model, and a guided walk-through.

This folder is self-contained. Nothing else in the repository imports it.

## Use it

Built pages are in `site/`; open `site/index.html`, or:

```bash
python f1_anatomy/launch.py          # index
python f1_anatomy/launch.py aero     # one figure
```

To hook it into a menu, call `open_showcase()` from `f1_anatomy/launch.py` (standard library only).

## Rebuild (needs Node 20+)

```bash
node f1_anatomy/build.mjs            # everything
node f1_anatomy/build.mjs corner     # one figure
node f1_anatomy/serve.mjs            # preview on http://127.0.0.1:5177/site/
```

## Layout

- `src/lib.mjs` page shell, controls, walk-through runtime. `src/figures/<id>.*` one figure each:
  `.model.mjs` numbers, `.geo.mjs` drawing, `.live.js` behaviour, `.mjs` page description.
- `vendor/` the Anatomy drawing kit (MIT, see `vendor/LICENSE-anatomy`).
- `tools/` headless screenshots (`shot.mjs`) and a small DevTools driver (`cdp.mjs`).
- Page URL hooks for testing: `#theme=light&view=stage&step=2&speed=300`.

Quantities are rounded, typical figures from recent regulations; the models are simplified.
