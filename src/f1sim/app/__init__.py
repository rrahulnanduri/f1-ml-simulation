"""The windowed app: the menu, the drive and replay screens.

pyglet checks ``glGetError`` after every GL call by default (``debug_gl``): a few thousand
calls a frame, a large share of the CPU's frame time on a slow machine. It's read once, when
pyglet's GL module loads, so it's switched off here, before anything in the app imports
arcade. ``F1SIM_GL_DEBUG=1`` turns the checks back on.
"""

import os

import pyglet

if not os.environ.get("F1SIM_GL_DEBUG"):
    pyglet.options.debug_gl = False
