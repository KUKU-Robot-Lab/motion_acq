// What the headset shows. Pure functions: viewer.js uses them in the page, and
// tests/test_quest_view_mode.py loads this file through node.
//
// User 10.04: once the Quest is connected the wearer sees the Quest's own front view
// (passthrough); the head camera picture covers the view only from the neck start on.
"use strict";

// Passthrough (immersive-ar) first; a browser without it gets the old VR screen.
const SESSION_MODES = ["immersive-ar", "immersive-vr"];

// One XR frame: draw the head camera screen, and the clear colour behind it.
// Transparent clear in immersive-ar = the room through the Quest cameras.
function frameLook(mode, hasImage) {
  const camera = hasImage || mode !== "immersive-ar";
  return { camera, clear: camera ? [0.02, 0.03, 0.05, 1.0] : [0.0, 0.0, 0.0, 0.0] };
}

if (typeof module !== "undefined") module.exports = { SESSION_MODES, frameLook };
