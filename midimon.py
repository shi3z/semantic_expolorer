"""Print every MIDI message from a controller. Use this to see what your pads
and knobs actually send before wiring them to anything.

    .venv/Scripts/python.exe midimon.py
"""
import argparse
import sys

import mido


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default=None, help="substring of the port name")
    a = p.parse_args()

    names = mido.get_input_names()
    if not names:
        sys.exit("no MIDI input ports found - is the controller plugged in?")

    if a.port:
        matches = [n for n in names if a.port.lower() in n.lower()]
        if not matches:
            sys.exit(f"no port matching {a.port!r}. available: {names}")
        name = matches[0]
    else:
        name = names[0]

    print(f"listening on {name!r}   (ctrl-c to stop)\n")
    seen_cc, seen_note = {}, {}
    with mido.open_input(name) as port:
        for msg in port:
            if msg.type == "control_change":
                seen_cc[msg.control] = msg.value
                print(f"  knob   cc={msg.control:<4} value={msg.value:<4} ch={msg.channel}"
                      f"   all ccs seen: {sorted(seen_cc)}")
            elif msg.type in ("note_on", "note_off"):
                if msg.type == "note_on" and msg.velocity > 0:
                    seen_note[msg.note] = msg.velocity
                    print(f"  pad ON note={msg.note:<4} vel={msg.velocity:<4} ch={msg.channel}"
                          f"   all notes seen: {sorted(seen_note)}")
                else:
                    print(f"  pad off note={msg.note}")
            else:
                print(f"  {msg}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped")
