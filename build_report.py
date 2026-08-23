"""Inject measured data + the contact sheet into report.html."""
import base64
import html
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
SCALE_MAX = 28.0

# (family, label, detail, img_s, ms, vram, steps, res, fbs, note)
DATA = [
    ("t", "384x384", "frame buf 8",            26.17, 38.2, 3.14, 1, "384x384", 8, ""),
    ("t", "512x512", "frame buf 8, tensor out", 17.12, 58.4, 3.69, 1, "512x512", 8, "no PIL conversion"),
    ("t", "512x512", "frame buf 12",           15.51, 64.5, 4.31, 1, "512x512", 12, ""),
    ("t", "512x512", "frame buf 8",            15.43, 64.8, 3.69, 1, "512x512", 8, ""),
    ("t", "512x512", "frame buf 4",            14.44, 69.2, 3.06, 1, "512x512", 4, ""),
    ("t", "512x512", "frame buf 1",            11.37, 88.0, 2.59, 1, "512x512", 1, ""),
    ("l", "512x512", "1 step, stream batch",   10.79, 92.7, 2.31, 1, "512x512", 1, "output unusable"),
    ("l", "512x512", "2 steps, stream batch",   7.23, 138.3, 2.31, 2, "512x512", 1, ""),
    ("l", "512x512", "4 steps, stream batch",   4.40, 227.2, 2.41, 4, "512x512", 1, ""),
    ("l", "512x512", "4 steps, sequential",     3.32, 300.9, 2.31, 4, "512x512", 1, "stream batch off"),
]

FAMILY = {"t": "SD-Turbo", "l": "SD 1.5 + LCM-LoRA"}


def bars():
    out = []
    for fam, res, detail, ips, ms, vram, *_rest, note in DATA:
        w = 100.0 * ips / SCALE_MAX
        tip = f"{FAMILY[fam]} &middot; {res} &middot; {detail} &mdash; {ips:.2f} img/s, {ms:.1f} ms/img, {vram:.2f} GiB peak"
        out.append(
            f'      <div class="row">\n'
            f'        <div class="lab"><b>{res}</b>{html.escape(detail)}</div>\n'
            f'        <div class="track" tabindex="0" role="img" '
            f'aria-label="{FAMILY[fam]}, {res}, {html.escape(detail)}: {ips:.2f} images per second">\n'
            f'          <div class="bar {fam}" style="width:{w:.2f}%"></div>\n'
            f'          <span class="val">{ips:.2f}<i>img/s</i></span>\n'
            f'        </div>\n'
            f'        <div class="tip">{tip}</div>\n'
            f'      </div>'
        )
    return "\n".join(out)


def table():
    out = []
    for fam, res, detail, ips, ms, vram, steps, resolution, fbs, note in DATA:
        best = ' class="best"' if ips == max(d[3] for d in DATA) else ""
        colour = "var(--turbo)" if fam == "t" else "var(--lcm)"
        extra = f' &middot; {html.escape(note)}' if note else ""
        out.append(
            f'        <tr{best}>'
            f'<td><i class="dot" style="background:{colour}"></i>{FAMILY[fam]}{extra}</td>'
            f'<td>{resolution.replace("x", "&times;")}</td>'
            f'<td>{steps}</td>'
            f'<td>{fbs}</td>'
            f'<td class="num">{ips:.2f}</td>'
            f'<td class="num">{ms:.1f}</td>'
            f'<td class="num">{vram:.2f} GiB</td>'
            f'</tr>'
        )
    return "\n".join(out)


def main():
    src = os.path.join(HERE, "report.html")
    with open(src, encoding="utf-8") as f:
        s = f.read()

    jpg = os.path.join(OUT, "contact_turbo.jpg")
    with open(jpg, "rb") as f:
        uri = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()

    s = s.replace("      __BARS__", bars())
    s = s.replace("        __TABLE__", table())
    s = s.replace("__CONTACT_SHEET__", uri)

    dst = os.path.join(HERE, "report_built.html")
    with open(dst, "w", encoding="utf-8") as f:
        f.write(s)
    print(f"wrote {dst}  ({os.path.getsize(dst)/1024:.0f} KiB)")
    for tok in ("__BARS__", "__TABLE__", "__CONTACT_SHEET__"):
        assert tok not in s, f"unreplaced token {tok}"
    print("all tokens replaced")


if __name__ == "__main__":
    main()
