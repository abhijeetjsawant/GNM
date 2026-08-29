#!/usr/bin/env python3
"""Build the road-trip storyboard sheet from standalone frames."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
FRAMES_DIR = ROOT / "assets" / "storyboard" / "frames"
OUTPUT = ROOT / "assets" / "storyboard" / "road-trip-crash-storyboard.png"

PANELS = [
    (1, "CAMERA A · PHONE", "Establishing selfie", ["GIRL: “Okay, we are officially on the road.”"]),
    (2, "CAMERA A · PHONE", "Road-trip excitement", ["GIRL: “I’ve been waiting for this—good weather, music, snacks… beach.”"]),
    (3, "CAMERA A · PHONE", "Travel bag reveal", ["GIRL: “And I brought my little road-trip kit.”"]),
    (4, "CAMERA A · PHONE", "Good vibes", ["GIRL: “Good vibes. Good energy. Safe travels. No drama.”"]),
    (5, "CAMERA A · PHONE", "Product introduction", ["GIRL: “I always carry this. It just makes me feel safer.”"]),
    (6, "CAMERA A · PHONE", "Link in bio", ["GIRL: “Link in my bio. Use discount code RAZMATAZ.”"]),
    (7, "CAMERA A · PHONE", "Reveal boyfriend", ["GIRL: “He’s actually the reason I got into all this.”"]),
    (8, "CAMERA A · PHONE", "Say hi", ["GIRL: “Say hi.”", "DRIVER: “I’m driving.”"]),
    (9, "CAMERA A · PHONE", "Beach conversation", ["GIRL: “Anything you want to add? Excited?”", "DRIVER: “Yeah. Can’t wait for those great sea-thorn barnacles.”"]),
    (10, "CAMERA B · FRONT DASHCAM", "Dashcam established · distant brake lights", ["AUDIO: Steady road and cabin noise."]),
    (11, "CAMERA A · PHONE", "Product question", ["GIRL: “Tell them about the thing I made you buy.”", "DRIVER: “Yeah, yeah. It’s actually pretty good.”"]),
    (12, "CAMERA A · PHONE", "The distracting glance", ["DRIVER: “No, seriously. The great thing about it is—”"]),
    (13, "CAMERA B · FRONT DASHCAM", "Traffic closing · 3–4 car lengths", ["AUDIO: His unfinished sentence continues under the dashcam cut."]),
    (14, "CAMERA A · PHONE", "Realization", ["DRIVER: “Oh shit—”"]),
    (15, "CAMERA B · FRONT DASHCAM", "Emergency swerve · near-collision distance", ["GIRL (O.S.): “What?!”", "SFX: Tires scrub hard across the road."]),
    (16, "CAMERA A · PHONE", "Sharp turn toward divider", ["GIRL: “Oh my God—”", "SFX: Loose objects shift across the cabin."]),
    (17, "CAMERA C · REAR DASHCAM", "Divider impact", ["SFX: Impact, tire noise and metal scraping."]),
    (18, "CAMERA A · PHONE", "Inside the roll", ["AUDIO: Shouts become unintelligible beneath the rollover."]),
    (19, "CAMERA C · REAR DASHCAM", "Immediate aftermath", ["SFX: Brakes, scattered debris, then settling road noise."]),
]

PAGE = "#F2F0EA"
INK = "#17191C"
MUTED = "#62666B"
RULE = "#C9C6BE"
CAMERA_COLORS = {
    "CAMERA A": "#2563A8",
    "CAMERA B": "#B66A1B",
    "CAMERA C": "#9E3434",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


def main() -> None:
    missing = [
        FRAMES_DIR / f"panel-{number:02d}.png"
        for number, _, _, _ in PANELS
        if not (FRAMES_DIR / f"panel-{number:02d}.png").exists()
    ]
    if missing:
        raise SystemExit("Missing frames: " + ", ".join(str(path) for path in missing))

    columns = 5
    rows = (len(PANELS) + columns - 1) // columns
    panel_w, image_h, label_h = 720, 405, 152
    gap_x, gap_y = 22, 34
    margin_x, margin_bottom, header_h = 56, 56, 128
    page_w = margin_x * 2 + columns * panel_w + (columns - 1) * gap_x
    page_h = header_h + rows * (image_h + label_h) + (rows - 1) * gap_y + margin_bottom

    sheet = Image.new("RGB", (page_w, page_h), PAGE)
    draw = ImageDraw.Draw(sheet)
    title_font = font(42, bold=True)
    subtitle_font = font(22)
    source_font = font(24, bold=True)
    action_font = font(22)
    dialogue_font = font(19)

    draw.text((margin_x, 34), "ROAD TRIP PHONE + DASHCAM CRASH", font=title_font, fill=INK)
    draw.text(
        (margin_x, 86),
        f"{len(PANELS)} sequential frames · reconstructed only from in-vehicle cameras",
        font=subtitle_font,
        fill=MUTED,
    )
    draw.line((margin_x, 116, page_w - margin_x, 116), fill=RULE, width=2)

    for index, (number, source, action, dialogue) in enumerate(PANELS):
        row, col = divmod(index, columns)
        items_in_row = min(columns, len(PANELS) - row * columns)
        row_offset = (columns - items_in_row) * (panel_w + gap_x) // 2
        x = margin_x + row_offset + col * (panel_w + gap_x)
        y = header_h + row * (image_h + label_h + gap_y)
        frame_path = FRAMES_DIR / f"panel-{number:02d}.png"
        with Image.open(frame_path) as source_image:
            fitted = ImageOps.fit(
                source_image.convert("RGB"),
                (panel_w, image_h),
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.5),
            )
        sheet.paste(fitted, (x, y))
        draw.rectangle((x, y, x + panel_w - 1, y + image_h - 1), outline=INK, width=2)

        camera_key = source.split(" · ")[0]
        source_color = CAMERA_COLORS[camera_key]
        source_line = f"{number:02d}  {source}"
        draw.text((x, y + image_h + 12), source_line, font=source_font, fill=source_color)
        draw.text((x, y + image_h + 47), action, font=action_font, fill=INK)

        dialogue_y = y + image_h + 80
        for dialogue_line in dialogue:
            words = dialogue_line.split()
            wrapped_lines = []
            current_line = ""
            for word in words:
                candidate = f"{current_line} {word}".strip()
                if draw.textlength(candidate, font=dialogue_font) <= panel_w - 12:
                    current_line = candidate
                else:
                    if current_line:
                        wrapped_lines.append(current_line)
                    current_line = word
            if current_line:
                wrapped_lines.append(current_line)

            for wrapped_line in wrapped_lines:
                draw.text((x, dialogue_y), wrapped_line, font=dialogue_font, fill=MUTED)
                dialogue_y += 24

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(OUTPUT, format="PNG", optimize=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()
