#!/usr/bin/env python3
"""make_checkerboard.py [out.pdf] -- camera calibration target: a 10 x 7 square checkerboard (9 x 6 INNER corners,
the pattern size OpenCV and ROS camera_calibration use), 25 mm squares, centred on an A4 landscape page.
Print at 100 % ("Actual size"), check with a ruler: 10 squares across = 250 mm, 7 squares down = 175 mm."""
import sys

from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

COLS, ROWS, SQ = 10, 7, 25.0          # squares across, squares down, square size [mm]
out = sys.argv[1] if len(sys.argv) > 1 else 'checkerboard_9x6_25mm_A4.pdf'
W, H = landscape(A4)                  # 297 x 210 mm in points
bw, bh = COLS * SQ * mm, ROWS * SQ * mm
x0, y0 = (W - bw) / 2, (H - bh) / 2
c = canvas.Canvas(out, pagesize=(W, H))
c.setTitle('Checkerboard 9x6 inner corners, 25 mm squares (A4)')
c.setFillColorRGB(0, 0, 0)
c.setStrokeColorRGB(0, 0, 0)
for r in range(ROWS):
    for k in range(COLS):
        if (r + k) % 2 == 0:          # top-left square black
            c.rect(x0 + k * SQ * mm, y0 + (ROWS - 1 - r) * SQ * mm, SQ * mm, SQ * mm, stroke=0, fill=1)
c.setFont('Helvetica', 7)
c.setFillColorRGB(0.35, 0.35, 0.35)
c.drawString(8 * mm, H - 7 * mm, 'Calibration board: 9 x 6 inner corners, 25.0 mm squares. Print at 100 % / "Actual size" '
             '(NOT "Fit to page"). Check: 10 squares across = 250 mm.')
c.drawString(8 * mm, 5 * mm, 'Glue it flat on stiff board (cardboard, foam board, clipboard). Measure the real square size '
             'with a ruler and write it here: ______ mm')
c.showPage()
c.save()
print(out)
