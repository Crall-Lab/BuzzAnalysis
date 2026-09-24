#!/usr/bin/env python3
"""Render intro_manual.html to an exactly three-page PDF using installed PySide6.

Run: QT_QPA_PLATFORM=offscreen python docs/build_manual.py
"""
from pathlib import Path
import re

from PySide6.QtCore import QMarginsF, QRectF
from PySide6.QtGui import QFont, QPageLayout, QPageSize, QPainter, QPdfWriter, QTextDocument
from PySide6.QtWidgets import QApplication


def main():
    root = Path(__file__).resolve().parent
    source = (root / 'intro_manual.html').read_text(encoding='utf-8')
    style = re.search(r'<style>(.*?)</style>', source, re.S).group(1)
    pages = re.findall(r'<article data-page="\d+">(.*?)</article>', source, re.S)
    if len(pages) != 3:
        raise ValueError('The manual must contain three page articles')
    app = QApplication.instance() or QApplication([])
    writer = QPdfWriter(str(root / 'intro_manual.pdf'))
    writer.setTitle('BuzzAnalysis — Introduction')
    writer.setCreator('BuzzAnalysis manual builder')
    writer.setResolution(72)
    writer.setPageSize(QPageSize(QPageSize.A4))
    writer.setPageMargins(QMarginsF(38, 25, 38, 25), QPageLayout.Point)
    painter = QPainter(writer)
    try:
        width = writer.width()
        height = writer.height() - 24
        for number, content in enumerate(pages, 1):
            doc = QTextDocument()
            font = QFont('DejaVu Sans')
            font.setPointSizeF(9.5)
            doc.setDefaultFont(font)
            doc.setHtml(f'<style>{style}</style>{content}')
            doc.setTextWidth(width)
            size = doc.size()
            if size.height() > height:
                raise ValueError(f'Page {number} overflows: {size.height():.1f} > {height}')
            if number > 1:
                writer.newPage()
            doc.drawContents(painter, QRectF(0, 0, width, height))
            painter.setFont(QFont('DejaVu Sans', 8))
            painter.drawText(QRectF(0, height + 9, width, 15), 0,
                             f'BuzzAnalysis · testing_May25                                         {number} / 3')
            print(f'Page {number}: {size.height():.1f}/{height} pt')
    finally:
        painter.end()
    print(root / 'intro_manual.pdf')


if __name__ == '__main__':
    main()
