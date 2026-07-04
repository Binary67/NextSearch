from __future__ import annotations


def build_text_pdf(page_texts: list[str]) -> bytes:
    objects: list[tuple[int, bytes]] = []
    page_refs: list[str] = []
    content_streams: list[tuple[int, bytes]] = []
    font_id = 3 + len(page_texts)

    for index, text in enumerate(page_texts, start=1):
        page_id = 2 + index
        content_id = font_id + index
        page_refs.append(f"{page_id} 0 R")

        content = _page_content(text)
        content_streams.append((content_id, content))
        objects.append(
            (
                page_id,
                (
                    "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                    f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
                    f"/Contents {content_id} 0 R >>"
                ).encode("utf-8"),
            )
        )

    objects.extend(
        [
            (1, b"<< /Type /Catalog /Pages 2 0 R >>"),
            (
                2,
                (
                    f"<< /Type /Pages /Kids [{' '.join(page_refs)}] "
                    f"/Count {len(page_refs)} >>"
                ).encode("utf-8"),
            ),
            (font_id, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"),
        ]
    )
    objects.extend(
        (
            object_id,
            (
                b"<< /Length "
                + str(len(stream)).encode("utf-8")
                + b" >>\nstream\n"
                + stream
                + b"\nendstream"
            ),
        )
        for object_id, stream in content_streams
    )
    objects.sort(key=lambda item: item[0])

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for object_id, body in objects:
        offsets.append(len(pdf))
        pdf.extend(f"{object_id} 0 obj\n".encode("utf-8"))
        pdf.extend(body)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("utf-8"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("utf-8"))
    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("utf-8")
    )
    return bytes(pdf)


def _page_content(text: str) -> bytes:
    commands = ["BT", "/F1 12 Tf", "72 720 Td"]
    for line_index, line in enumerate(text.split("\n")):
        if line_index:
            commands.append("0 -16 Td")
        commands.append(f"({_escape_pdf_text(line)}) Tj")
    commands.append("ET")
    return "\n".join(commands).encode("utf-8")


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
