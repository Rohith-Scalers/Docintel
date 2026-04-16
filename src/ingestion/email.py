"""Email ingestor for .eml and .msg formats.

Extracts subject, sender, date, and body content from email files.
The body is rendered to a page image via TextIngestor. Attachments are
recorded in metadata but not processed inline.
"""
from __future__ import annotations

import email as stdlib_email
import email.policy
import logging
from email.message import Message

from src.ingestion.base import IngestorBase, PageResult
from src.ingestion.text import TextIngestor, _render_text_to_image, MAX_CHARS_PER_PAGE

logger = logging.getLogger(__name__)


def _render_body_to_pages(body: str, is_html: bool) -> list[PageResult]:
    """Convert an email body string to page results using text rendering.

    HTML bodies are passed through TextIngestor's HTML stripping logic by
    writing to a temporary in-memory path alias; plain text is rendered
    directly.

    Returns:
        list[PageResult]: page results for the email body.
    """
    import io
    import os
    import tempfile

    ext = ".html" if is_html else ".txt"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=ext)
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            fh.write(body)
        ingestor = TextIngestor()
        return ingestor.ingest(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _extract_eml(source_path: str) -> tuple[str, str, str, str, bool, list[str]]:
    """Parse an .eml file and return its key fields.

    Returns:
        tuple: (subject, from_addr, date, body_text, is_html, attachment_filenames)
    """
    with open(source_path, "rb") as fh:
        msg: Message = stdlib_email.message_from_binary_file(
            fh, policy=email.policy.default
        )

    subject: str = str(msg.get("Subject", ""))
    from_addr: str = str(msg.get("From", ""))
    date: str = str(msg.get("Date", ""))
    body: str = ""
    is_html: bool = False
    attachments: list[str] = []

    if msg.is_multipart():
        html_part: str = ""
        plain_part: str = ""
        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = str(part.get("Content-Disposition", ""))
            if "attachment" in disposition:
                filename = part.get_filename("")
                if filename:
                    attachments.append(filename)
                continue
            if content_type == "text/html" and not html_part:
                payload = part.get_payload(decode=True)
                if payload:
                    html_part = payload.decode(part.get_content_charset("utf-8"), errors="replace")
            elif content_type == "text/plain" and not plain_part:
                payload = part.get_payload(decode=True)
                if payload:
                    plain_part = payload.decode(part.get_content_charset("utf-8"), errors="replace")

        if html_part:
            body = html_part
            is_html = True
        else:
            body = plain_part
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset("utf-8")
            body = payload.decode(charset, errors="replace")
        is_html = msg.get_content_type() == "text/html"

    return subject, from_addr, date, body, is_html, attachments


def _extract_msg(source_path: str) -> tuple[str, str, str, str, bool, list[str]]:
    """Parse a .msg file using extract_msg and return its key fields.

    Returns:
        tuple: (subject, from_addr, date, body_text, is_html, attachment_filenames)
    """
    import extract_msg as extract_msg_lib

    msg = extract_msg_lib.openMsg(source_path)
    try:
        subject: str = msg.subject or ""
        from_addr: str = msg.sender or ""
        date: str = str(msg.date) if msg.date else ""
        body: str = msg.htmlBody or ""
        is_html: bool = bool(msg.htmlBody)
        if not body:
            body = msg.body or ""
        attachments: list[str] = [att.longFilename or att.shortFilename or "" for att in msg.attachments]
        attachments = [a for a in attachments if a]
    finally:
        msg.close()

    return subject, from_addr, date, body, is_html, attachments


class EmailIngestor(IngestorBase):
    """Ingestor for email files in .eml and .msg formats.

    Renders the email body as page images. Attachments are not processed
    inline; their filenames are stored in metadata['attachments'] so the
    caller can submit them as separate pipeline tasks.
    """

    @property
    def supported_extensions(self) -> list[str]:
        """File extensions handled by this ingestor.

        Returns:
            list[str]: ['.eml', '.msg']
        """
        return [".eml", ".msg"]

    def ingest(self, source_path: str) -> list[PageResult]:
        """Extract an email file and return one PageResult per body page.

        The first page carries full email metadata in its metadata dict.
        Subsequent pages (for long bodies) carry a subset for traceability.

        Returns:
            list[PageResult]: ordered list of page results, 1-indexed page_number.
        """
        ext = source_path.rsplit(".", 1)[-1].lower()

        if ext == "msg":
            subject, from_addr, date, body, is_html, attachments = _extract_msg(source_path)
        else:
            subject, from_addr, date, body, is_html, attachments = _extract_eml(source_path)

        body_pages = _render_body_to_pages(body, is_html)

        email_meta = {
            "email_subject": subject,
            "email_from": from_addr,
            "email_date": date,
            "attachments": attachments,
        }

        results: list[PageResult] = []
        for idx, page in enumerate(body_pages):
            page.is_scanned = False
            page.metadata = {**email_meta} if idx == 0 else {"email_subject": subject}
            results.append(page)

        return results
