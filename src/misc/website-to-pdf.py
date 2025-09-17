#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#     "typing-extensions==4.15.0",
#     "typer==0.16.1",
#     "beautifulsoup4==4.13.5",
#     "pypdf==6.0.0",
#     "playwright==1.54.0",
# ]
# ///

"""Convert websites to PDF.

# Setup
chmod +x website-to-pdf.py

# Use
./website-to-pdf.py https://site.com
./website-to-pdf.py https://site.com --depth 3  # deeper crawl
./website-to-pdf.py https://site.com --no-merge  # individual PDFs
./website-to-pdf.py https://site.com --max-concurrent 10  # faster crawling
"""

import asyncio
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urljoin, urlparse

import typer
from bs4 import BeautifulSoup
from playwright.async_api import TimeoutError as PlaywrightTimeout
from playwright.async_api import async_playwright
from pypdf import PdfWriter

TIMEOUT = 30000  # 30 seconds
PAGE_DELAY = 2000  # 2 seconds for JavaScript sites
BAD_EXTENSIONS = {
    ".pdf",
    ".zip",
    ".mp4",
    ".jpg",
    ".png",
    ".exe",
    ".json",
    ".mp3",
    ".avi",
    ".txt",
    ".rst",
}


def get_max_concurrent():
    cpu_count = os.cpu_count() or 4
    return min(cpu_count * 2, 10)


def make_filename(url: str) -> str:
    path = urlparse(url).path.strip("/")
    if not path:
        return "index"

    # replace bad chars
    name = re.sub(r"[^\w\-]", "_", path)

    for ext in [".html", ".htm", ".php", ".asp"]:
        if name.endswith(ext):
            name = name[: -len(ext)]

    return name or "page"


def make_site_name(url: str) -> str:
    domain = urlparse(url).netloc.replace("www.", "")
    name = domain.split(".")[0]
    return re.sub(r"[^\w\-]", "_", name) or "website"


def url_ok(url: str, base_url: str, seen: set) -> bool:
    if not url or url in seen:
        return False

    # normalize
    url = url.split("#")[0].rstrip("/")
    if not url or url in seen:
        return False

    # must be same site
    parsed_url = urlparse(url)
    parsed_base = urlparse(base_url)
    if parsed_url.netloc and parsed_url.netloc != parsed_base.netloc:
        return False

    # skip bad stuff
    lower = url.lower()
    if any(ext in lower for ext in BAD_EXTENSIONS):
        return False

    # skip malformed sphinx/doc URLs that end with _html but no extension
    if re.search(r"[^/]+_html$", url) and "." not in url.split("/")[-1]:
        return False

    # skip bare paths that commonly 404
    if url.endswith("/schedule") or url.endswith("/genindex"):
        return False

    # skip non-English language variants but allow /en/ (English)
    lang_match = re.search(r"/([a-z]{2}(?:-[A-Z]{2})?)(?:/|$)", url)
    if lang_match:
        lang_code = lang_match.group(1).lower()
        if not lang_code.startswith("en"):
            return False

    # skip language suffixes like -zh, -fr, -ja (common in some sites)
    if re.search(
        r"-(?:zh|fr|ja|ko|de|es|pt|ru|ar|it|nl|sv|pl)(?:-[A-Z]{2})?(?:/|$)", lower
    ):
        return False

    # skip common language query params
    if re.search(r"[?&](?:lang|locale|hl)=[a-z]{2}", lower):
        return False

    # skip source/raw files (common in documentation sites)
    if "_sources_" in lower or "/sources/" in lower:
        return False

    if parsed_url.scheme in ["mailto", "javascript", "tel"]:
        return False

    return True


async def wait_for_javascript(page):
    try:
        # wait for common loading indicators to disappear
        await page.wait_for_selector(".loading", state="hidden", timeout=5000)
    except Exception:
        pass

    try:
        # wait for common content containers
        await page.wait_for_selector("main, article, .content, #content", timeout=5000)
    except Exception:
        pass

    # always wait a bit for dynamic content
    await page.wait_for_timeout(PAGE_DELAY)


async def get_page_content(context, url: str, pdf_path: Path) -> tuple[bool, str]:
    page = await context.new_page()
    try:
        # navigate with longer timeout for slow sites
        response = await page.goto(url, wait_until="domcontentloaded", timeout=TIMEOUT)

        if response and response.status >= 400:
            print(f"  ✗ HTTP {response.status}")
            return False, ""

        # wait for JavaScript content
        await wait_for_javascript(page)

        # scroll to trigger lazy loading
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(500)
        await page.evaluate("window.scrollTo(0, 0)")

        await page.pdf(
            path=str(pdf_path),
            format="A4",
            print_background=True,
            margin={"top": "10mm", "right": "10mm", "bottom": "10mm", "left": "10mm"},
        )
        html = await page.content()
        return True, html

    except PlaywrightTimeout:
        print("  ✗ Timeout")
        return False, ""
    except Exception as e:
        print(f"  ✗ Error: {str(e)[:50]}")
        return False, ""
    finally:
        await page.close()


def extract_links(html: str, base_url: str) -> list[str]:
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    links = []
    seen_links = set()

    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        if href:
            full_url = urljoin(base_url, href)
            full_url = full_url.split("#")[0].rstrip("/")
            if full_url and full_url not in seen_links:
                links.append(full_url)
                seen_links.add(full_url)

    # also check for links in JS
    for script in soup.find_all("script"):
        if script.string:
            # look for common patterns
            js_links = re.findall(r'["\']/([\w\-/]+\.html?)["\']', script.string)
            for link in js_links:
                full_url = urljoin(base_url, "/" + link)
                full_url = full_url.split("#")[0].rstrip("/")
                if full_url and full_url not in seen_links:
                    links.append(full_url)
                    seen_links.add(full_url)

    return links


async def process_batch(
    context, batch, pdf_dir, base_url, seen, semaphore, pdf_info
) -> list[tuple[str, int]]:
    async def process_single(url_info):
        url, depth, order = url_info
        async with semaphore:
            if url in seen:
                return []
            seen.add(url)

            filename = make_filename(url)
            pdf_path = pdf_dir / f"{filename}_{len(seen)}.pdf"

            print(f"  📄 {filename}...")
            success, html = await get_page_content(context, url, pdf_path)

            if success and pdf_path.exists():
                print(f"  ✓ {filename}")
                pdf_info.append((depth, order, pdf_path))

                new_links = extract_links(html, base_url)
                valid_links = [
                    link for link in new_links if url_ok(link, base_url, seen)
                ]
                return [(link, depth + 1, i) for i, link in enumerate(valid_links)]
            return []

    results = await asyncio.gather(*[process_single(url_info) for url_info in batch])

    all_new_links = []
    for links in results:
        all_new_links.extend(links)

    return all_new_links


async def crawl_site(
    url: str, depth: int = 3, merge: bool = True, max_concurrent: int = None
):
    print(f"\n🌐 Converting {url} to PDF...")

    if max_concurrent is None:
        max_concurrent = get_max_concurrent()

    # setup directories
    site_name = make_site_name(url)
    if merge:
        pdf_dir = Path(f"temp_{site_name}")
        pdf_dir.mkdir(exist_ok=True)
        final_pdf = f"{site_name}.pdf"
    else:
        pdf_dir = Path(site_name)
        pdf_dir.mkdir(exist_ok=True)

    print("🌐 Starting browser...")
    subprocess.run(
        [sys.executable, "-m", "playwright", "install", "chromium"],
        capture_output=True,
        check=False,
    )

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()

        seen = set()
        to_crawl = [(url, 0, 0)]  # (url, depth, order)
        current_depth = 0
        semaphore = asyncio.Semaphore(max_concurrent)
        pdf_info = []  # collect PDFs with ordering info

        print(f"⚡ Using {max_concurrent} concurrent connections")

        while to_crawl and current_depth <= depth:
            # get all URLs at current depth
            current_batch = []
            next_depth_urls = []

            for url_info in to_crawl:
                u, d, o = url_info
                if d == current_depth:
                    current_batch.append(url_info)
                elif d > current_depth:
                    next_depth_urls.append(url_info)

            if not current_batch:
                current_depth += 1
                to_crawl = next_depth_urls
                continue

            print(
                f"\n📊 Depth {current_depth}: Processing {len(current_batch)} pages..."
            )

            new_links = await process_batch(
                context, current_batch, pdf_dir, url, seen, semaphore, pdf_info
            )

            # add new links for next iteration
            to_crawl = next_depth_urls + new_links

            if not any(d == current_depth for _, d, _ in to_crawl):
                current_depth += 1

        await context.close()
        await browser.close()

    pdf_info.sort(key=lambda x: (x[0], x[1]))
    pdfs = [info[2] for info in pdf_info]

    if merge and pdfs:
        print(f"\n📚 Merging {len(pdfs)} PDFs in order...")
        merger = PdfWriter()

        for pdf in pdfs:
            try:
                merger.append(str(pdf))
            except Exception as e:
                print(f"  ✗ Cannot merge {pdf.name}: {e}")

        merger.write(final_pdf)
        merger.close()

        # cleanup
        for pdf in pdfs:
            pdf.unlink()
        pdf_dir.rmdir()

        print(f"✅ Saved to {final_pdf}")
    else:
        print(f"✅ Saved {len(pdfs)} PDFs to {pdf_dir}/")


# cli

app = typer.Typer(help="Convert websites to PDF with async crawling")


@app.command()
def main(
    url: str = typer.Argument(..., help="Website URL"),
    depth: int = typer.Option(3, "-d", "--depth", help="Crawl depth (default: 3)"),
    merge: bool = typer.Option(
        True, "--merge/--no-merge", help="Merge into single PDF"
    ),
    max_concurrent: int = typer.Option(
        None, "--max-concurrent", help="Max concurrent pages"
    ),
):
    asyncio.run(crawl_site(url, depth, merge, max_concurrent))


if __name__ == "__main__":
    app()
