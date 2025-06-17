# /// script
# dependencies = [
#     "modal>=1.0.3",
#     "requests>=2.32.3",
#     "python-dotenv>=1.0.1"
# ]
# ///

# in .env, set EMAIL_SENDER, SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD
# run locally with `uv run --script pubmed.py`
# run remotely with `uvx --with python-dotenv modal run pubmed.py`
# deploy with `uvx --with python-dotenv modal deploy pubmed.py`

import os
import smtplib
import ssl
import xml.etree.ElementTree as ET
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import modal

RECIPIENTS = ["ajhinh@gmail.com"]

app = modal.App("pubmed-nutrition-alerts")
secrets = [
    modal.Secret.from_dotenv(
        path=Path(__file__).parent,
        filename=".env",
    )
]
image = modal.Image.debian_slim().pip_install(
    "requests>=2.32.3",
)

with image.imports():
    import requests


def _send_email(email: str, subject: str, text_body: str, html_body: str) -> bool:
    sender_email = os.getenv("EMAIL_SENDER")
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = int(os.getenv("SMTP_PORT"))
    smtp_username = os.getenv("SMTP_USERNAME")
    smtp_password = os.getenv("SMTP_PASSWORD")

    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = sender_email
    message["To"] = email

    message.attach(MIMEText(text_body, "plain"))
    message.attach(MIMEText(html_body, "html"))

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            if smtp_username and smtp_password:
                server.login(smtp_username, smtp_password)
            server.sendmail(sender_email, email, message.as_string())
            return True
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return False


def fetch_latest_pubmed_papers(max_results: int = 5, timeout: int = 10):
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {
        "db": "pubmed",
        "sort": "pub+date",
        "term": "nutrition[Title/Abstract]",
        "retmax": max_results,
        "retmode": "json",
    }
    try:
        id_resp = requests.get(search_url, params=search_params, timeout=timeout)
        id_resp.raise_for_status()
        id_json = id_resp.json()
    except Exception as e:
        print(f"Error fetching PubMed IDs: {e}")
        return []
    ids = id_json.get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []

    summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    summary_params = {
        "db": "pubmed",
        "id": ",".join(ids),
        "retmode": "json",
    }
    try:
        summary_resp = requests.get(summary_url, params=summary_params, timeout=timeout)
        summary_resp.raise_for_status()
        summary_json = summary_resp.json()
    except Exception as e:
        print(f"Error fetching PubMed summaries: {e}")
        return []

    # Fetch abstracts via efetch
    abstract_map = {}
    efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    try:
        efetch_resp = requests.get(
            efetch_url,
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml"},
            timeout=timeout,
        )
        efetch_resp.raise_for_status()
        root = ET.fromstring(efetch_resp.text)
        for article in root.findall(".//PubmedArticle"):
            pmid = article.findtext(".//PMID")
            abstract_elems = article.findall(".//Abstract/AbstractText")
            abstract = " ".join((elem.text or "") for elem in abstract_elems).strip()
            abstract_map[pmid] = abstract
    except Exception as e:
        print(f"Error fetching PubMed abstracts: {e}")

    papers = []
    for pid in ids:
        item = summary_json["result"].get(pid, {})
        title = item.get("title", "No title available")
        journal = item.get("fulljournalname", "Unknown journal")
        pub_date = item.get("pubdate") or item.get("sortpubdate") or "n.d."
        authors_list = item.get("authors", [])
        authors = ", ".join(a.get("name") for a in authors_list)
        abstract = abstract_map.get(pid, "No abstract available")
        papers.append(
            {
                "title": title,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pid}/",
                "journal": journal,
                "pub_date": pub_date,
                "authors": authors,
                "abstract": abstract,
            }
        )
    return papers


@app.function(image=image, secrets=secrets)
def send_pubmed_update_email():
    papers = fetch_latest_pubmed_papers()
    print(f"Retrieved {len(papers)} papers from PubMed.")
    if not papers:
        print("No new papers found.")
        return

    text_lines = [
        (
            f"- {p['title']} ({p['journal']}, {p['pub_date']}) by {p['authors']}\n"
            f"  {p['url']}\n"
            f"  Abstract: {p['abstract'][:400]}{'...' if len(p['abstract']) > 400 else ''}"
        )
        for p in papers
    ]
    text_body = "Latest nutrition papers on PubMed:\n\n" + "\n".join(text_lines)

    html_items = "".join(
        f"""
         <li>
           <strong>{p["title"]}</strong><br/>
           <em>{p["journal"]}</em>, {p["pub_date"]}<br/>
           by {p["authors"]}<br/>
           <a href="{p["url"]}">{p["url"]}</a><br/>
           <p>{p["abstract"]}</p>
         </li>
         """
        for p in papers
    )
    html_body = f"""
        <html>
          <body>
            <h2>Latest nutrition papers on PubMed</h2>
            <ul>{html_items}</ul>
          </body>
        </html>
    """

    subject = "Daily PubMed Nutrition Digest"
    for recipient in RECIPIENTS:
        print(f"Dispatching digest to {recipient}")
        _send_email(recipient, subject, text_body, html_body)


@app.function(schedule=modal.Cron("0 8 * * *", timezone="America/Los_Angeles"))
def daily_pubmed_alert():
    print("Starting daily_pubmed_alert job.")
    if modal.is_local():
        send_pubmed_update_email.local()
    else:
        send_pubmed_update_email.remote()


@app.local_entrypoint()
def main():
    daily_pubmed_alert.remote()


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    daily_pubmed_alert.local()
