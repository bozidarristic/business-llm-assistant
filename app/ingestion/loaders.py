from pathlib import Path
import pandas as pd

from app.data.schema import BusinessDocument


def load_customers(path: Path) -> list[BusinessDocument]:
    df = pd.read_csv(path)
    docs = []

    for _, row in df.iterrows():
        content = f"""
Customer record:
Customer ID: {row["customer_id"]}
Company: {row["company_name"]}
Industry: {row["industry"]}
Plan: {row["plan"]}
Status: {row["status"]}
Account owner: {row["account_owner"]}
Renewal date: {row["renewal_date"]}
ARR: {row["arr_eur"]} EUR
Health score: {row["health_score"]}
Notes: {row["notes"]}
""".strip()

        docs.append(
            BusinessDocument(
                id=f'customer-{row["customer_id"]}',
                source="customers.csv",
                content=content,
                metadata={
                    "type": "customer",
                    "customer_id": row["customer_id"],
                    "company_name": row["company_name"],
                    "status": row["status"],
                },
            )
        )

    return docs


def load_leads(path: Path) -> list[BusinessDocument]:
    df = pd.read_csv(path)
    docs = []

    for _, row in df.iterrows():
        content = f"""
Lead record:
Lead ID: {row["lead_id"]}
Company: {row["company_name"]}
Industry: {row["industry"]}
Lead source: {row["lead_source"]}
Estimated value: {row["estimated_value_eur"]} EUR
Stage: {row["stage"]}
Priority: {row["priority"]}
Next action: {row["next_action"]}
Notes: {row["notes"]}
""".strip()

        docs.append(
            BusinessDocument(
                id=f'lead-{row["lead_id"]}',
                source="leads.csv",
                content=content,
                metadata={
                    "type": "lead",
                    "lead_id": row["lead_id"],
                    "company_name": row["company_name"],
                    "priority": row["priority"],
                    "stage": row["stage"],
                },
            )
        )

    return docs


def load_support_tickets(path: Path) -> list[BusinessDocument]:
    df = pd.read_csv(path)
    docs = []

    for _, row in df.iterrows():
        content = f"""
Support ticket:
Ticket ID: {row["ticket_id"]}
Customer ID: {row["customer_id"]}
Company: {row["company_name"]}
Created at: {row["created_at"]}
Category: {row["category"]}
Priority: {row["priority"]}
Status: {row["status"]}
Subject: {row["subject"]}
Description: {row["description"]}
Latest update: {row["latest_update"]}
""".strip()

        docs.append(
            BusinessDocument(
                id=f'ticket-{row["ticket_id"]}',
                source="support_tickets.csv",
                content=content,
                metadata={
                    "type": "support_ticket",
                    "ticket_id": row["ticket_id"],
                    "customer_id": row["customer_id"],
                    "company_name": row["company_name"],
                    "priority": row["priority"],
                    "status": row["status"],
                },
            )
        )

    return docs


def load_internal_docs(path: Path) -> list[BusinessDocument]:
    text = path.read_text(encoding="utf-8")
    sections = [section.strip() for section in text.split("\n## ") if section.strip()]
    docs = []

    for idx, section in enumerate(sections, start=1):
        title = section.splitlines()[0].replace("#", "").strip()
        docs.append(
            BusinessDocument(
                id=f"internal-doc-{idx}",
                source="internal_docs.md",
                content=section,
                metadata={
                    "type": "internal_document",
                    "title": title,
                },
            )
        )

    return docs


def load_all_documents(raw_data_dir: Path) -> list[BusinessDocument]:
    documents: list[BusinessDocument] = []
    documents.extend(load_customers(raw_data_dir / "customers.csv"))
    documents.extend(load_leads(raw_data_dir / "leads.csv"))
    documents.extend(load_support_tickets(raw_data_dir / "support_tickets.csv"))
    documents.extend(load_internal_docs(raw_data_dir / "internal_docs.md"))
    return documents
