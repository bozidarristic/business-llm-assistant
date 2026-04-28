from app.data.schema import RetrievedChunk


def _field(content: str, name: str) -> str | None:
    prefix = f"{name}:"
    for line in content.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return None


def extract_facts(chunks: list[RetrievedChunk]) -> str:
    facts = []

    for chunk in chunks:
        doc_type = chunk.metadata.get("type")
        content = chunk.content

        if doc_type == "customer":
            facts.append(
                "Customer: "
                f"{_field(content, 'Company')} | "
                f"ID {_field(content, 'Customer ID')} | "
                f"plan {_field(content, 'Plan')} | "
                f"status {_field(content, 'Status')} | "
                f"health {_field(content, 'Health score')} | "
                f"notes {_field(content, 'Notes')}"
            )
        elif doc_type == "lead":
            facts.append(
                "Lead: "
                f"{_field(content, 'Company')} | "
                f"ID {_field(content, 'Lead ID')} | "
                f"priority {_field(content, 'Priority')} | "
                f"value {_field(content, 'Estimated value')} | "
                f"stage {_field(content, 'Stage')} | "
                f"next action {_field(content, 'Next action')} | "
                f"notes {_field(content, 'Notes')}"
            )
        elif doc_type == "support_ticket":
            facts.append(
                "Ticket: "
                f"{_field(content, 'Ticket ID')} | "
                f"{_field(content, 'Company')} | "
                f"priority {_field(content, 'Priority')} | "
                f"status {_field(content, 'Status')} | "
                f"subject {_field(content, 'Subject')} | "
                f"latest update {_field(content, 'Latest update')}"
            )
        elif doc_type == "internal_document":
            title = chunk.metadata.get("title", "Internal policy")
            compact_content = " ".join(content.split())
            facts.append(f"Policy: {title} | {compact_content}")

    return "\n".join(f"- {fact}" for fact in facts if fact)


def format_context(chunks: list[RetrievedChunk]) -> str:
    formatted = []

    for i, chunk in enumerate(chunks, start=1):
        metadata_parts = []
        for label, key in [
            ("Source", "source"),
            ("Type", "type"),
            ("Company", "company_name"),
            ("Customer ID", "customer_id"),
            ("Ticket ID", "ticket_id"),
            ("Lead ID", "lead_id"),
            ("Policy", "title"),
        ]:
            value = chunk.metadata.get(key)
            if value:
                metadata_parts.append(f"{label}: {value}")

        formatted.append(
            f"[Context {i}]\n{' | '.join(metadata_parts)}\n{chunk.content}"
        )

    return "\n\n".join(formatted)


def build_rag_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    facts = extract_facts(chunks)
    context = format_context(chunks)

    return f"""
You are a practical business assistant.

Your task:
- Answer the user question using ONLY the provided context.
- If the context is insufficient, say clearly: "I do not have enough information in the provided data."
- Do not invent customer facts, ticket details, numbers, policies, dates, or commitments.
- Be concise, professional, and useful.
- If the user asks you to draft a reply, write the reply directly from our company to the external customer or lead.
- Drafted replies must include a subject line, greeting, body, and sign-off.
- Never sign a drafted reply as the customer, lead, or recipient company.
- For customer replies, apologize when the context says there was an SLA gap, state the concrete next action, and provide a realistic follow-up without inventing exact times.
- Use the customer's plan from the customer record when applying SLA policy.
- If the user asks for a summary, group the answer by customer, ticket, lead, or policy when that improves clarity.
- Mention relevant source identifiers such as customer ID, ticket ID, lead ID, or policy title when useful.
- Do not explain your process or list the context unless the user asks for sources.

Key facts extracted from the provided context:
{facts}

Full provided context:
{context}

User question:
{question}

Answer:
""".strip()
