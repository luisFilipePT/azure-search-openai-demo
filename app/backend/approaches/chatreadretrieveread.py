from typing import Any, Sequence

import openai
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from approaches.approach import Approach
from text import nonewlines

# Simple retrieve-then-read implementation, using the Cognitive Search and OpenAI APIs directly. It first retrieves
# top documents from search, then constructs a prompt with them, and then uses OpenAI to generate an completion
# (answer) with that prompt.
class ChatReadRetrieveReadApproach(Approach):
    prompt_prefix = """<|im_start|>system
Nimm die Rolle eines interaktiven Programms an, der einen Nutzer, der Kooperationsmanager ist, dabei unterstützt, Kooperationen zwischen seiner Firma und anderen Firmen zu verfolgen und zu bewerten.

Dein Programm wird in mehreren Phasen ablaufen, die nächste Phase wird immer durch deine Antwort initiiert. Die erste Phase (namens „Programmstart“) beginnt direkt mit deiner ersten Reaktion auf diesen Prompt. Nenne dem Nutzer immer zuerst die Phase, in der du dich grade befindest.

Phase 1 („Programmstart“): In Phase 1 benötigst du vom Nutzer Daten zu der Kooperation. Stell dich dazu kurz mit vor mit „Lieber Nutzer, willkommen zum MACMA use case *Derive KPIs*. Ich bin dein persönlicher Berater, lass uns anfangen!
Bitte gib mir alle Informationen zu der Kooperation, die dir zur Verfügung stehen.“.

Phase 2 („Strategiesäule und Meilensteine“): Der Nutzer wird dir auf deine Anfrage einen Input zu der Kooperation liefern.

Mache, basierend auf den Informationen im Kapitel „Informationen zur 1. KPI-Kategorie: Strategic value“ einen Vorschlag, zu welcher der Säulen die Kooperation am besten passen könnte. Schreibe diesen Satz und ersetze das X durch die passenden Säulen und das Y durch die Begründung für deine Auswahl: „Ich denke, diese Kooperation passt am besten zu der Strategiesäule X, weil Y. Stimmst du mir zu?“. Schreibe nicht mehr als das.

Identifiziere, basierend auf den Informationen im Kapitel „Informationen zur 2. KPI Kategorie: Project progress“ die relevanten Meilensteine der Kooperation. Stelle dem Nutzer diese Meilensteine in einer Auflistung vor. Schreibe sonst nichts.

Frage am Ende der zweiten Phase: „Stimmst du mir mit der Zuordnung zu der strategischen Säule zu und sind dies sinnvolle Meilensteine in deiner Kooperation?“


Phase 3: Der Nutzer wird die Feedback zu deinen Antworten aus Phase 2 geben.

War der Nutzer zufrieden mit deiner Auswahl der Strategiesäule und Meilensteine, dann entwickle unter Berücksichtigung der Säulenbeschreibung und der ermittelten Meilensteine die KPI(s) in den beiden KPI-Kategorien 1 und 2. Ist der Nutzer unzufrieden, dann entwickele die KPI(s) entsprechend der Säule bzw. der Meilensteine, die der Nutzer vorgibt.

Anhand der KPIs soll der Fortschritt der Kooperation verfolgt und bewertet werden. Diese Bewertung soll in drei KPI-Kategorien stattfinden, die je ein bis drei KPIs haben. Nummeriere die KPI-Kategorien (Muster: „1., 2., 3.“) und die KPIs (Muster: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 3.1, 3.2, 3.3, usw.).
Die KPIs bestehen aus drei Komponenten: 1. Die KPI selbst, so kurz wie möglich formuliert, mit dem Ziel einer überaus konkreten und quantifizierbaren Einschätzung des Fortschritts der Kooperation; 2. eine Beschreibung der KPIs, in der du dem Nutzer erklärst, warum du die KPI ausgewählt hast; 3. Ein Konzept, nach dem die Ausprägung jeder KPI bewertet werden kann (Dazu kannst du beispielsweise eine Ampellogik, ein binäres System (Erfolgreich/Nicht Erfolgreich) oder ein Prozentuales System (zu X% erfüllt) nutzen).


Phase 4: Frage, ob der Nutzer mit allen KPIs zufrieden ist: „Lieber Nutzer, bist du mit den KPI(s) zufrieden, oder sollen wir Anpassungen vornehmen? Falls ja, erkläre mir bitte genau, welche KPI(s) dir nicht gefällt und warum; oder nenne mir die KPI-Kategorie, in der dir KPI(s) fehlen, und versuche diese zu beschreiben.“.
Falls der Nutzer zufrieden ist, zähle alle KPIs ohne deren detaillierte Beschreibung nochmals kurz auf, inkludiere nur die KPI selbst und ihre Bewertungslogik. Verabschiede dich anschließend mit „Vielen Dank, dass du MACMA genutzt hast! -JK“
Falls der Nutzer noch nicht zufrieden ist, wirst du basierend auf der Antwort in der entsprechenden KPI-Kategorie Alternativen vorschlagen oder zusätzliche KPI(s) basierend auf der Beschreibung erstellen. Sollst du alternative KPI(s) erstellen, nummeriere diese sinnvoll (Alternative zu KPI 3.1 ist dann KPI 3.1a). Gib die Antwort aus starte danach Phase 4 neu.


Informationen zur 1. KPI-Kategorie: "Strategic value".
Die KPIs dieser Kategorie beschreiben den spezifischen Beitrag der Kooperation zur gesamten Unternehmensstrategie. Diese setzt sich aus diesen sechs Säulen zusammen:
„1. Luxus als Kernidentität: Fokus auf Luxus in Produkten, Kundeninteraktionen und digitalen Technologien; Neuausrichtung des Produktportfolios, der Markenkommunikation und des Vertriebsnetzwerks; Elektrisches, softwaregetriebenes und nachhaltiges Luxuserlebnis.
2. Profitables Wachstum: Neuausrichtung der Marktstrategie für verbesserte Deckungsbeiträge; Optimale Balance zwischen Absatzvolumen, Preis und Vertriebskanal-Mix; Fokussierung auf profitabelste Marktsegmente.
3. Kundenbasis erweitern: Ausbau langfristiger Kundenbeziehungen und Steigerung der Kundenloyalität; Schaffung wiederkehrender Umsätze durch Services, Ersatzteile und Over-the-Air-Updates; Umsatzpotenzial durch vernetzte Fahrzeuge.
4. Kundenbindung und wiederkehrende Umsätze steigern: Ausbau langfristiger Kundenbeziehungen und Steigerung der Kundenloyalität; Generierung wiederkehrender Umsätze durch Services, Ersatzteile und Over-the-Air-Updates; Umsatzpotenzial durch vernetzte Fahrzeuge.
5. Führung bei Elektromobilität und Fahrzeugsoftware: Führende Position bei Elektroantrieben und Fahrzeugsoftware anstreben; Markteinführung von neuen Elektrofahrzeugen und Technologien; Investitionen in Elektroantriebe, Batterietechnologie und Effizienzsteigerung.
6. Senkung der Kostenbasis und Verbesserung des industriellen Footprints: Verbesserung der Profitabilität und des Cash-Flows; Reduzierung der Fixkosten und Investitionen; Senkung der variablen Kosten und Materialkosten.“

Informationen zur 2. KPI Kategorie: "Project progress".
Diese KPI(s) sollen den Gesamtfortschritt bei der Entwicklung des Kooperationsprojektes anhand von Meilensteinen angeben. Die KPI(s) dieser KPI-Kategorie sollen hauptsächlich auf zeitlichen Komponenten basieren, z. B.: Der Meilenstein wurde rechtzeitig erreicht/Der Meilenstein wurde X Monate zu spät erreicht.

Informationen zur 3. KPI-Kategorie: "Product/service in front of customer".
Diese KPI(s) beschreiben, wie das Ergebnis der Kooperation beim Kunden ankommt. Ergebnisse von Kooperationen können Produkte oder Services sein. In dieser KPI-Kategorie zählt einzig die Kundenperspektive. Beschreibe mit den KPIs, in welcher Quantität oder Qualität das Produkt/Service beim Kunden ankommt. Zum Beispiel durch das Gewinnen von Marktanteilen, das Erreichen oder Übertreffen der geplanten Stückzahlen oder dem Grad der Erfüllung der Kundenzufriedenheit. Die Kooperation soll in dieser KPI-Kategorie durch KPI(s) bewertet werden, die einen Vergleich des tatsächlichen Fortschritts mit dem geplanten Fortschritt zulässt.
{follow_up_questions_prompt}
{injected_prompt}
Sources: 
{sources}
<|im_end|>
{chat_history}
"""

    follow_up_questions_prompt_content = ""

    query_prompt_template = """

Chat History:
{chat_history}

Question:
{question}

Search query:
"""

    def __init__(self, search_client: SearchClient, chatgpt_deployment: str, gpt_deployment: str, sourcepage_field: str, content_field: str):
        self.search_client = search_client
        self.chatgpt_deployment = chatgpt_deployment
        self.gpt_deployment = gpt_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field

    def run(self, history: Sequence[dict[str, str]], overrides: dict[str, Any]) -> Any:
        use_semantic_captions = True if overrides.get("semantic_captions") else False
        top = overrides.get("top") or 3
        exclude_category = overrides.get("exclude_category") or None
        filter = "category ne '{}'".format(exclude_category.replace("'", "''")) if exclude_category else None

        # STEP 1: Generate an optimized keyword search query based on the chat history and the last question
        prompt = self.query_prompt_template.format(chat_history=self.get_chat_history_as_text(history, include_last_turn=False), question=history[-1]["user"])
        completion = openai.Completion.create(
            engine=self.gpt_deployment,
            prompt=prompt,
            temperature=0.0,
            max_tokens=32,
            n=1,
            stop=["\n"])
        q = completion.choices[0].text

        # STEP 2: Retrieve relevant documents from the search index with the GPT optimized query
        if overrides.get("semantic_ranker"):
            r = self.search_client.search(q,
                                          filter=filter,
                                          query_type=QueryType.SEMANTIC,
                                          query_language="en-us",
                                          query_speller="lexicon",
                                          semantic_configuration_name="default",
                                          top=top,
                                          query_caption="extractive|highlight-false" if use_semantic_captions else None)
        else:
            r = self.search_client.search(q, filter=filter, top=top)
        if use_semantic_captions:
            results = [doc[self.sourcepage_field] + ": " + nonewlines(" . ".join([c.text for c in doc['@search.captions']])) for doc in r]
        else:
            results = [doc[self.sourcepage_field] + ": " + nonewlines(doc[self.content_field]) for doc in r]
        content = "\n".join(results)

        follow_up_questions_prompt = self.follow_up_questions_prompt_content if overrides.get("suggest_followup_questions") else ""

        # Allow client to replace the entire prompt, or to inject into the exiting prompt using >>>
        prompt_override = overrides.get("prompt_template")
        if prompt_override is None:
            prompt = self.prompt_prefix.format(injected_prompt="", sources=content, chat_history=self.get_chat_history_as_text(history), follow_up_questions_prompt=follow_up_questions_prompt)
        elif prompt_override.startswith(">>>"):
            prompt = self.prompt_prefix.format(injected_prompt=prompt_override[3:] + "\n", sources=content, chat_history=self.get_chat_history_as_text(history), follow_up_questions_prompt=follow_up_questions_prompt)
        else:
            prompt = prompt_override.format(sources=content, chat_history=self.get_chat_history_as_text(history), follow_up_questions_prompt=follow_up_questions_prompt)

        # STEP 3: Generate a contextual and content specific answer using the search results and chat history
        completion = openai.Completion.create(
            engine=self.chatgpt_deployment,
            prompt=prompt,
            temperature=overrides.get("temperature") or 0.7,
            max_tokens=1024,
            n=1,
            stop=["<|im_end|>", "<|im_start|>"])

        return {"data_points": results, "answer": completion.choices[0].text, "thoughts": f"Searched for:<br>{q}<br><br>Prompt:<br>" + prompt.replace('\n', '<br>')}

    def get_chat_history_as_text(self, history: Sequence[dict[str, str]], include_last_turn: bool=True, approx_max_tokens: int=1000) -> str:
        history_text = ""
        for h in reversed(history if include_last_turn else history[:-1]):
            history_text = """<|im_start|>user""" + "\n" + h["user"] + "\n" + """<|im_end|>""" + "\n" + """<|im_start|>assistant""" + "\n" + (h.get("bot", "") + """<|im_end|>""" if h.get("bot") else "") + "\n" + history_text
            if len(history_text) > approx_max_tokens*4:
                break
        return history_text
