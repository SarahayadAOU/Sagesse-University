"""
Stage 2: Extract structured commitments from the SER sections using Claude API.
"""

import anthropic
import json
import os

API_KEY = os.environ.get("CLAUDE_API_KEY", "")
client = anthropic.Anthropic(api_key=API_KEY)
MODEL = "claude-haiku-4-5-20251001"   # cheapest available

INDICATORS = [
    {"id": "I01", "dim": "D1", "name": "Salary Fairness",
     "text": "Rate your satisfaction with the fairness of your salary relative to your job role."},
    {"id": "I02", "dim": "D1", "name": "Salary Competitiveness",
     "text": "Rate your satisfaction with the competitiveness of your salary relative to peer institutions."},
    {"id": "I03", "dim": "D1", "name": "Pay Transparency",
     "text": "Rate your satisfaction with the clarity and transparency of your work compensation."},
    {"id": "I04", "dim": "D1", "name": "Health Coverage",
     "text": "Rate your satisfaction with the health coverage benefits (insurance, health care center, etc.)."},
    {"id": "I05", "dim": "D1", "name": "Non-Monetary Benefits",
     "text": "Rate your satisfaction with Non-Monetary benefits including tuition discounts, transportation support, wellness programs, EV loans etc."},
    {"id": "I06", "dim": "D1", "name": "End of Service Compensation",
     "text": "Rate your satisfaction about the end of service compensation."},
    {"id": "I07", "dim": "D2", "name": "Strategic Clarity",
     "text": "Rate your satisfaction with the clarity of the university's strategic objectives."},
    {"id": "I08", "dim": "D2", "name": "Decision Transparency",
     "text": "Rate your satisfaction with how clearly management communicates its decisions and their underlying rationales."},
    {"id": "I09", "dim": "D2", "name": "Participation in Decisions",
     "text": "Rate your satisfaction with the opportunities to participate in decision-making processes within your office."},
    {"id": "I10", "dim": "D2", "name": "Empowerment",
     "text": "Rate your satisfaction with the extent of empowerment granted in carrying your responsibilities."},
    {"id": "I11", "dim": "D2", "name": "Responsiveness",
     "text": "Rate your satisfaction with the effectiveness of management in responding to academic operational challenges."},
    {"id": "I12", "dim": "D2", "name": "Workflow Efficiency",
     "text": "Rate your satisfaction with how effectively ULS current workflows support your responsibilities."},
    {"id": "I13", "dim": "D2", "name": "Conflict Resolution",
     "text": "Rate your satisfaction with how management addresses and resolves disputes between stakeholders."},
    {"id": "I14", "dim": "D3", "name": "Policy Communication",
     "text": "Rate your satisfaction with how well bylaws, regulations, policies and procedures are communicated and understood."},
    {"id": "I15", "dim": "D3", "name": "Clarity of Communication",
     "text": "Rate your satisfaction with the clarity of university-wide communications including announcements, events, and campus updates."},
    {"id": "I16", "dim": "D4", "name": "Workspace Comfort",
     "text": "Rate your satisfaction with the comfort of your workspace including ergonomics, lighting, noise levels, ventilation, and spatial layout."},
    {"id": "I17", "dim": "D4", "name": "Access to Equipment and Technology",
     "text": "Rate your satisfaction with the accessibility of equipment and technology needed to perform your job effectively."},
    {"id": "I18", "dim": "D5", "name": "Promotion Clarity",
     "text": "Rate your satisfaction with the clarity of promotion criteria and processes."},
    {"id": "I19", "dim": "D5", "name": "Training Effectiveness",
     "text": "Rate your satisfaction with the effectiveness of training programs."},
    {"id": "I20", "dim": "D5", "name": "Extra-Role Involvement",
     "text": "Rate your satisfaction with the opportunities to participate in initiatives beyond your formal job responsibilities."},
    {"id": "I21", "dim": "D5", "name": "Recognition",
     "text": "Rate your satisfaction with the recognition of your achievements and contributions."},
    {"id": "I22", "dim": "D6", "name": "Flexibility",
     "text": "Rate your satisfaction with the availability of flexible working arrangements."},
    {"id": "I23", "dim": "D6", "name": "Work-Life Balance",
     "text": "Rate your satisfaction with the support provided to balance professional responsibilities and personal life."},
    {"id": "I24", "dim": "D6", "name": "Burnout Risk",
     "text": "Rate your satisfaction with the support provided to manage stress and avoid emotional fatigue."},
    {"id": "I25", "dim": "D7", "name": "Collegiality",
     "text": "Rate your satisfaction with the collaboration and teamwork climate among colleagues."},
    {"id": "I26", "dim": "D7", "name": "Psychological Safety",
     "text": "Rate your satisfaction with the ability to voice ideas and raise concerns."},
    {"id": "I27", "dim": "D7", "name": "Diversity and Inclusion",
     "text": "Rate your satisfaction with the university's commitment to diversity, inclusion and respect for all."},
    {"id": "I28", "dim": "D8", "name": "Sense of Belonging",
     "text": "Rate your satisfaction with your sense of belonging and identification with ULS."},
    {"id": "I29", "dim": "D8", "name": "Family Enrollment Likelihood",
     "text": "Rate your satisfaction with your likelihood of recommending ULS or enrolling family members or children."},
    {"id": "I30", "dim": "D8", "name": "Brand Identity",
     "text": "Rate your satisfaction with your perception of ULS brand identity."},
]

SECTION_22 = """§2.2 Internal Organizational Structure and Governance
ULS governance ensures representation and active participation in decision-making of academic
and administrative staff. The structure relies on clear leadership roles, faculty governance,
financial oversight, and institutional decision-making bodies.
Vice-Presidents support the President by overseeing academic, research, and administrative
functions. They are appointed for variable terms and play a central role in policy implementation.
Each faculty is led by a Dean appointed for a renewable three-year term. The Faculty Council
includes elected representatives of academic staff and is responsible for curriculum development,
strategic planning, and quality assurance.
The University Council is the highest executive body. It establishes institutional policies,
approves academic programmes, and includes the elected representative of the full-time academic staff.
ULS has established entities to confirm its commitment to sustainability and quality."""

SECTION_325 = """§3.2.5 Transparency and Public Information
ULS is committed to transparency by ensuring comprehensive, accurate, and accessible information
about its academic programmes, policies, and activities is systematically communicated to all stakeholders.
Transparency is achieved through dissemination of ULS Bylaws, Academic Staff Manual, Rules and
Regulations for Academic Affairs, and institutional policies, made accessible on the university website.
Transparency supports an environment of trust and accountability, promoting a culture of integrity.
ULS provides multiple platforms for interaction including email, online portals, social media, and
face-to-face meetings. ULS maintains robust communication through SIS, university email, website,
and Moodle, ensuring timely responsiveness, feedback collection, and continuous improvement."""

SECTION_343 = """§3.4.3 Academic Staff: Recruitment, Performance Monitoring, Further Training
ULS ensures highly-qualified instructors through: hiring instructors with advanced degrees;
evaluating teaching methods; encouraging continuous learning and professional development;
supporting research activities; retaining talented instructors by offering growth opportunities,
recognizing contributions, and creating a positive work environment.

§3.4.3.1 Recruitment
ULS recruitment philosophy relies on a merit-based approach, prioritizing academic qualifications,
teaching ability, research experience, and professional accomplishments. The university adopts a
clear recruitment policy ensuring selection of the most qualified candidates. Recruitment includes
an ad-hoc committee review, interviews, and demo-lectures for candidates with limited teaching experience.
ULS prioritizes internationally recognized degrees and recruits senior experts such as judges and CEOs.

§3.4.3.2 Performance Monitoring
ULS implements a structured, multi-source performance monitoring system. Students complete mandatory
semester evaluations assessing course content, teaching delivery, and overall satisfaction. For
promotion, staff submit a comprehensive file including student evaluations, teaching portfolio,
research output, and community service contributions. Evaluation results are shared and discussed
in performance review meetings.

§3.4.3.3 Further Training and Research Support
The CTL organizes training sessions covering pedagogy, technology, student engagement, and mental
health in the workplace. ULS supports research through sabbatical leaves, international mobility,
access to research databases, conference participation, and load reductions by rank. The university
encourages publication in peer-reviewed, internationally recognized journals.

§3.4.3.4 Measures to Retain Qualified Academic Staff
ULS offers a comprehensive compensation and benefits package including competitive transparent
salaries based on rank, healthcare coverage, seniority-based end-of-service indemnities, and
educational funding for instructors' children. Transportation and accommodation allowances are
provided for professional mobility. ULS values long-term commitment through seniority recognition
with clearly defined contractual terms. ULS provides access to workshops, international training
programs, and academic mobility initiatives to ensure professional growth."""

EXTRACTION_PROMPT = """You are an institutional quality-assurance analyst. Extract from the following
Self-Evaluation Report (SER) section every distinct, verifiable COMMITMENT made by the university
toward its employees (academic and administrative staff).

A commitment is an explicit or strongly implied promise about what the institution provides, ensures,
maintains, supports, or guarantees for its staff.

For EACH commitment output EXACTLY this JSON structure:
{"id": "C00", "theme": "<2-4 word theme>", "section": "<section>", "commitment": "<one sentence ≤25 words starting with action verb>", "keywords": ["kw1","kw2","kw3"]}

Output ONLY a valid JSON array. No markdown fences, no explanation.

SER TEXT:
"""

def extract_commitments_from_section(section_text):
    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": EXTRACTION_PROMPT + section_text}]
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        start = raw.find("[")
        end = raw.rfind("]") + 1
        raw = raw[start:end]
    return json.loads(raw)

def main():
    all_commitments = []
    counter = 1

    for sec_id, sec_text in [("§2.2", SECTION_22), ("§3.2.5", SECTION_325), ("§3.4.3", SECTION_343)]:
        print(f"  Extracting from {sec_id}...", flush=True)
        comms = extract_commitments_from_section(sec_text)
        for c in comms:
            c["id"] = f"C{counter:02d}"
            counter += 1
        all_commitments.extend(comms)
        print(f"    -> {len(comms)} commitments")

    # Save indicators too
    with open("/home/sandbox/experiments/commitments.json", "w") as f:
        json.dump(all_commitments, f, indent=2)
    with open("/home/sandbox/experiments/indicators.json", "w") as f:
        json.dump(INDICATORS, f, indent=2)

    print(f"\nTotal: {len(all_commitments)} commitments saved.")
    for c in all_commitments:
        print(f"  {c['id']} [{c['section']}] {c['theme']}: {c['commitment'][:70]}")

if __name__ == "__main__":
    main()
