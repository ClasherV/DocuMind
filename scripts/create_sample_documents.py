"""
Creates the DEMO documents in data/sample_documents/.

All text is fictional and written for this project. These files exist only so you can test
DocuMind immediately; they are NOT real user documents. The "scanned" files are clean,
computer-generated images - real scans (skewed, noisy, photographed) are harder for OCR.

Requires reportlab (pip install -r requirements-dev.txt). The finished files are already
included in the project, so you only need this script if you want to regenerate them.

Run:  python scripts/create_sample_documents.py
"""
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parent.parent / "data" / "sample_documents"
INVALID = OUT / "invalid"

ACADEMIC = [
    ("h", "Demo Institute of Technology - Academic Regulations (DEMO DOCUMENT)"),
    ("p", "This fictional document summarises the academic regulations that apply to all undergraduate students of the Demo Institute of Technology. It was written only for testing the DocuMind application."),
    ("h", "1. Attendance Requirement"),
    ("p", "Every student must maintain a minimum attendance of 75 percent in each course in order to be eligible to appear for the end-semester examination. Attendance is calculated separately for theory classes and laboratory sessions, and it is recorded by the course instructor at the start of every class."),
    ("p", "A student whose attendance falls between 65 and 75 percent may apply for condonation of the shortage. Condonation is granted only on medical grounds or for participation in official institute events, and the application must be supported by a valid certificate. Students with less than 65 percent attendance will not be allowed to take the examination and must repeat the course."),
    ("h", "2. Examinations and Assessment"),
    ("p", "Each course is assessed through continuous internal assessment worth 40 percent and an end-semester examination worth 60 percent. Internal marks are awarded for class tests, assignments and laboratory record books. A student must score at least 40 percent in the end-semester examination and 50 percent overall to pass a course."),
    ("p", "A student who fails a course may appear for a supplementary examination, which is conducted within two months of the declaration of results. Requests for re-evaluation of answer scripts must be submitted to the Controller of Examinations within ten days of the results being published."),
    ("h", "3. Grading System"),
    ("p", "Grades are awarded on a ten point scale. A score of 90 percent and above earns grade O (outstanding, 10 points), while scores between 80 and 89 percent earn grade A plus (9 points). The Cumulative Grade Point Average is computed by weighting each course grade by its credits. A minimum CGPA of 5.0 is required for the award of the degree."),
    ("h", "4. Academic Integrity"),
    ("p", "Plagiarism, copying in examinations and impersonation are serious offences. Students found guilty by the disciplinary committee may receive a zero in the course, suspension for one semester, or expulsion for repeated offences. All project reports are checked using similarity detection software before the final viva voce."),
    ("h", "5. Library, Hostel and Scholarships"),
    ("p", "The central library is open from 8 am to 10 pm on working days and provides access to textbooks, journals and digital resources. Hostel accommodation is allotted on the basis of distance from home and academic merit, and residents must follow the hostel rules regarding visitors and quiet hours."),
    ("p", "Merit scholarships are awarded to the top ten percent of students in each department. A scholarship is renewed every year provided the student maintains a CGPA of at least 8.0 and meets the attendance requirement."),
]

FINANCE = [
    ("h", "Fictional Quarterly Financial Report - Q3 (DEMO DOCUMENT)"),
    ("p", "This fictional report describes the financial position of Demo Retail Limited for the third quarter. All figures are invented for testing the DocuMind application."),
    ("h", "Income Statement Highlights"),
    ("p", "Total revenue for the quarter was 48.5 million rupees, an increase of eight percent over the previous quarter. Cost of goods sold was 29.1 million rupees, giving a gross profit of 19.4 million rupees and a gross margin of 40 percent. Operating expenses rose by nine percent to 11.2 million rupees, mainly because of higher payroll and rent."),
    ("p", "Net profit after tax was 5.3 million rupees compared with 4.6 million rupees in the previous quarter. Earnings per share improved to 2.65 rupees, and the board has recommended an interim dividend of one rupee per share."),
    ("h", "Balance Sheet and Cash Flow"),
    ("p", "Total assets at the end of the quarter stood at 210 million rupees, while total liabilities were 96 million rupees, leaving shareholder equity of 114 million rupees. The company repaid a term loan instalment of 3 million rupees and reduced its bank borrowings to 41 million rupees, which lowered the interest expense."),
    ("p", "Cash flow from operating activities was positive at 7.8 million rupees. Capital expenditure of 2.4 million rupees was spent on new store fittings. The closing cash balance improved to 18.6 million rupees, which strengthens the working capital position."),
    ("h", "Tax, Audit and Outlook"),
    ("p", "The provision for income tax was calculated at 25 percent of profit before tax. The audit committee reviewed the interim financial statements and reported no material misstatements. Management expects the budget for the next quarter to be met, although currency fluctuations and rising interest rates remain risks for the investment plan."),
]

TECHNOLOGY = [
    ("h", "Cloud Migration Design Note (DEMO DOCUMENT)"),
    ("p", "This fictional technical note describes how a small software team moves a web application from a single server to a cloud platform. It was written only for testing the DocuMind application."),
    ("h", "Architecture Overview"),
    ("p", "The current application is a monolithic Python service that stores data in a relational database on one virtual machine. The new design splits the system into three containerised microservices: an API gateway, an authentication service and a recommendation service that uses a machine learning model."),
    ("p", "Each service exposes a REST API that returns JSON and is deployed as a Docker container. A load balancer distributes incoming network traffic across several instances so that the platform stays available when a server fails."),
    ("h", "Security and Data"),
    ("p", "All traffic is protected with TLS encryption, and passwords are stored as salted hashes. Two-factor authentication is required for administrators. A firewall allows only the load balancer to reach the application servers, and the database accepts connections only from the private network."),
    ("p", "The database is migrated with an automated data pipeline that extracts the tables, validates them and loads them into a managed cloud database. Nightly backups are written to remote storage and tested every month."),
    ("h", "Deployment and Testing"),
    ("p", "Developers use Git for version control. A continuous integration pipeline compiles the code, runs unit tests and builds the container image after every commit. Debugging in production is done through centralised logs and monitoring dashboards that track latency, errors and memory usage."),
]

LEGAL_IMAGE_TEXT = (
    "SERVICE AGREEMENT (DEMO DOCUMENT)\n\n"
    "This Agreement is entered into between the Client and the Service Provider "
    "on the Effective Date. The Service Provider shall deliver the services "
    "described herein and the Client shall pay the fees agreed in writing.\n\n"
    "1. Confidentiality. Each party shall keep the other party's confidential "
    "information secret and shall not disclose it to any third party.\n\n"
    "2. Liability. The liability of the Service Provider under this Agreement "
    "shall not exceed the total fees paid by the Client.\n\n"
    "3. Termination. Either party may terminate this Agreement by giving thirty "
    "days written notice. Any breach of this Agreement entitles the other party "
    "to seek damages in court.\n\n"
    "4. Governing Law. All disputes shall be subject to the exclusive "
    "jurisdiction of the courts of India."
)

BUSINESS_SCAN_TEXT = (
    "MEMO - MARKETING STRATEGY (DEMO DOCUMENT)\n\n"
    "The marketing team will launch a new campaign to grow brand awareness "
    "and attract more customers in the next quarter. The sales director expects "
    "strong demand, so the company will hire two new account managers.\n\n"
    "Management asked each department to share its goals, timeline and "
    "customer feedback before the next meeting. The product team will "
    "prioritise features that clients requested most often."
)


def build_pdf(path: Path, blocks):
    styles = getSampleStyleSheet()
    story = []
    for kind, text in blocks:
        if kind == "h":
            story.append(Paragraph(text, styles["Heading2"]))
        else:
            story.append(Paragraph(text, styles["BodyText"]))
            story.append(Spacer(1, 6))
    SimpleDocTemplate(str(path), pagesize=A4, title=blocks[0][1]).build(story)


def load_font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_to_image(text: str, width: int = 1500, font_size: int = 34) -> Image.Image:
    """Render text on a white 'page' image, wrapping lines like a printed page."""
    font = load_font(font_size)
    lines = []
    for paragraph in text.split("\n"):
        lines.extend(textwrap.wrap(paragraph, width=62) or [""])
    line_height = int(font_size * 1.5)
    image = Image.new("RGB", (width, line_height * (len(lines) + 3)), "white")
    draw = ImageDraw.Draw(image)
    y = line_height
    for line in lines:
        draw.text((70, y), line, fill="black", font=font)
        y += line_height
    return image


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    INVALID.mkdir(parents=True, exist_ok=True)

    build_pdf(OUT / "demo_academic_regulations.pdf", ACADEMIC)       # digital PDF
    build_pdf(OUT / "demo_finance_report.pdf", FINANCE)              # digital PDF
    build_pdf(OUT / "demo_technology_note.pdf", TECHNOLOGY)          # digital PDF

    text_to_image(LEGAL_IMAGE_TEXT).save(OUT / "demo_legal_agreement.png")     # image -> OCR
    text_to_image(BUSINESS_SCAN_TEXT).save(OUT / "demo_scanned_business_memo.pdf")  # image-only PDF

    # Deliberately broken files for testing error handling
    (INVALID / "empty.pdf").write_bytes(b"")
    (INVALID / "corrupted.pdf").write_bytes(b"%PDF-1.4\nthis is not a real pdf body \x00\x01\x02 garbage")
    (INVALID / "not_an_image.png").write_bytes(b"this text file pretends to be a PNG image")
    (INVALID / "notes.txt").write_text("Plain text is not a supported upload type.", encoding="utf-8")
    Image.new("RGB", (800, 600), "white").save(INVALID / "blank_image.png")     # valid image, no text
    print("Sample documents written to", OUT)


if __name__ == "__main__":
    main()
