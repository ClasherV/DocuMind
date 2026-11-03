"""
Builds data/training_data.csv  -  a SYNTHETIC DEMO DATASET.

IMPORTANT - read this before quoting any accuracy number:
  * Every sentence below was written by hand for this project. It is NOT real-world data.
  * Documents are made by randomly combining 3-6 sentences of one category (plus occasional
    "noise" sentences from other categories so the task is not trivially easy).
  * Training documents and test documents are built from DIFFERENT sentence pools, so the test
    split never contains a sentence seen in training. Even so, the vocabulary is far cleaner
    than real documents, so accuracy measured on this data is OPTIMISTIC.
  * To make the model genuinely better, replace data/training_data.csv with real labelled
    documents (columns: text,label,split) and retrain.

Run:  python scripts/generate_demo_dataset.py
"""
import csv
import random
from pathlib import Path

SEED = 42
DOCS_PER_CATEGORY = {"train": 80, "test": 25}
NOISE_PROBABILITY = 0.20

# Each category: (training sentences, held-out test sentences)
POOLS = {
    "Education": (
        [
            "Students must maintain a minimum attendance of seventy-five percent to be eligible for the semester examinations.",
            "The university offers undergraduate and postgraduate programmes across engineering, science and the humanities.",
            "Each course syllabus lists the learning outcomes, weekly lecture topics and recommended textbooks.",
            "Internal assessment marks are awarded for assignments, class tests and laboratory performance.",
            "The academic calendar specifies the dates for admissions, lectures, examinations and vacations.",
            "A scholarship is available for students who secure a high grade point average in the previous semester.",
            "Faculty members hold weekly office hours to help students with doubts about the curriculum.",
            "Candidates who fail a subject may apply for a supplementary examination after the results are declared.",
            "The library provides students with access to journals, reference books and digital learning resources.",
            "Every student must submit the project report to the department before the final viva voce.",
            "The professor explained the lesson plan and distributed the reading list for the new term.",
            "Admission to the degree programme is based on entrance test scores and the previous qualifying examination.",
            "Teachers encourage classroom discussion so that pupils understand concepts instead of memorising answers.",
            "The college hostel provides accommodation to enrolled students during the academic year.",
            "Semester results are published on the university portal along with the cumulative grade point average.",
            "Students found guilty of plagiarism in assignments will face disciplinary action from the academic council.",
            "The department organises seminars and workshops to improve the research skills of postgraduate scholars.",
            "Attendance shortage may be condoned only on medical grounds with a valid certificate from a doctor.",
            "The school introduced a new mathematics curriculum focused on problem solving and critical thinking.",
            "Laboratory sessions are compulsory and students must complete all experiments in the record book.",
            "The examination hall rules prohibit mobile phones, calculators and unauthorised study material.",
            "Course credits are earned by passing the theory paper and the associated practical component.",
            "The principal addressed the graduating batch at the convocation ceremony held in the auditorium.",
            "Enrolment for elective subjects opens at the start of every semester through the academic office.",
            "Tutorial classes revise the lectures and give learners extra practice with sample question papers.",
            "The thesis supervisor reviews the dissertation draft and recommends corrections before submission.",
            "Undergraduate learners are assessed through a mix of midterm tests, quizzes and a final examination.",
            "The student council works with the dean to improve campus facilities and the learning environment.",
            "Teacher training programmes help instructors apply modern pedagogy in the classroom.",
            "Grading follows a ten point scale where a score above nine indicates an outstanding performance.",
        ],
        [
            "Hall tickets are issued only to students who satisfy the attendance and fee requirements of the institution.",
            "The department of computer science revised its curriculum and added new elective courses for the coming term.",
            "Pupils in the primary school take part in reading clubs and science exhibitions throughout the academic year.",
            "A re-evaluation request must be submitted to the controller of examinations within ten days of the result.",
            "The convocation for the graduating class will be held after the final semester marks are verified.",
            "Lecturers record attendance during every class and upload the data to the university portal each week.",
            "Research scholars present their progress to the doctoral committee at the end of every academic year.",
            "The orientation programme introduces freshmen to the campus, the syllabus and the examination system.",
            "Merit scholarships are renewed annually when a student maintains the required grade point average.",
            "Assignments submitted after the deadline lose marks unless the instructor has approved an extension.",
        ],
    ),
    "Business": (
        [
            "The marketing team presented a new strategy to increase brand awareness among younger customers.",
            "Our company plans to expand into three new regional markets during the next financial year.",
            "The quarterly meeting reviewed sales performance, customer feedback and the product launch schedule.",
            "Management approved the hiring of additional sales executives to support the growing client base.",
            "The supply chain team negotiated better delivery terms with vendors to reduce operational costs.",
            "A strong value proposition helps a startup differentiate itself from competitors in a crowded market.",
            "The project manager shared the timeline, deliverables and responsibilities of each department.",
            "Customer satisfaction scores improved after the support team introduced a faster response process.",
            "The board discussed a merger with a smaller competitor to strengthen the company market share.",
            "Our business model relies on subscription customers who renew their service plans every year.",
            "The human resources department updated the employee onboarding process and performance review cycle.",
            "Executives set key performance indicators to measure progress toward the organisation's goals.",
            "The retailer opened a flagship store and launched a loyalty programme for repeat customers.",
            "Stakeholders asked for a clearer roadmap showing how the new product will reach the market.",
            "The consulting firm advised the client on improving productivity and streamlining its operations.",
            "A successful entrepreneur builds a capable team, listens to customers and adapts quickly to change.",
            "The sales director forecasted strong demand for the upcoming holiday season across all channels.",
            "Corporate partnerships allow both companies to share distribution networks and reach more clients.",
            "The manager held a workshop to improve collaboration between the marketing and sales teams.",
            "Market research showed that consumers prefer affordable products with fast delivery.",
            "The company announced a rebranding campaign with a new logo and a refreshed corporate identity.",
            "Operational efficiency improved when the factory reorganised its production workflow and inventory.",
            "The proposal outlines the target audience, competitive landscape and go-to-market plan.",
            "Our organisation values leadership, teamwork and long-term relationships with clients and partners.",
            "The chief executive thanked employees for their effort during a challenging year for the industry.",
            "Franchise owners receive training, marketing support and a proven business playbook.",
            "A customer relationship management system helps the sales team track leads and follow up on deals.",
            "The firm outsourced its customer service department to cut costs and extend support hours.",
            "Meeting minutes record the action items assigned to each team member before the next review.",
            "The product team prioritised features requested by enterprise customers for the next release cycle.",
        ],
        [
            "The agenda for tomorrow's meeting includes the sales pipeline, hiring plans and the new vendor contract.",
            "Our competitors are launching similar products, so the marketing department must sharpen its positioning.",
            "The startup pitched its business plan to investors and described its customer acquisition strategy.",
            "Team leaders were asked to submit a quarterly report on productivity, targets and customer complaints.",
            "The company will open regional offices to serve clients faster and strengthen its brand presence.",
            "A new manager was appointed to lead the operations team and improve delivery timelines.",
            "The partnership agreement between the two firms focuses on joint marketing and shared distribution.",
            "Employees attended a leadership seminar about motivation, communication and performance management.",
            "The organisation restructured its departments to make decision making faster and more transparent.",
            "Retail sales rose after the promotional campaign attracted new customers to the online store.",
        ],
    ),
    "Finance": (
        [
            "The bank raised its interest rate on home loans by fifty basis points this quarter.",
            "Total revenue for the year was eighty million rupees, with a net profit margin of twelve percent.",
            "The balance sheet shows assets, liabilities and shareholder equity at the end of the reporting period.",
            "Investors diversify their portfolio across equities, bonds and mutual funds to manage risk.",
            "Income tax returns must be filed before the deadline to avoid penalties and interest charges.",
            "The audit committee reviewed the financial statements and found no material misstatements.",
            "Monthly expenses were reduced after the budget committee cut discretionary spending.",
            "The central bank increased the repo rate to control inflation and stabilise the currency.",
            "A fixed deposit offers a guaranteed return at a predetermined interest rate for a fixed tenure.",
            "The company reported strong quarterly earnings, and its share price rose on the stock exchange.",
            "Credit card statements show the outstanding balance, the minimum payment due and the billing date.",
            "Depreciation is charged on machinery over its useful life and recorded in the profit and loss account.",
            "The loan was approved with a repayment schedule of sixty equal monthly instalments.",
            "Cash flow from operating activities was positive, which improved the company's liquidity position.",
            "The insurance premium is payable annually and covers the policyholder against financial loss.",
            "Dividends are distributed to shareholders out of the retained earnings of the company.",
            "The finance department prepared the annual budget with projected income, costs and capital expenditure.",
            "Mutual fund returns depend on market volatility, the expense ratio and the investment horizon.",
            "The accountant reconciled the bank statement with the ledger and corrected the pending entries.",
            "Gross domestic product grew by six percent while the fiscal deficit narrowed compared with last year.",
            "Mortgage payments consist of principal and interest, and early repayment may attract a prepayment fee.",
            "A savings account earns interest that is credited to the account holder every quarter.",
            "The treasury invested surplus funds in short-term government securities to preserve capital.",
            "Net present value and internal rate of return are used to evaluate long-term investment projects.",
            "The invoice lists the taxable amount, goods and services tax, and the total payable.",
            "Foreign exchange losses reduced the profit because the rupee weakened against the dollar.",
            "Venture capital funding was raised in two rounds to finance product development and hiring.",
            "Payroll expenses, employee provident fund contributions and tax deductions are reported each month.",
            "The financial analyst recommended buying the stock because of its low debt and rising dividend yield.",
            "Working capital is the difference between current assets such as cash and current liabilities.",
        ],
        [
            "Operating expenses increased by nine percent, which reduced the net income reported for the fiscal year.",
            "The pension fund allocates contributions to bonds and equities to earn a steady long-term return.",
            "Borrowers with a high credit score usually qualify for a lower interest rate on personal loans.",
            "The statement of cash flows reconciles the opening balance with the closing balance of cash holdings.",
            "Capital gains from the sale of shares are taxed according to the holding period of the investment.",
            "The bank charges a processing fee on loans and deducts it from the sanctioned amount.",
            "Shareholders approved the final dividend after reviewing the audited annual accounts of the company.",
            "Inflation erodes the purchasing power of savings unless the interest earned is higher than price growth.",
            "The quarterly profit and loss statement shows revenue, cost of goods sold and operating margin.",
            "A budget deficit occurs when government spending exceeds the tax revenue collected during the year.",
        ],
    ),
    "Technology": (
        [
            "The application programming interface returns data in JSON format and requires an authentication token.",
            "A relational database stores records in tables and uses SQL queries to retrieve information.",
            "Machine learning models learn patterns from training data and make predictions on new inputs.",
            "The server runs on a cloud platform and scales automatically when network traffic increases.",
            "Developers use version control to track changes in the source code and collaborate on software.",
            "Encryption protects sensitive data in transit by converting it into unreadable ciphertext.",
            "The mobile app was built with a modern framework and deployed to both Android and iOS.",
            "Neural networks consist of layers of interconnected nodes that transform input features.",
            "A firewall monitors incoming network packets and blocks suspicious connections to the system.",
            "The software update fixes a security vulnerability and improves the performance of the operating system.",
            "Python is a popular programming language for data analysis, automation and web development.",
            "The microservices architecture splits a large application into small independent services.",
            "Cybersecurity teams scan the infrastructure for malware and unpatched software.",
            "The processor has eight cores, sixteen gigabytes of memory and a fast solid state drive.",
            "Continuous integration pipelines compile the code and run automated tests after every commit.",
            "Computer vision algorithms detect objects in images captured by the camera sensor.",
            "The router assigns an IP address to each device that connects to the wireless network.",
            "Containerisation packages an application with its libraries so it runs the same on any machine.",
            "Natural language processing allows computers to analyse text, translate languages and answer questions.",
            "The data pipeline extracts logs from the servers, transforms them and loads them into a warehouse.",
            "Debugging involves finding the cause of a runtime error and fixing the faulty function.",
            "Blockchain technology records transactions in a distributed ledger that is difficult to alter.",
            "The web browser sends an HTTP request to the server and renders the HTML response.",
            "Artificial intelligence systems require large datasets, powerful GPUs and careful model evaluation.",
            "A load balancer distributes requests across several servers to improve availability.",
            "The algorithm sorts the array in logarithmic time using a divide and conquer approach.",
            "Internet of things devices send sensor readings to the cloud over a wireless connection.",
            "The operating system manages memory, processes and hardware drivers for the computer.",
            "Software engineers write unit tests to verify that each function behaves as expected.",
            "Two-factor authentication adds a one-time password to the login process for extra security.",
        ],
        [
            "The new smartphone features a faster chip, an improved camera and longer battery life.",
            "The backend service exposes REST endpoints and stores user sessions in an in-memory cache.",
            "Training a deep learning model on a graphics card is much faster than using a standard processor.",
            "The system administrator configured the network switch and updated the server firmware last night.",
            "Open source libraries let programmers reuse tested code instead of writing every function from scratch.",
            "A phishing attack tricks users into revealing passwords through a fake website or email.",
            "The compiler translates source code into machine instructions that the processor can execute.",
            "Cloud storage keeps backups of files on remote servers that can be accessed from anywhere.",
            "The chatbot uses a transformer model to understand the intent behind each message.",
            "Hardware engineers designed a compact circuit board with a low power microcontroller.",
        ],
    ),
    "Legal": (
        [
            "This agreement is entered into between the first party and the second party on the effective date.",
            "The tenant shall pay rent on or before the fifth day of each month as stipulated in the lease.",
            "The court dismissed the petition and directed the respondent to pay the costs of the proceedings.",
            "Either party may terminate this contract by giving thirty days written notice to the other party.",
            "The contractor shall indemnify and hold harmless the client against any claims, losses or damages.",
            "All disputes arising under this agreement shall be subject to the exclusive jurisdiction of the courts.",
            "The plaintiff filed a lawsuit alleging breach of contract and seeking compensation for damages.",
            "Confidential information disclosed under this agreement shall not be shared with any third party.",
            "The defendant denies all allegations and requests that the case be dismissed with prejudice.",
            "Pursuant to the statute, the licence holder must comply with all applicable laws and regulations.",
            "The liability of the supplier shall not exceed the total fees paid under this agreement.",
            "A witness gave sworn testimony before the judge, and the advocate cross-examined him.",
            "The copyright in the work remains the property of the author unless assigned in writing.",
            "The arbitration clause requires the parties to resolve their disagreement before a neutral arbitrator.",
            "This will revokes all earlier wills and appoints the executor to distribute the estate to the heirs.",
            "The employer shall not disclose the personal data of employees except as required by law.",
            "Force majeure events excuse a party from performance when obligations become impossible.",
            "The judgment of the high court was appealed to the supreme court on a question of law.",
            "The buyer acknowledges that the seller makes no warranties, express or implied, regarding the goods.",
            "Any amendment to this deed must be in writing and signed by all the parties concerned.",
            "The accused was granted bail subject to conditions set by the magistrate.",
            "The licensor grants the licensee a non-exclusive, non-transferable right to use the trademark.",
            "In the event of default, the lender may enforce the security and recover the outstanding amount.",
            "The affidavit states that the facts mentioned herein are true to the best of the deponent's knowledge.",
            "The governing law of this contract shall be the law of India, without regard to conflict principles.",
            "Whereas the parties wish to set out the terms and conditions of their relationship, they agree as follows.",
            "The trustee holds the property for the benefit of the beneficiaries in accordance with the trust deed.",
            "The notice of eviction was served on the tenant in accordance with the rent control act.",
            "A breach of the non-compete clause entitles the company to seek an injunction from the court.",
            "The constitution guarantees fundamental rights, and the judiciary may strike down unlawful legislation.",
        ],
        [
            "The undersigned parties hereby agree that this contract is binding and enforceable under applicable law.",
            "The lawyer drafted a legal notice demanding payment and warning of proceedings in the district court.",
            "The warranty is void if the buyer alters the product, and the seller bears no further liability.",
            "The tribunal heard arguments from both counsels and reserved its verdict for a later date.",
            "The licensee shall not sublicense, assign or transfer any rights granted under this licence agreement.",
            "This non-disclosure agreement remains in force for five years after the termination of the contract.",
            "The petitioner approached the court seeking a writ against the unlawful order of the authority.",
            "The clause is severable, and the invalidity of one provision does not affect the remaining terms.",
            "Under the act, the offender is liable to imprisonment, a fine or both upon conviction.",
            "The landlord reserves the right to inspect the premises after giving reasonable written notice.",
        ],
    ),
    "General": (
        [
            "We spent the weekend hiking in the hills and enjoyed a picnic by the lake.",
            "The weather forecast predicts light rain in the morning and clear skies by the evening.",
            "Grandmother's recipe uses fresh tomatoes, garlic and basil simmered slowly in a heavy pan.",
            "The local football team won the match with a late goal in the final minutes.",
            "Regular exercise, enough sleep and a balanced diet help people stay healthy and energetic.",
            "The family planned a holiday to the beach and booked a hotel near the old town.",
            "Our neighbour adopted a puppy that loves chasing butterflies in the garden.",
            "The new movie features beautiful scenery, a gripping story and a memorable soundtrack.",
            "Planting roses in spring gives colourful flowers that attract bees and butterflies all summer.",
            "The festival fills the streets with music, lanterns, sweets and cheerful crowds.",
            "He enjoys painting landscapes, reading novels and cooking dinner for his friends on Sundays.",
            "The museum exhibition showed ancient pottery, old photographs and traditional costumes.",
            "A long walk after dinner helps digestion and clears the mind before bedtime.",
            "The train journey through the mountains offered stunning views of rivers and pine forests.",
            "She baked a chocolate cake for her brother's birthday and decorated it with strawberries.",
            "The cricket tournament drew large crowds, and the final was played under floodlights.",
            "Morning yoga and a cup of green tea became part of her relaxing daily routine.",
            "The children built a sandcastle, collected shells and splashed in the waves all afternoon.",
            "The town market sells fresh vegetables, handmade crafts and warm bread every Saturday.",
            "They watched the sunset from the terrace while the cat slept on the sofa.",
            "The photographer captured a rare bird perched on a branch at dawn.",
            "Autumn leaves turned golden and red, and the park looked like a painting.",
            "Our road trip included a stop at a roadside cafe famous for its homemade pie.",
            "The marathon runner trained every morning and finished the race in under four hours.",
            "The bookshop on the corner has a cosy reading nook and an old wooden staircase.",
            "Birthday guests played board games, sang songs and shared stories late into the night.",
            "The gardener trimmed the hedges, watered the herbs and swept the garden path.",
            "A gentle breeze rustled the trees as the family sat outside drinking lemonade.",
            "The singer performed her favourite songs at an open air concert on a warm evening.",
            "Winter mornings are perfect for hot soup, a warm blanket and a good book.",
        ],
        [
            "The children laughed as they flew a colourful kite on the windy hilltop.",
            "A cup of ginger tea and some quiet music made the rainy afternoon feel cosy.",
            "The tourists wandered through narrow streets, tasted street food and bought souvenirs.",
            "Her grandfather tells wonderful stories about fishing trips on the river when he was young.",
            "The swimming team practised early in the morning before the annual championship.",
            "We adopted a kitten, and it spends all day playing with a ball of wool.",
            "The sunny weekend was ideal for cycling around the village and visiting the farm.",
            "The comedy show had the audience laughing from the first scene to the last.",
            "Fresh bread, jam and a glass of milk make a simple and happy breakfast.",
            "The couple celebrated their anniversary with a quiet dinner and a walk along the beach.",
        ],
    ),
}


def build_document(rng: random.Random, own_pool, other_pools):
    """Combine 3-6 random sentences of one category; occasionally add one 'noise' sentence."""
    n_sentences = rng.randint(3, 6)
    chosen = rng.sample(own_pool, k=min(n_sentences, len(own_pool)))
    if rng.random() < NOISE_PROBABILITY:
        noise_pool = rng.choice(other_pools)
        chosen.insert(rng.randrange(len(chosen) + 1), rng.choice(noise_pool))
    return " ".join(chosen)


def main():
    rng = random.Random(SEED)
    rows = []
    for label, (train_pool, test_pool) in POOLS.items():
        others_train = [p[0] for l, p in POOLS.items() if l != label]
        others_test = [p[1] for l, p in POOLS.items() if l != label]
        for _ in range(DOCS_PER_CATEGORY["train"]):
            rows.append((build_document(rng, train_pool, others_train), label, "train"))
        for _ in range(DOCS_PER_CATEGORY["test"]):
            rows.append((build_document(rng, test_pool, others_test), label, "test"))
    rng.shuffle(rows)

    out_path = Path(__file__).resolve().parent.parent / "data" / "training_data.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text", "label", "split"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} synthetic demo documents to {out_path}")


if __name__ == "__main__":
    main()
