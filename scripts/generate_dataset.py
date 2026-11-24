"""Synthetic Dataset Generator for Multimodal Plagiarism Detection Engine.

Generates realistic case-study submissions across PDF, DOCX, and MP4 formats,
along with reference corpus articles and a ground-truth dataset manifest.
"""

import asyncio
from pathlib import Path
import json
import subprocess
import sys

from docx import Document
from docx.shared import Pt, RGBColor
import edge_tts
import imageio_ffmpeg
import pymupdf
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = PROJECT_ROOT / "sample_submissions"
REF_DIR = PROJECT_ROOT / "reference_corpus"
MANIFEST_PATH = SAMPLE_DIR / "dataset_manifest.json"

# Exact 147-word boilerplate header shared across sub_01 through sub_05
SHARED_BOILERPLATE = (
    "Case Competition Prompt & Submission Guidelines: All participants in this "
    "strategic analysis tournament must adhere strictly to the submission directives "
    "outlined in this charter. Submitted case reports must represent entirely "
    "independent analytical inquiry authored by registered team members without external "
    "unaccredited assistance. Strategic frameworks must be grounded in quantitative "
    "market data, rigorous unit economic modeling, and defendable financial forecasts. "
    "Any secondary research, public filings, industry benchmarks, or macroeconomic "
    "databases cited within the memorandum must carry complete academic citations. "
    "Submissions exceeding structural length limits or failing to articulate a concise "
    "executive summary will face scoring penalties during evaluation. Plagiarism, "
    "multi-team collusion, unauthorized peer document sharing, or uncredited reproduction "
    "of proprietary third-party trade publications will trigger immediate disqualification. "
    "By submitting this document, the authors affirm that all empirical evaluations, "
    "competitive positioning matrices, and implementation roadmaps are their own original work."
)

# Copied paragraph between reference_corpus/ev_battery_supply_chain.txt and sub_05_orig.pdf
EV_COPIED_PARAGRAPH = (
    "The critical path for next-generation electric vehicle adoption depends heavily "
    "on stabilizing the upstream refining throughput of battery-grade lithium hydroxide "
    "and nickel sulfate. While raw spodumene extraction has expanded across Australia "
    "and South America, chemical conversion remains geographically concentrated in East "
    "Asian coastal industrial zones. Developing domestic hydrometallurgical refining hubs "
    "requires significant capital expenditure, prolonged environmental permitting cycles, "
    "and specialized chemical engineering expertise. Until regional supply chains establish "
    "closed-loop battery recycling pipelines that recover cathode black mass at commercial scale, "
    "automotive original equipment manufacturers must navigate volatile feedstock pricing, "
    "geopolitical export controls, and tier-two supplier delivery bottlenecks."
)

# Verbatim identical paragraphs for sub_06_copy_src.pdf and sub_07_copy_dup.docx
SEMI_VERBATIM_P1 = (
    "Advanced packaging technologies, notably Chip-on-Wafer-on-Substrate (CoWoS) and "
    "high-density 2.5D interposers, have transitioned from niche high-performance computing "
    "options into the primary operational bottleneck for modern generative accelerator manufacturing. "
    "Front-end wafer lithography at sub-three-nanometer nodes yields negligible commercial utility "
    "if die interconnect density cannot overcome memory-bandwidth wall constraints. Foundry operators "
    "are consequently diverting capital allocation from pure substrate fabrication toward localized "
    "advanced packaging cleanrooms and thermal dissipation testing facilities."
)

SEMI_VERBATIM_P2 = (
    "The extreme capital intensity of next-generation extreme ultraviolet (EUV) lithography systems "
    "creates insurmountable barriers to entry for late-stage foundry contenders. Amortizing "
    "multi-billion dollar cleanroom tooling requires sustained utilization rates exceeding ninety percent, "
    "alongside multi-year anchor tenant purchase commitments from hyper-scaler clients. When cyclical "
    "semiconductor demand dips, foundries without diversified product lines across automotive and industrial "
    "microcontrollers face catastrophic margin erosion under fixed depreciation burdens."
)

# Paraphrased pair for sub_08_para_src.docx and sub_09_para_dup.docx
PARA_SRC_P1 = (
    "Vertical SaaS platforms are uniquely positioned to embed credit underwriting solutions directly "
    "into merchant daily workflows. By analyzing continuous real-time point-of-sale transaction telemetry "
    "and inventory turnover rates rather than static quarterly balance sheets, vertical software providers "
    "can offer working capital advances with substantially reduced default probabilities. This tight "
    "product integration minimizes customer acquisition costs and transforms standard software platforms "
    "into indispensable operating infrastructure for small enterprises."
)

PARA_SRC_P2 = (
    "However, shifting from an asset-light subscription revenue model to an embedded balance-sheet "
    "lending structure exposes enterprise software firms to severe macroeconomic interest rate volatility. "
    "Without sophisticated debt facility securitization and programmatic loan-loss reserve protocols, "
    "unexpected spikes in merchant delinquency can rapidly decimate operational cash reserves and undermine "
    "primary enterprise valuations."
)

PARA_DUP_P1 = (
    "Vertical software platforms are exceptionally positioned to integrate commercial credit solutions directly "
    "into merchant operational workflows. By evaluating real-time point-of-sale sales data and inventory turnover "
    "rates instead of static historical financial statements, vertical software vendors can provide working capital "
    "loans with significantly lower default risk. This seamless product integration reduces customer acquisition "
    "expenses and turns ordinary software systems into essential operating infrastructure for small businesses."
)

PARA_DUP_P2 = (
    "However, transitioning from an asset-light subscription model to balance-sheet loan origination exposes "
    "enterprise software companies to significant macroeconomic interest rate fluctuations. In the absence of "
    "structured credit facility securitization and automated loan loss reserve protocols, unexpected surges in "
    "merchant defaults can quickly deplete cash reserves and erode primary company valuations."
)

# Cross-modal paragraph for sub_10_pitch_src.pdf (page 2) and sub_11_pitch_video.mp4
CROSS_MODAL_PITCH_TEXT = (
    "Our pro forma financial model projects reaching operating break-even by month eighteen, "
    "underpinned by an average order value expansion to forty-two dollars. We anticipate blended "
    "gross margins will stabilize at sixty-four percent as centralized procurement volumes lower "
    "unit hardware costs. Net working capital requirements will peak during the third fiscal quarter "
    "before stabilizing alongside sustained free cash flow."
)

ORIGINAL_VIDEO_SPEECH = (
    "Autonomous long-haul freight operations unlock unprecedented efficiency gains by eliminating "
    "driver shift constraints and optimizing highway fuel consumption. Our deployment strategy "
    "prioritizes dedicated hub-to-hub interstate corridors where safety validation is streamlined "
    "and municipal regulatory overhead remains minimal."
)


def build_pdf(
    filepath: Path,
    title: str,
    paragraphs: list[str],
    header_box: str | None = None,
    page_break_indices: list[int] | None = None,
) -> None:
    """Builds a formatted PDF document using ReportLab."""
    doc = SimpleDocTemplate(
        str(filepath),
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54,
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=18,
        leading=22,
        textColor=HexColor("#0F172A"),
        spaceAfter=14,
    )
    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["Normal"],
        fontSize=10,
        leading=14.5,
        textColor=HexColor("#334155"),
        spaceAfter=11,
    )
    header_style = ParagraphStyle(
        "HeaderBox",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=12,
        textColor=HexColor("#475569"),
        spaceAfter=14,
    )

    story = []
    if header_box:
        story.append(Paragraph(f"<b>[METADATA & GUIDELINES]</b><br/>{header_box}", header_style))
        story.append(Spacer(1, 10))

    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 6))

    page_break_indices = page_break_indices or []
    for idx, para in enumerate(paragraphs):
        if idx in page_break_indices:
            story.append(PageBreak())
        story.append(Paragraph(para, body_style))

    doc.build(story)


def build_docx(
    filepath: Path,
    title: str,
    paragraphs: list[str],
    header_box: str | None = None,
) -> None:
    """Builds a formatted Word document using python-docx."""
    doc = Document()

    if header_box:
        hdr = doc.add_paragraph()
        run_title = hdr.add_run("[SUBMISSION METADATA & GUIDELINES]\n")
        run_title.bold = True
        run_title.font.size = Pt(8.5)
        run_title.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

        run_body = hdr.add_run(header_box)
        run_body.italic = True
        run_body.font.size = Pt(8.5)
        run_body.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

    doc.add_heading(title, level=1)
    for para in paragraphs:
        doc.add_paragraph(para)

    doc.save(str(filepath))


async def generate_speech_file(text: str, output_path: Path, voice: str = "en-US-ChristopherNeural") -> None:
    """Synthesizes text to an MP3 file using Edge-TTS."""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(output_path))


def render_video_with_audio(audio_path: Path, output_path: Path) -> None:
    """Renders an MP4 video with a dark slide background and synchronized audio."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-f", "lavfi",
        "-i", "color=c=0x0F172A:s=1280x720:r=24",
        "-i", str(audio_path),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def render_silent_video(output_path: Path, duration_sec: int = 5) -> None:
    """Renders a 5-second video without any audio track."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-f", "lavfi",
        "-i", "color=c=0x0F172A:s=1280x720:r=24",
        "-t", str(duration_sec),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-an",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def generate_reference_corpus() -> None:
    """Generates 4 synthetic public-domain reference articles."""
    REF_DIR.mkdir(parents=True, exist_ok=True)

    # 1. ev_battery_supply_chain.txt
    ev_article = (
        "TITLE: Global Electric Vehicle Battery Supply Chain Resilience & Mineral Refining Economics\n\n"
        "SECTION 1: UPSTREAM EXTRACTION AND CHEMICAL CONVERSION\n"
        + EV_COPIED_PARAGRAPH
        + "\n\n"
        "SECTION 2: CELL MANUFACTURING AND GIGAFACTORY DYNAMICS\n"
        "Automotive cell manufacturing requires tight cleanroom humidity control and precise slurry coating "
        "mechanisms. Scrap rates during initial factory commissioning regularly exceed twenty-five percent, "
        "imposing heavy cash burn on joint venture gigafactories. As production stabilizes, scrap rates decline "
        "toward five percent, unlocking economies of scale that drive pack-level costs toward the targeted eighty "
        "dollar per kilowatt-hour parity benchmark.\n\n"
        "SECTION 3: STRATEGIC OFFTAKE AND GEOPOLITICAL BUFFERING\n"
        "OEMs are actively signing multi-year direct equity-linked offtake agreements with junior miners to hedge "
        "against future market deficits. Creating decentralized cathode precursor synthesis facilities in North America "
        "and Europe serves as an essential strategic moat against maritime shipping disruptions and tariff restructuring."
    )
    (REF_DIR / "ev_battery_supply_chain.txt").write_text(ev_article, encoding="utf-8")

    # 2. saas_metrics_guide.txt
    saas_article = (
        "TITLE: Modern B2B SaaS Benchmarks: Unit Economics, Capital Efficiency, and Net Retention\n\n"
        "SECTION 1: THE CORE SAAS GROWTH ENGINE\n"
        "Sustainable software-as-a-service scalability requires rigorous balance between customer acquisition "
        "cost (CAC) payback periods and customer lifetime value (LTV). Best-in-class enterprise software companies "
        "target CAC payback intervals under fourteen months on a fully loaded sales and marketing expense basis.\n\n"
        "SECTION 2: NET RETENTION AND EXPANSION LEVERS\n"
        "Net Revenue Retention (NRR) above one hundred and twenty percent distinguishes top-quartile cloud platforms. "
        "Achieving strong NRR demands consumption-based pricing tiers, proactive customer success interventions, and "
        "cross-selling ancillary feature modules before renewal milestones.\n\n"
        "SECTION 3: THE RULE OF 40 IN TIGHT CAPITAL MARKETS\n"
        "In periods of elevated cost of capital, public markets reward software companies that satisfy the Rule of 40—where "
        "annual revenue growth rate plus free cash flow margin exceeds forty percent. Balancing disciplined headcount growth "
        "with automated customer onboarding enables companies to preserve positive free cash flow while capturing market share."
    )
    (REF_DIR / "saas_metrics_guide.txt").write_text(saas_article, encoding="utf-8")

    # 3. quick_commerce_india.txt
    qcom_article = (
        "TITLE: Quick-Commerce Dark Store Unit Economics: Urban Density and Fulfillment Optimization\n\n"
        "SECTION 1: CATCHMENT AREA METRICS\n"
        "Ten-minute grocery delivery models rely on highly localized micro-fulfillment centers (dark stores) serving a "
        "tight radius of two to three kilometers. High order density is the paramount determinant of financial viability; "
        "dark stores must process at least eight hundred daily orders to amortize fixed rental overhead and cold-chain energy costs.\n\n"
        "SECTION 2: RIDER UTILIZATION AND INVENTORY PACKING\n"
        "Pick-and-pack times must be compressed to under two minutes through algorithmic SKU slotting and picker routing. "
        "Simultaneously, delivery rider batching algorithms aggregate proximate drops, raising rider efficiency from 1.6 to 2.8 "
        "deliveries per hour during evening demand peaks.\n\n"
        "SECTION 3: MARGIN EXPANSION VIA PRIVATE LABELS AND AD SALES\n"
        "FMCG brand advertising and high-margin private label staples provide the necessary gross margin expansion to offset "
        "last-mile delivery subsidies. Reaching store-level EBITDA break-even requires maintaining gross product margins above "
        "twenty-one percent alongside minimum order values exceeding twelve dollars."
    )
    (REF_DIR / "quick_commerce_india.txt").write_text(qcom_article, encoding="utf-8")

    # 4. solar_microgrids.txt
    microgrid_article = (
        "TITLE: Decentralized Renewable Microgrids: Levelized Cost of Storage and Hybrid System Architecture\n\n"
        "SECTION 1: SYSTEM TOPOLOGY AND HYBRIDIZATION\n"
        "Commercial and industrial microgrids integrate rooftop photovoltaic arrays with lithium iron phosphate battery energy "
        "storage systems (BESS). Combining bifacial solar panels with intelligent inverters enables islanded operation during grid "
        "blackouts while shaving peak demand charges during maximum tariff windows.\n\n"
        "SECTION 2: LEVELIZED COST OF STORAGE DYNAMICS\n"
        "The levelized cost of storage (LCOS) has decreased substantially due to standardized containerized battery enclosures and "
        "advanced battery management system software. Degradation modeling indicates that keeping battery state of charge between "
        "ten and ninety percent extends cell operational lifespan to over four thousand full equivalent cycles.\n\n"
        "SECTION 3: TRANSACTIVE ENERGY AND PEER-TO-PEER SETTLEMENT\n"
        "Modern smart microgrid networks utilize peer-to-peer algorithmic power trading to balance localized surplus and deficit "
        "nodes. Automated transactive energy contracts dynamically adjust dispatch parameters based on real-time weather forecasts "
        "and day-ahead wholesale electricity price signals."
    )
    (REF_DIR / "solar_microgrids.txt").write_text(microgrid_article, encoding="utf-8")


async def generate_all_submissions() -> None:
    """Generates all 15 synthetic case submissions and the ground-truth manifest."""
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # sub_01_orig.pdf: SaaS Pricing Strategy (Shared Boilerplate)
    # -------------------------------------------------------------
    build_pdf(
        SAMPLE_DIR / "sub_01_orig.pdf",
        title="B2B Enterprise SaaS: Pricing Architecture and Product-Led Expansion",
        paragraphs=[
            "Enterprise software organizations are increasingly transitioning from seat-based subscription "
            "licenses to hybrid value-metric pricing. Charging customers based on actual compute throughput, API "
            "invocations, or business transactions aligns software expenditure with tangible client value creation.",
            "Our quantitative churn analysis demonstrates that customers under hybrid usage models exhibit 18% higher "
            "net retention over twenty-four months compared to legacy fixed seat deployments. Furthermore, tiered feature "
            "gating incentivizes enterprise accounts to self-upgrade as internal departmental adoption expands.",
            "To mitigate revenue unpredictability inherent in purely variable billing, software vendors must mandate "
            "baseline annual platform commit fees. This floor guarantees foundational recurring revenue while allowing "
            "topline expansion through variable quarterly true-up mechanisms.",
        ],
        header_box=SHARED_BOILERPLATE,
    )

    # -------------------------------------------------------------
    # sub_02_orig.docx: Commercial Fleet Telematics (Shared Boilerplate)
    # -------------------------------------------------------------
    build_docx(
        SAMPLE_DIR / "sub_02_orig.docx",
        title="Commercial Fleet Telematics: Predictive Route Optimization and Sensor Analytics",
        paragraphs=[
            "Fleet operators managing heavy regional distribution assets are adopting edge sensor telemetry "
            "to minimize scheduled maintenance overhead. Continuous engine vibration monitoring and tire pressure "
            "telematics reduce unscheduled highway roadside breakdowns by nearly forty percent across nationwide networks.",
            "Route dispatch algorithms that dynamically account for interstate grade changes, localized temperature "
            "fluctuations, and real-time congestion allow commercial trucking carriers to cut diesel expenditure by eleven percent. "
            "Integrating telematics with onboard driver coaching software fosters sustained compliance with federal driving hours.",
            "Commercial fleet electrification introduces novel charging optimization challenges. Fleets must balance depot "
            "megawatt charging schedules against peak utility tariffs to preserve the total cost of ownership advantages of commercial electric trucks.",
        ],
        header_box=SHARED_BOILERPLATE,
    )

    # -------------------------------------------------------------
    # sub_03_orig.pdf: Offshore Floating Wind (Shared Boilerplate)
    # -------------------------------------------------------------
    build_pdf(
        SAMPLE_DIR / "sub_03_orig.pdf",
        title="Offshore Floating Wind: Mooring Tension Economics and Deep-Water Subsea Cabling",
        paragraphs=[
            "Deep-water offshore wind development necessitates floating spar and semi-submersible foundation "
            "structures where conventional bottom-fixed monopile foundations are technically infeasible. Tension-leg "
            "platforms anchored with synthetic taut lines dramatically reduce steel fabrication requirements in marine depths.",
            "Dynamic inter-array electrical cabling must withstand severe continuous hydrodynamic wave fatigue cycles. "
            "Implementing distributed fiber-optic acoustic sensing along subsea export lines enables early detection of cable "
            "vortex-induced vibrations before mechanical insulation failure occurs.",
            "Vessel installation logistics remain the dominant capital expenditure bottleneck. Standardizing modular turbine "
            "quayside assembly eliminates the need for expensive heavy-lift jack-up crane vessels during offshore turbine integration.",
        ],
        header_box=SHARED_BOILERPLATE,
    )

    # -------------------------------------------------------------
    # sub_04_orig.docx: Healthcare AI Diagnostics (Shared Boilerplate)
    # -------------------------------------------------------------
    build_docx(
        SAMPLE_DIR / "sub_04_orig.docx",
        title="Commercializing Healthcare AI: Reimbursement Pathways and Hospital Procurement",
        paragraphs=[
            "Developing regulatory-cleared computer vision diagnostic algorithms requires immense clinical trial expenditure, "
            "yet hospital procurement cycles remain the primary bottleneck to commercial scale. AI developers must secure dedicated "
            "Current Procedural Terminology (CPT) reimbursement codes to ensure healthcare provider adoption.",
            "Health systems operate on tight operational margins and rarely invest in standalone software tools that lack direct fee-for-service "
            "or value-based reimbursement mechanisms. Enterprise sales strategies must directly demonstrate radiologist productivity gains "
            "or measurable reductions in patient readmission penalties.",
            "Channel partnerships with established medical imaging hardware manufacturers provide an accelerated route to market, embedding "
            "diagnostic software directly onto hospital picture archiving and communication systems.",
        ],
        header_box=SHARED_BOILERPLATE,
    )

    # -------------------------------------------------------------
    # sub_05_orig.pdf: EV Supply Chain (Shared Boilerplate + External Match)
    # -------------------------------------------------------------
    build_pdf(
        SAMPLE_DIR / "sub_05_orig.pdf",
        title="Electric Mobility Strategy: Cathode Chemistry and Upstream Mineral Security",
        paragraphs=[
            "The global transition toward electrified automotive transport requires unprecedented restructuring of mineral procurement "
            "strategies. Automotive manufacturers are re-evaluating traditional tier-one supplier dependencies in favor of direct raw material "
            "refining investments.",
            # Verbatim paragraph from reference_corpus/ev_battery_supply_chain.txt
            EV_COPIED_PARAGRAPH,
            "In conclusion, automotive firms that secure dedicated hydrometallurgical refining capacity through equity joint ventures will "
            "maintain superior cost competitiveness over competitors relying on volatile spot commodity markets.",
        ],
        header_box=SHARED_BOILERPLATE,
    )

    # -------------------------------------------------------------
    # sub_06_copy_src.pdf & sub_07_copy_dup.docx: Verbatim Peer Plagiarism Pair
    # -------------------------------------------------------------
    build_pdf(
        SAMPLE_DIR / "sub_06_copy_src.pdf",
        title="Semiconductor Foundries: CapEx Dynamics and Advanced Packaging Bottlenecks",
        paragraphs=[
            "The geopolitical prioritization of domestic semiconductor manufacturing has spurred massive sovereign capital subsidies. "
            "However, commercial success ultimately depends on mastering packaging economics rather than raw silicon wafer output alone.",
            SEMI_VERBATIM_P1,
            SEMI_VERBATIM_P2,
            "Consequently, future foundry competitiveness will be dictated by co-locating packaging assembly facilities directly adjacent "
            "to primary fabrication plants.",
        ],
    )

    build_docx(
        SAMPLE_DIR / "sub_07_copy_dup.docx",
        title="Strategic Analysis of Modern Semiconductor Manufacturing and Fabrication Economics",
        paragraphs=[
            "Modern microelectronics manufacturing requires balanced capital expenditure across front-end lithography and back-end integration. "
            "Foundries must adapt to structural industry shifts to sustain operating margins.",
            SEMI_VERBATIM_P1,
            SEMI_VERBATIM_P2,
            "Strategic leadership in microchip manufacturing therefore necessitates long-term capital preservation and disciplined yield ramp management.",
        ],
    )

    # -------------------------------------------------------------
    # sub_08_para_src.docx & sub_09_para_dup.docx: Paraphrased Peer Pair
    # -------------------------------------------------------------
    build_docx(
        SAMPLE_DIR / "sub_08_para_src.docx",
        title="Embedded Finance in Vertical Software: Expanding ARPU Through Integrated Lending",
        paragraphs=[
            "Modern business software platforms are seeking novel monetization channels beyond traditional subscription pricing models. "
            "Embedded financial services offer substantial revenue expansion potential.",
            PARA_SRC_P1,
            PARA_SRC_P2,
            "In summary, SaaS providers should partner with licensed balance-sheet lenders rather than holding unhedged loan risk directly.",
        ],
    )

    build_docx(
        SAMPLE_DIR / "sub_09_para_dup.docx",
        title="Strategic Commercial Lending Integration Across Sector-Focused Software Platforms",
        paragraphs=[
            "Niche software applications are exploring innovative growth avenues to augment conventional SaaS licensing turnover. "
            "Embedding financial products presents significant upside for total customer value.",
            PARA_DUP_P1,
            PARA_DUP_P2,
            "To conclude, enterprise software platforms ought to collaborate with regulated credit institutions rather than underwriting credit liabilities internally.",
        ],
    )

    # -------------------------------------------------------------
    # sub_10_pitch_src.pdf: Multi-page PDF containing 60-word pitch on page 2
    # -------------------------------------------------------------
    build_pdf(
        SAMPLE_DIR / "sub_10_pitch_src.pdf",
        title="Next-Generation Hardware Infrastructure: Strategic Business Plan",
        paragraphs=[
            "Executive Summary: Our platform engineers enterprise IoT hardware solutions designed for industrial predictive maintenance. "
            "By capturing high-frequency vibration and thermal telemetry across heavy industrial machinery, our edge appliances prevent catastrophic "
            "equipment failure before unbudgeted shutdowns occur.",
            "Market Opportunity: The global market for smart manufacturing asset management is projected to surpass thirty billion dollars by twenty thirty. "
            "Our early enterprise pilots demonstrate a three hundred percent return on investment within nine months of deployment.",
            # PageBreak occurs before page 2
            CROSS_MODAL_PITCH_TEXT,
            "Sensitivity Analysis: Even under conservative hardware adoption scenarios assuming twenty percent lower device throughput, "
            "operating margins remain positive owing to our recurring predictive maintenance software subscription tiers.",
        ],
        page_break_indices=[2],  # Break before CROSS_MODAL_PITCH_TEXT so it is on page 2
    )

    # -------------------------------------------------------------
    # sub_11_pitch_video.mp4: Cross-modal spoken video matching sub_10 page 2
    # -------------------------------------------------------------
    audio_pitch_tmp = SAMPLE_DIR / "sub_11_pitch_tmp.mp3"
    await generate_speech_file(CROSS_MODAL_PITCH_TEXT, audio_pitch_tmp, voice="en-US-ChristopherNeural")
    render_video_with_audio(audio_pitch_tmp, SAMPLE_DIR / "sub_11_pitch_video.mp4")
    if audio_pitch_tmp.exists():
        audio_pitch_tmp.unlink()

    # -------------------------------------------------------------
    # sub_12_orig_video.mp4: Original spoken video (~15s)
    # -------------------------------------------------------------
    audio_orig_tmp = SAMPLE_DIR / "sub_12_orig_tmp.mp3"
    await generate_speech_file(ORIGINAL_VIDEO_SPEECH, audio_orig_tmp, voice="en-US-JennyNeural")
    render_video_with_audio(audio_orig_tmp, SAMPLE_DIR / "sub_12_orig_video.mp4")
    if audio_orig_tmp.exists():
        audio_orig_tmp.unlink()

    # -------------------------------------------------------------
    # sub_13_ai_gen.docx: Textbook uniform AI-generated essay (~400 words)
    # -------------------------------------------------------------
    build_docx(
        SAMPLE_DIR / "sub_13_ai_gen.docx",
        title="Navigating the Digital Transformation Paradigm: A Strategic Framework for Contemporary Enterprises",
        paragraphs=[
            "In today's rapidly evolving digital landscape, it is crucial to note that digital transformation represents an indispensable strategic "
            "imperative for contemporary business organizations. Furthermore, companies must meticulously align their technological infrastructure "
            "with shifting consumer behaviors to maintain enduring relevance. In addition, the integration of cloud-native computing paradigms facilitates "
            "enhanced operational agility, thereby enabling enterprise leaders to respond dynamically to volatile market conditions.",
            "Moreover, data-driven decision-making serves as a foundational cornerstone for sustaining competitive advantage in the modern economy. "
            "Furthermore, advanced analytical frameworks allow firms to extract actionable intelligence from vast informational repositories. "
            "It is important to emphasize that cross-functional collaboration plays a pivotal role in democratizing access to analytical tools across departments. "
            "Additionally, organizational silos must be systematically eliminated to foster seamless interdepartmental communication and operational synergy.",
            "Furthermore, corporate governance and cybersecurity protocols must be proactively instituted to safeguard vital proprietary assets. "
            "It is crucial to recognize that emerging threats necessitate continuous vulnerability assessments and rigorous compliance monitoring. "
            "Moreover, comprehensive employee upskilling initiatives ensure that organizational human capital adapts seamlessly to evolving algorithmic workflows. "
            "Additionally, cultivating a corporate culture of lifelong learning empowers personnel to embrace technological innovation enthusiastically.",
            "In conclusion, it is evident that modern enterprises must navigate digital transformation with strategic clarity, discipline, and intentionality. "
            "Furthermore, organizations that harmoniously synthesize cutting-edge technology, collaborative corporate structures, and proactive risk mitigation "
            "are exceptionally well-positioned to achieve sustainable long-term prosperity. Ultimately, digital readiness is not merely an operational luxury, "
            "but rather an existential prerequisite for enduring market leadership.",
        ],
    )

    # -------------------------------------------------------------
    # sub_14_corrupted.pdf: 0-byte corrupted file
    # -------------------------------------------------------------
    (SAMPLE_DIR / "sub_14_corrupted.pdf").write_bytes(b"")

    # -------------------------------------------------------------
    # sub_15_silent_video.mp4: 5-second video with NO audio track
    # -------------------------------------------------------------
    render_silent_video(SAMPLE_DIR / "sub_15_silent_video.mp4", duration_sec=5)

    # -------------------------------------------------------------
    # sub_16_ai_natural.docx: Marker-free natural LLM business analysis (~350 words)
    # -------------------------------------------------------------
    build_docx(
        SAMPLE_DIR / "sub_16_ai_natural.docx",
        title="Evaluating Robotics-as-a-Service (RaaS) Unit Economics in Mid-Market Warehousing",
        paragraphs=[
            "Warehouse operators in the mid-market segment often hesitate to purchase autonomous mobile robots outright because upfront hardware outlays strain quarterly working capital. "
            "Subscription-based Robotics-as-a-Service contracts remove that initial capital barrier by converting robotics deployment into a predictable monthly operating expense tied directly to throughput volume. "
            "For robotics vendors, however, shifting from one-time equipment sales to recurring leases creates an immediate financing gap between component procurement and multi-year lease collection.",
            "Closing this cash conversion gap depends on lowering field deployment friction and lengthening hardware payback durability. "
            "Early RaaS providers underestimated the engineering hours needed to map dynamic warehouse aisles, calibrate safety sensors, and integrate fleet software with legacy warehouse management databases. "
            "When site commissioning stretches past six weeks, installation labor quickly consumes the first year of gross subscription margin. "
            "Vendors that standardize modular sensor kits and automate aisle mapping can reduce onboarding cycles below ten days, turning unit contribution margins positive before month fourteen.",
            "Fleet uptime guarantees also shape long-term profitability under recurring service agreements. "
            "Unlike traditional industrial machinery sold with limited warranties, leased warehouse robots require vendor-funded preventive maintenance, battery replacements, and remote telemetry support. "
            "Predictive diagnostics built into edge motor controllers allow service teams to schedule component swaps during off-peak shifts rather than dispatching emergency field technicians after unexpected stoppages. "
            "Keeping fleet availability above ninety-eight percent protects contract renewal rates while preventing punitive service-level credits.",
            "Over a three-year contract horizon, secondary hardware redeployment determines whether a RaaS business compounds capital efficiently or stalls under depreciation weight. "
            "Robots returned after an initial facility lease retain substantial mechanical value if core chassis frames and drive assemblies are engineered for rapid refurbishment. "
            "Refurbishing and re-leasing second-cycle units to seasonal distribution centers raises lifetime asset return on invested capital without requiring fresh bill-of-materials spending. "
            "Disciplined fleet financing, rapid commissioning, and multi-cycle asset utilization together separate durable RaaS platforms from hardware vendors merely discounting equipment through installment plans.",
        ],
    )

    # -------------------------------------------------------------
    # dataset_manifest.json: Ground-truth dataset documentation
    # -------------------------------------------------------------
    manifest = [
        {
            "filename": "sub_01_orig.pdf",
            "file_type": "pdf",
            "expected_peer_plagiarism": False,
            "expected_peer_sources": [],
            "expected_multimodal_match": False,
            "expected_external_match": False,
            "expected_ai_generated": False,
            "is_corrupted_or_empty": False,
            "has_shared_boilerplate": True,
            "notes": "Original PDF submission on B2B SaaS pricing models with competition boilerplate header.",
        },
        {
            "filename": "sub_02_orig.docx",
            "file_type": "docx",
            "expected_peer_plagiarism": False,
            "expected_peer_sources": [],
            "expected_multimodal_match": False,
            "expected_external_match": False,
            "expected_ai_generated": False,
            "is_corrupted_or_empty": False,
            "has_shared_boilerplate": True,
            "notes": "Original DOCX submission on commercial fleet telematics with competition boilerplate header.",
        },
        {
            "filename": "sub_03_orig.pdf",
            "file_type": "pdf",
            "expected_peer_plagiarism": False,
            "expected_peer_sources": [],
            "expected_multimodal_match": False,
            "expected_external_match": False,
            "expected_ai_generated": False,
            "is_corrupted_or_empty": False,
            "has_shared_boilerplate": True,
            "notes": "Original PDF submission on offshore floating wind with competition boilerplate header.",
        },
        {
            "filename": "sub_04_orig.docx",
            "file_type": "docx",
            "expected_peer_plagiarism": False,
            "expected_peer_sources": [],
            "expected_multimodal_match": False,
            "expected_external_match": False,
            "expected_ai_generated": False,
            "is_corrupted_or_empty": False,
            "has_shared_boilerplate": True,
            "notes": "Original DOCX submission on healthcare AI diagnostics with competition boilerplate header.",
        },
        {
            "filename": "sub_05_orig.pdf",
            "file_type": "pdf",
            "expected_peer_plagiarism": False,
            "expected_peer_sources": [],
            "expected_multimodal_match": False,
            "expected_external_match": True,
            "expected_ai_generated": False,
            "is_corrupted_or_empty": False,
            "has_shared_boilerplate": True,
            "notes": "Contains competition boilerplate header and copies one full paragraph verbatim from reference_corpus/ev_battery_supply_chain.txt.",
        },
        {
            "filename": "sub_06_copy_src.pdf",
            "file_type": "pdf",
            "expected_peer_plagiarism": True,
            "expected_peer_sources": ["sub_07_copy_dup.docx"],
            "expected_peer_chunk_count": 2,
            "expected_multimodal_match": False,
            "expected_external_match": False,
            "expected_ai_generated": False,
            "is_corrupted_or_empty": False,
            "has_shared_boilerplate": False,
            "notes": "Source PDF submission for verbatim peer plagiarism pair on semiconductor packaging economics.",
        },
        {
            "filename": "sub_07_copy_dup.docx",
            "file_type": "docx",
            "expected_peer_plagiarism": True,
            "expected_peer_sources": ["sub_06_copy_src.pdf"],
            "expected_peer_chunk_count": 2,
            "expected_multimodal_match": False,
            "expected_external_match": False,
            "expected_ai_generated": False,
            "is_corrupted_or_empty": False,
            "has_shared_boilerplate": False,
            "notes": "Duplicate DOCX submission containing 2 paragraphs copied verbatim from sub_06_copy_src.pdf.",
        },
        {
            "filename": "sub_08_para_src.docx",
            "file_type": "docx",
            "expected_peer_plagiarism": True,
            "expected_peer_sources": ["sub_09_para_dup.docx"],
            "expected_peer_chunk_count": 2,
            "expected_multimodal_match": False,
            "expected_external_match": False,
            "expected_ai_generated": False,
            "is_corrupted_or_empty": False,
            "has_shared_boilerplate": False,
            "notes": "Source DOCX submission for paraphrased pair on embedded lending in vertical SaaS.",
        },
        {
            "filename": "sub_09_para_dup.docx",
            "file_type": "docx",
            "expected_peer_plagiarism": True,
            "expected_peer_sources": ["sub_08_para_src.docx"],
            "expected_peer_chunk_count": 2,
            "expected_multimodal_match": False,
            "expected_external_match": False,
            "expected_ai_generated": False,
            "is_corrupted_or_empty": False,
            "has_shared_boilerplate": False,
            "notes": "Paraphrased DOCX submission expressing the exact same 2-paragraph arguments as sub_08_para_src.docx with altered phrasing.",
        },
        {
            "filename": "sub_10_pitch_src.pdf",
            "file_type": "pdf",
            "expected_peer_plagiarism": False,
            "expected_peer_sources": [],
            "expected_multimodal_match": True,
            "expected_external_match": False,
            "expected_ai_generated": False,
            "is_corrupted_or_empty": False,
            "has_shared_boilerplate": False,
            "notes": "Multi-page PDF containing a financial projection paragraph on page 2 that matches sub_11_pitch_video.mp4 spoken audio.",
        },
        {
            "filename": "sub_11_pitch_video.mp4",
            "file_type": "mp4",
            "expected_peer_plagiarism": False,
            "expected_peer_sources": [],
            "expected_multimodal_match": True,
            "expected_external_match": False,
            "expected_ai_generated": False,
            "is_corrupted_or_empty": False,
            "has_shared_boilerplate": False,
            "notes": "Video pitch reciting the exact financial analysis paragraph from page 2 of sub_10_pitch_src.pdf.",
        },
        {
            "filename": "sub_12_orig_video.mp4",
            "file_type": "mp4",
            "expected_peer_plagiarism": False,
            "expected_peer_sources": [],
            "expected_multimodal_match": False,
            "expected_external_match": False,
            "expected_ai_generated": False,
            "is_corrupted_or_empty": False,
            "has_shared_boilerplate": False,
            "notes": "Original video submission discussing autonomous freight logistics corridors.",
        },
        {
            "filename": "sub_13_ai_gen.docx",
            "file_type": "docx",
            "expected_peer_plagiarism": False,
            "expected_peer_sources": [],
            "expected_multimodal_match": False,
            "expected_external_match": False,
            "expected_ai_generated": True,
            "is_corrupted_or_empty": False,
            "has_shared_boilerplate": False,
            "notes": "Formulaic ~400-word essay with uniform syntax and repetitive transition phrases to test AI detection.",
        },
        {
            "filename": "sub_14_corrupted.pdf",
            "file_type": "pdf",
            "expected_peer_plagiarism": False,
            "expected_peer_sources": [],
            "expected_multimodal_match": False,
            "expected_external_match": False,
            "expected_ai_generated": False,
            "is_corrupted_or_empty": True,
            "has_shared_boilerplate": False,
            "notes": "Corrupted 0-byte PDF submission.",
        },
        {
            "filename": "sub_15_silent_video.mp4",
            "file_type": "mp4",
            "expected_peer_plagiarism": False,
            "expected_peer_sources": [],
            "expected_multimodal_match": False,
            "expected_external_match": False,
            "expected_ai_generated": False,
            "is_corrupted_or_empty": True,
            "has_shared_boilerplate": False,
            "notes": "Silent 5-second video submission with no audio track.",
        },
        {
            "filename": "sub_16_ai_natural.docx",
            "file_type": "docx",
            "expected_peer_plagiarism": False,
            "expected_peer_sources": [],
            "expected_multimodal_match": False,
            "expected_external_match": False,
            "expected_ai_generated": True,
            "is_corrupted_or_empty": False,
            "has_shared_boilerplate": False,
            "notes": "Marker-free ~350-word natural-sounding LLM business analysis avoiding cliché transition phrases to test RoBERTa vs heuristic blending.",
        },
    ]

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def main() -> None:
    """Entry point for synthetic dataset generation."""
    print("Generating reference corpus...")
    generate_reference_corpus()
    print("Generating sample submissions (PDF, DOCX, MP4)...")
    asyncio.run(generate_all_submissions())
    print(f"Dataset generated successfully! Manifest saved to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
