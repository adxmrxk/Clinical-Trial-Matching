# Clinical Trial Matcher

An AI-powered clinical trial matching system that helps patients find relevant clinical trials through conversational interaction.

## Overview

Clinical trial recruitment is a major bottleneck in medical research. Many eligible patients never learn about trials that could benefit them due to complex eligibility criteria, fragmented information sources, and lack of accessible screening tools.

This project addresses these challenges by building an intelligent, conversational system that:

- **Collects patient information** through natural dialogue rather than rigid forms
- **Discovers relevant trials** from sources like ClinicalTrials.gov
- **Extracts and interprets eligibility criteria** using LLMs
- **Matches patients to trials** with explainable reasoning
- **Asks adaptive follow-up questions** to fill information gaps
- **Generates clear reports** explaining eligibility decisions

## System Architecture

### Multi-Agent Design

The system uses a multi-agent architecture where specialized agents handle different aspects of the matching process:

| Agent | Responsibility |
|-------|----------------|
| **Coordinator Agent** | Orchestrates workflow between agents |
| **Patient Profiling Agent** | Extracts structured patient data from conversation |
| **Trial Discovery Agent** | Queries clinical trial APIs and filters results |
| **Criteria Extraction Agent** | Parses eligibility criteria into structured rules |
| **Eligibility Matching Agent** | Compares patient profile against trial criteria |
| **Gap Analysis Agent** | Identifies missing information needed for eligibility |
| **Question Generation Agent** | Creates targeted follow-up questions |
| **Reporting Agent** | Generates explainable eligibility summaries |

### Three-Phase Questioning Strategy

1. **Phase 1 - Baseline Screening**: Collect universal attributes (age, sex, diagnosis, medications)
2. **Phase 2 - Trial-Driven Questioning**: Ask questions based on specific trial criteria
3. **Phase 3 - Gap-Filling**: Adaptive clarification to resolve uncertain eligibility

## Tech Stack

- **Frontend**: Next.js 16, React 19, Tailwind CSS, shadcn/ui
- **Backend**: FastAPI (Python)
- **LLMs**: Groq-hosted LLaMA 3 / Ollama / Hugging Face models
- **Speech-to-Text**: Whisper
- **Workflow Orchestration**: n8n
- **Data Sources**: ClinicalTrials.gov API, Canadian clinical trial registries

## Project Structure

```
clinical-trial-matching/
├── frontend/                 # Next.js web application
│   ├── src/
│   │   ├── app/             # Next.js app router pages
│   │   ├── components/      # React components
│   │   │   ├── chat/        # Chat interface components
│   │   │   ├── results/     # Trial results display
│   │   │   └── disclaimer/  # Medical disclaimers
│   │   ├── lib/             # Utilities and mock data
│   │   └── types/           # TypeScript type definitions
│   └── package.json
├── backend/                  # FastAPI backend (coming soon)
└── README.md
```

## Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn

### Running the Frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the application.

## Current Status

This project is under active development as part of a directed research project.

- [x] Frontend prototype with chat interface
- [x] Trial results display with eligibility badges
- [x] Mock conversational responses
- [ ] Backend API implementation
- [ ] Clinical trial API integration
- [ ] LLM-powered eligibility matching
- [ ] Multi-agent orchestration with n8n

## Disclaimer

This tool is for informational and research purposes only. It does not provide medical advice. Clinical trial eligibility shown is preliminary and must be confirmed by healthcare providers and trial coordinators. Always consult with your doctor before making decisions about participating in clinical trials.

## License

This project is part of academic research. Contact the author for licensing information.
