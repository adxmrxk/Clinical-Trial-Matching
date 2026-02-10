import { ClinicalTrial, ChatResponse, TrialMatch } from '@/types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';
const N8N_WEBHOOK_URL = process.env.NEXT_PUBLIC_N8N_URL || 'http://localhost:5678/webhook/chat';

// Toggle this to switch between n8n and direct backend
const USE_N8N = false;

class ApiService {
  private sessionId: string | null = null;
  private patientProfile: Record<string, any> | null = null;

  async sendMessage(message: string): Promise<{
    response: string;
    responses: string[];  // Multiple responses for phase transitions
    trials: ClinicalTrial[];
    sessionId: string;
  }> {
    if (USE_N8N) {
      return this.sendMessageViaN8n(message);
    } else {
      return this.sendMessageDirect(message);
    }
  }

  // New: Send via n8n workflow
  private async sendMessageViaN8n(message: string): Promise<{
    response: string;
    responses: string[];
    trials: ClinicalTrial[];
    sessionId: string;
  }> {
    const res = await fetch(N8N_WEBHOOK_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message,
        patient_profile: this.patientProfile,
      }),
    });

    if (!res.ok) {
      throw new Error(`n8n error: ${res.status}`);
    }

    const data = await res.json();

    // Store updated patient profile for next request
    if (data.patient_profile) {
      this.patientProfile = data.patient_profile;
    }

    // Convert trial matches if present
    const trials = (data.trial_matches || []).map((match: any) => this.convertN8nTrialMatch(match));

    return {
      response: data.message || 'How can I help you find a clinical trial?',
      responses: [data.message || 'How can I help you find a clinical trial?'],
      trials,
      sessionId: 'n8n-session',
    };
  }

  // Original: Send directly to backend
  private async sendMessageDirect(message: string): Promise<{
    response: string;
    responses: string[];
    trials: ClinicalTrial[];
    sessionId: string;
  }> {
    const res = await fetch(`${API_BASE_URL}/chat/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        session_id: this.sessionId,
        message,
      }),
    });

    if (!res.ok) {
      throw new Error(`API error: ${res.status}`);
    }

    const data: ChatResponse = await res.json();
    this.sessionId = data.session_id;

    // Convert trial matches to simplified format for UI
    const trials = data.trial_matches.map(this.convertTrialMatch);

    // Get all message contents (for phase transitions with multiple messages)
    const responses = data.messages && data.messages.length > 0
      ? data.messages.map(m => m.content)
      : [data.message.content];

    return {
      response: data.message.content,  // Primary response (backwards compatible)
      responses,  // All responses for multi-message display
      trials,
      sessionId: data.session_id,
    };
  }

  // Convert n8n trial match format
  private convertN8nTrialMatch(match: any): ClinicalTrial {
    const trialId = match.trial_id || 'unknown';
    return {
      id: trialId,
      nctId: trialId,
      title: trialId || 'Clinical Trial',
      status: 'Recruiting',
      condition: 'Not specified',
      location: 'Not specified',
      eligibility: match.eligibility_status || 'uncertain',
      explanation: match.explanation || 'Eligibility being evaluated',
      criteriaMatched: (match.criteria_satisfied || []).map((c: any) => c.original_text || c),
      criteriaViolated: (match.criteria_violated || []).map((c: any) => c.original_text || c),
      criteriaUnknown: (match.criteria_unknown || []).map((c: any) => c.original_text || c),
      officialUrl: `https://clinicaltrials.gov/study/${trialId}`,
    };
  }

  private convertTrialMatch(match: TrialMatch): ClinicalTrial {
    const trial = match.trial;

    // Format location from first location in array
    const location = trial.locations.length > 0
      ? [trial.locations[0].facility, trial.locations[0].city, trial.locations[0].state, trial.locations[0].country]
          .filter(Boolean)
          .join(', ')
      : 'Location not specified';

    // Format age range
    const ageRange = trial.minimum_age || trial.maximum_age
      ? `${trial.minimum_age || 'No min'} - ${trial.maximum_age || 'No max'}`
      : undefined;

    // Get contact info if available
    const contact = trial.contacts && trial.contacts.length > 0 ? trial.contacts[0] : null;

    return {
      id: trial.nct_id,
      nctId: trial.nct_id,
      title: trial.title,
      status: trial.overall_status,
      condition: trial.conditions.join(', ') || 'Not specified',
      location,
      eligibility: match.eligibility_status,
      explanation: match.explanation,
      criteriaMatched: match.criteria_satisfied.map(c => c.original_text),
      criteriaViolated: match.criteria_violated.map(c => c.original_text),
      criteriaUnknown: match.criteria_unknown.map(c => c.original_text),
      // Additional details
      phase: trial.phase || undefined,
      briefSummary: trial.brief_summary || undefined,
      ageRange,
      sponsor: trial.lead_sponsor || undefined,
      contactName: contact?.name || undefined,
      contactEmail: contact?.email || undefined,
      contactPhone: contact?.phone || undefined,
      officialUrl: `https://clinicaltrials.gov/study/${trial.nct_id}`,
    };
  }

  async searchTrials(condition: string, location?: string): Promise<ClinicalTrial[]> {
    const params = new URLSearchParams();
    if (condition) params.append('condition', condition);
    if (location) params.append('location', location);

    const res = await fetch(`${API_BASE_URL}/trials/search?${params}`);

    if (!res.ok) {
      throw new Error(`API error: ${res.status}`);
    }

    const trials = await res.json();

    return trials.map((trial: any) => ({
      id: trial.nct_id,
      title: trial.title,
      status: trial.overall_status,
      condition: trial.conditions.join(', '),
      location: trial.locations[0]
        ? `${trial.locations[0].city}, ${trial.locations[0].country}`
        : 'Not specified',
      eligibility: 'uncertain' as const,
      explanation: 'Eligibility not yet evaluated',
      criteriaMatched: [],
      criteriaViolated: [],
      criteriaUnknown: [],
    }));
  }

  resetSession() {
    this.sessionId = null;
    this.patientProfile = null;
  }

  async transcribeAudio(audioBlob: Blob): Promise<string> {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.webm');

    const res = await fetch(`${API_BASE_URL}/chat/transcribe`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const error = await res.json().catch(() => ({ detail: 'Transcription failed' }));
      throw new Error(error.detail || `API error: ${res.status}`);
    }

    const data = await res.json();
    return data.text;
  }
}

export const apiService = new ApiService();
