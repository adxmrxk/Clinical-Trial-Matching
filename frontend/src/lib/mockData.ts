import { ClinicalTrial } from '@/types';

export const mockTrials: ClinicalTrial[] = [
  {
    id: 'NCT12345678',
    nctId: 'NCT12345678',
    title: 'A Phase 3 Study of Novel Treatment for Type 2 Diabetes',
    status: 'Recruiting',
    condition: 'Type 2 Diabetes Mellitus',
    location: 'Toronto, Ontario, Canada',
    eligibility: 'eligible',
    explanation:
      'Based on the information provided, you appear to meet the key eligibility criteria for this trial.',
    criteriaMatched: [
      'Age between 18-65 years',
      'Diagnosed with Type 2 Diabetes',
      'HbA1c between 7.0% and 10.0%',
    ],
    criteriaViolated: [],
    criteriaUnknown: ['Current kidney function (eGFR)'],
    officialUrl: 'https://clinicaltrials.gov/study/NCT12345678',
  },
  {
    id: 'NCT87654321',
    nctId: 'NCT87654321',
    title: 'Evaluating a New Oral Medication for Diabetes Management',
    status: 'Recruiting',
    condition: 'Type 2 Diabetes Mellitus',
    location: 'Vancouver, British Columbia, Canada',
    eligibility: 'uncertain',
    explanation:
      'We need more information about your current medications to determine full eligibility.',
    criteriaMatched: ['Age requirement met', 'Diabetes diagnosis confirmed'],
    criteriaViolated: [],
    criteriaUnknown: [
      'Current diabetes medications',
      'History of diabetic ketoacidosis',
      'Liver function tests',
    ],
    officialUrl: 'https://clinicaltrials.gov/study/NCT87654321',
  },
  {
    id: 'NCT11223344',
    nctId: 'NCT11223344',
    title: 'Lifestyle Intervention Study for Prediabetes Prevention',
    status: 'Recruiting',
    condition: 'Prediabetes',
    location: 'Montreal, Quebec, Canada',
    eligibility: 'ineligible',
    explanation:
      'This trial is specifically for patients with prediabetes, not those already diagnosed with Type 2 Diabetes.',
    criteriaMatched: ['Age requirement met', 'Located in Canada'],
    criteriaViolated: ['Must have prediabetes, not Type 2 Diabetes'],
    criteriaUnknown: [],
    officialUrl: 'https://clinicaltrials.gov/study/NCT11223344',
  },
  {
    id: 'NCT99887766',
    nctId: 'NCT99887766',
    title: 'Continuous Glucose Monitoring Device Trial',
    status: 'Recruiting',
    condition: 'Diabetes Mellitus',
    location: 'Ottawa, Ontario, Canada',
    eligibility: 'eligible',
    explanation:
      'You meet the eligibility criteria for this device study evaluating a new continuous glucose monitor.',
    criteriaMatched: [
      'Diagnosed with diabetes',
      'Currently using insulin or considering insulin therapy',
      'Willing to wear a continuous glucose monitor',
    ],
    criteriaViolated: [],
    criteriaUnknown: ['Skin allergies or sensitivities'],
    officialUrl: 'https://clinicaltrials.gov/study/NCT99887766',
  },
];

export function getMockResponse(userMessage: string): string {
  const lowerMessage = userMessage.toLowerCase();

  if (
    lowerMessage.includes('hello') ||
    lowerMessage.includes('hi') ||
    lowerMessage.includes('hey')
  ) {
    return "Hello! I'm here to help you find clinical trials that might be relevant to your health condition. Could you tell me about your medical condition, age, and any current treatments you're receiving?";
  }

  if (lowerMessage.includes('diabetes') || lowerMessage.includes('diabetic')) {
    return "Thank you for sharing that you have diabetes. To find the most relevant clinical trials, I have a few follow-up questions:\n\n1. What type of diabetes do you have (Type 1, Type 2, or gestational)?\n2. How long have you been diagnosed?\n3. What is your current HbA1c level, if known?\n4. Are you currently taking any diabetes medications?\n\nThis information will help me match you with appropriate clinical trials.";
  }

  if (lowerMessage.includes('type 2') || lowerMessage.includes('type2')) {
    return "Thank you for confirming Type 2 Diabetes. I've found several clinical trials that might be relevant to you. Based on the information you've provided, I'm showing trials that match your profile.\n\nI notice we're still missing some information that could affect your eligibility:\n- Current kidney function (eGFR)\n- Complete list of current medications\n- Recent liver function tests\n\nWould you like to provide any of this information to refine your matches?";
  }

  if (lowerMessage.includes('age') || lowerMessage.includes('years old')) {
    return "Thank you for providing your age. This helps narrow down trials with specific age requirements. Many diabetes trials require participants to be between 18-75 years old.\n\nWhat other health conditions do you have, if any? This helps us identify trials you might be eligible for and avoid those where certain conditions might exclude you.";
  }

  if (
    lowerMessage.includes('medication') ||
    lowerMessage.includes('medicine') ||
    lowerMessage.includes('taking')
  ) {
    return "Thank you for sharing your medication information. This is important because:\n- Some trials require you to be on specific medications\n- Others may exclude certain medications\n- Some trials are testing alternatives to current standard treatments\n\nBased on everything you've shared, I've updated your eligibility status for the matching trials. Is there anything else you'd like to tell me about your health history?";
  }

  return "Thank you for that information. To help find the most suitable clinical trials for you, could you tell me more about:\n\n- Your primary medical condition or diagnosis\n- Your age\n- Any current medications you're taking\n- Your general location (city/province)\n\nThe more details you provide, the better I can match you with relevant trials.";
}
