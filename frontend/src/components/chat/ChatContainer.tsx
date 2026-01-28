'use client';

import { useState, useEffect } from 'react';
import { Message, ClinicalTrial } from '@/types';
import { MessageList } from './MessageList';
import { ChatInput } from './ChatInput';
import { apiService } from '@/lib/api';
import { getMockResponse, mockTrials } from '@/lib/mockData';

interface ChatContainerProps {
  onTrialsFound: (trials: ClinicalTrial[]) => void;
}

// Set to true to use real backend, false for mock data
const USE_REAL_API = process.env.NEXT_PUBLIC_USE_REAL_API === 'true';

// Initial greeting message - shown immediately when chat loads
const INITIAL_GREETING: Message = {
  id: 'greeting',
  role: 'assistant',
  content: "Hi there! I'm your Clinical Trial Assistant. I'm here to help you find clinical trials that might be a good fit for your situation.\n\nTo get started, could you tell me about the medical condition you're looking for a trial for?",
  timestamp: new Date(),
};

export function ChatContainer({ onTrialsFound }: ChatContainerProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  // Show greeting when component mounts
  useEffect(() => {
    setMessages([INITIAL_GREETING]);
  }, []);

  // Helper to add messages with delay for natural feel
  const addMessagesWithDelay = async (responses: string[], trials: ClinicalTrial[]) => {
    for (let i = 0; i < responses.length; i++) {
      const assistantMessage: Message = {
        id: (Date.now() + i + 1).toString(),
        role: 'assistant',
        content: responses[i],
        timestamp: new Date(),
      };

      // Show loading before each message except the first
      if (i > 0) {
        setIsLoading(true);
        await new Promise((resolve) => setTimeout(resolve, 800)); // Typing delay
      }

      setMessages((prev) => [...prev, assistantMessage]);
      setIsLoading(false);

      // Small pause between messages for readability
      if (i < responses.length - 1) {
        await new Promise((resolve) => setTimeout(resolve, 300));
      }
    }

    // Show trials after all messages are displayed
    if (trials.length > 0) {
      onTrialsFound(trials);
    }
  };

  const handleSendMessage = async (content: string) => {
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    try {
      let responses: string[] = [];
      let trials: ClinicalTrial[] = [];

      if (USE_REAL_API) {
        // Use real backend API
        const result = await apiService.sendMessage(content);
        responses = result.responses;
        trials = result.trials;
      } else {
        // Use mock data for development
        await new Promise((resolve) => setTimeout(resolve, 1000));
        responses = [getMockResponse(content)];

        // Show mock trials after a few messages
        if (messages.length >= 2) {
          trials = mockTrials;
        }
      }

      // Add messages with natural delay
      setIsLoading(false);
      await addMessagesWithDelay(responses, trials);

    } catch (error) {
      console.error('Error sending message:', error);

      // Fallback to mock on error
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: "I'm having trouble connecting to the server. Please try again in a moment.",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, assistantMessage]);
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full border rounded-lg bg-card overflow-hidden">
      <div className="p-4 border-b shrink-0">
        <h2 className="font-semibold">Chat with Clinical Trial Assistant</h2>
        <p className="text-sm text-muted-foreground">
          Share your health information to find matching clinical trials
        </p>
      </div>
      <div className="flex-1 overflow-hidden min-h-0">
        <MessageList messages={messages} isLoading={isLoading} />
      </div>
      <div className="shrink-0">
        <ChatInput onSendMessage={handleSendMessage} disabled={isLoading} />
      </div>
    </div>
  );
}
