'use client';

import { useEffect, useRef } from 'react';
import { Message } from '@/types';
import { ChatMessage } from './ChatMessage';
import { ScrollArea } from '@/components/ui/scroll-area';

interface MessageListProps {
  messages: Message[];
  isLoading?: boolean;
}

// Loading animation component
function LoadingIndicator() {
  return (
    <div className="flex items-start gap-2 animate-in fade-in duration-300">
      <div className="bg-muted rounded-md px-3 py-2">
        <div className="flex items-center gap-1.5">
          <div className="flex gap-0.5">
            <span className="w-1.5 h-1.5 bg-primary/60 rounded-full animate-bounce [animation-delay:-0.3s]"></span>
            <span className="w-1.5 h-1.5 bg-primary/60 rounded-full animate-bounce [animation-delay:-0.15s]"></span>
            <span className="w-1.5 h-1.5 bg-primary/60 rounded-full animate-bounce"></span>
          </div>
          <span className="text-xs text-muted-foreground ml-1">Processing...</span>
        </div>
      </div>
    </div>
  );
}

export function MessageList({ messages, isLoading = false }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  return (
    <ScrollArea className="h-full">
      <div className="flex flex-col gap-4 p-4">
        {messages.length === 0 && !isLoading ? (
          <div className="flex flex-col items-center justify-center text-center text-muted-foreground py-12">
            <h3 className="text-lg font-medium mb-2">Welcome to Clinical Trial Matcher</h3>
            <p className="max-w-md">
              Tell me about your medical condition, and I&apos;ll help you find relevant clinical trials.
              You can share details like your diagnosis, age, current medications, and location.
            </p>
          </div>
        ) : (
          <>
            {messages.map((message) => (
              <ChatMessage key={message.id} message={message} />
            ))}
            {isLoading && <LoadingIndicator />}
          </>
        )}
        <div ref={bottomRef} />
      </div>
    </ScrollArea>
  );
}
