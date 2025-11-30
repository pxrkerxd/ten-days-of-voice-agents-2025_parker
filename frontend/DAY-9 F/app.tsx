'use client';

import { RoomAudioRenderer, StartAudio } from '@livekit/components-react';
import type { AppConfig } from '@/app-config';
import { SessionProvider } from '@/components/app/session-provider';
import { ViewController } from '@/components/app/view-controller';
import { Toaster } from '@/components/livekit/toaster';

interface AppProps {
  appConfig: AppConfig;
}

export function App({ appConfig }: AppProps) {
  return (
    <SessionProvider appConfig={appConfig}>
      {/* Set the background image on the main container */}
      <main 
        className="grid h-svh grid-cols-1 place-content-center p-8" // Added p-8 for padding
        style={{
          backgroundImage: 'url(/Walmart_bg.jpg)',
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          backgroundRepeat: 'no-repeat',
        }}
      >
        <div className="flex flex-col items-center justify-center w-full max-w-4xl mx-auto space-y-6">
          
        
          
          {/* The main application view */}
          <ViewController />
        </div>
      </main>
      
      <StartAudio label="Start Audio" />
      <RoomAudioRenderer />
      <Toaster />
    </SessionProvider>
  );
}