// frontend/components/app/welcome-view.tsx

import { Button } from '@/components/livekit/button'; // Keep this import for the Button component

interface WelcomeViewProps {
  startButtonText: string; // This prop is now unused since we're hardcoding, but you can leave it.
  onStartCall: () => void;
}

export const WelcomeView = ({
  startButtonText, // This prop is effectively ignored now for the button text.
  onStartCall,
  ref,
}: React.ComponentProps<'div'> & WelcomeViewProps) => {
  return (
    <div ref={ref}>
      <section className="bg-background flex flex-col items-center justify-center text-center">
        
        {/* --- THIS IS WHERE YOUR GLITCHLANDS LOGO GOES, REPLACING THE SVG ICON --- */}
        <img 
            src="/transparent-removebg-preview.png" // Path to your logo in the public folder
            alt="The Improv Battle Logo" 
            // Tailwind classes to center and size the logo. Adjust 'w-96' for desired width.
            className="mx-auto mb-6 w-96 max-w-full h-auto" 
        />
        {/* ----------------------------------------------------------------------- */}

        <p className="text-foreground max-w-prose pt-1 leading-6 font-medium">
          {/* --- UPDATED INTRODUCTORY TEXT --- */}
          Begin your Game.
        </p>

        <Button 
          variant="primary" 
          size="lg" 
          onClick={onStartCall} 
          className="mt-6 w-64 font-mono"
        >
          {/* --- UPDATED BUTTON TEXT --- */}
          Start Improv Battle
        </Button>
      </section>

      <div className="fixed bottom-5 left-0 flex w-full items-center justify-center">
        <p className="text-muted-foreground max-w-prose pt-1 text-xs leading-5 font-normal text-pretty md:text-sm">
          Need help getting set up? Check out the{' '}
          <a
            target="_blank"
            rel="noopener noreferrer"
            href="https://docs.livekit.io/agents/start/voice-ai/"
            className="underline"
          >
            Voice AI quickstart
          </a>
          .
        </p>
      </div>
    </div>
  );
};