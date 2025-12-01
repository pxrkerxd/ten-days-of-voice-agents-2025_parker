import { useState } from 'react';
// Error: Could not resolve '@/components/livekit/button'. Replaced with standard HTML button below.
// import { Button } from '@/components/livekit/button'; 

// IMPORTANT: Updated the signature of onStartCall to accept the user's name
interface WelcomeViewProps {
  startButtonText: string;
  onStartCall: (userName: string) => void;
}

export const WelcomeView = ({
  startButtonText,
  onStartCall,
  ref,
}: React.ComponentProps<'div'> & WelcomeViewProps) => {
  const [userName, setUserName] = useState('');

  // New handler: validates the name and passes it to the parent
  const handleStartCall = () => {
    if (userName.trim()) {
      onStartCall(userName.trim());
    }
  };

  return (
    // ----------------------------------------------------
    // 🎯 MAIN CONTAINER
    // ----------------------------------------------------
    <div
      ref={ref}
      className="
        min-h-screen flex flex-col items-center justify-center
        bg-cover bg-center bg-fixed bg-no-repeat
        bg-[url('/improv_battle_background.jpg')]
        relative
      "
    >
      
      {/* Dark Overlay */}
      <div className="absolute inset-0 bg-black opacity-0"></div> 

      <section className="
        relative z-10
        p-8 bg-white/20 backdrop-blur-sm rounded-xl shadow-2xl 
        flex flex-col items-center justify-center text-center
      ">
        
        

        {/* --- USER NAME INPUT FIELD --- */}
        <div className="w-full max-w-xs mb-6">
          <label htmlFor="userName" className="block text-xl font-semibold text-white mb-2 shadow-text-lg">
            Enter Your Name
          </label>
          <input
            id="userName"
            type="text"
            value={userName}
            onChange={(e) => setUserName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && userName.trim()) {
                handleStartCall();
              }
            }}
            placeholder="e.g., Jane Doe"
            className="
              w-full p-3 text-lg text-gray-900 bg-white
              border-2 border-green-500 rounded-lg shadow-md
              focus:ring-green-500 focus:border-green-500 outline-none
            "
            maxLength={30}
          />
        </div>
        
        {/* BUTTON: Replaced external Button component with native <button> */}
        <button
          onClick={handleStartCall}
          disabled={!userName.trim()} // Disabled if name is empty
          className="
            mt-4 w-64 font-mono
            py-3 px-6 text-lg rounded-xl font-bold transition-colors
            bg-gray-900 hover:bg-gray-700
            text-white
            border-2 border-gray-900
            shadow-lg shadow-gray-500/50
            disabled:opacity-50 disabled:cursor-not-allowed
            focus:outline-none focus:ring-4 focus:ring-green-500/50
          "
        >
          {startButtonText || "Start Improv Battle"}
        </button>
        
        <p className="mt-2 text-sm text-white/80">
          The name will be used to identify you in the Improv Battle.
        </p>

      </section>
    </div>
  );
};
