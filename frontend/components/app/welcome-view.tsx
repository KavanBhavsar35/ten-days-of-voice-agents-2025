import { Button } from '@/components/livekit/button';

import { motion } from 'motion/react';
import { ShoppingBag, Sparkles, Mic } from 'lucide-react';

function WelcomeImage() {
  return (
    <motion.div
      initial={{ scale: 0, rotate: -180 }}
      animate={{ scale: 1, rotate: 0 }}
      transition={{ duration: 0.6, type: "spring", bounce: 0.5 }}
      className="relative mb-8"
    >
      <motion.div
        animate={{
          scale: [1, 1.1, 1],
          rotate: [0, 5, -5, 0],
        }}
        transition={{
          duration: 4,
          repeat: Infinity,
          ease: "easeInOut"
        }}
        className="relative"
      >
        <div className="absolute inset-0 bg-gradient-to-r from-primary/30 to-accent/30 blur-3xl rounded-full" />
        <div className="relative bg-gradient-to-br from-primary/20 to-accent/20 p-8 rounded-3xl border-2 border-primary/30 shadow-2xl">
          <ShoppingBag className="w-20 h-20 text-primary" strokeWidth={1.5} />
          <motion.div
            animate={{
              y: [0, -10, 0],
              opacity: [0.5, 1, 0.5]
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: "easeInOut"
            }}
            className="absolute -top-2 -right-2"
          >
            <Sparkles className="w-8 h-8 text-yellow-500 fill-yellow-500" />
          </motion.div>
          <motion.div
            animate={{
              scale: [1, 1.2, 1],
              opacity: [0.3, 0.6, 0.3]
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: "easeInOut",
              delay: 0.5
            }}
            className="absolute -bottom-2 -left-2"
          >
            <Mic className="w-6 h-6 text-primary" />
          </motion.div>
        </div>
      </motion.div>
    </motion.div>
  );
}

interface WelcomeViewProps {
  startButtonText: string;
  onStartCall: () => void;
}

export const WelcomeView = ({
  startButtonText,
  onStartCall,
  ref,
}: React.ComponentProps<'div'> & WelcomeViewProps) => {
  return (
    <div ref={ref}>
      <section className="bg-background flex flex-col items-center justify-center text-center px-4">
        <WelcomeImage />

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.5 }}
        >
          <h1 className="text-4xl md:text-5xl font-black mb-3 bg-gradient-to-r from-primary via-accent to-primary bg-clip-text text-transparent">
            QuickMart Express
          </h1>
          <p className="text-foreground/80 max-w-prose text-lg leading-7 font-medium mb-2">
            🛒 Your AI Shopping Assistant
          </p>
          <p className="text-muted-foreground max-w-md text-sm leading-6">
            Order groceries naturally with voice! Just say what you need, and I'll add it to your cart.
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.5, duration: 0.4, type: "spring", bounce: 0.4 }}
        >
          <Button 
            variant="primary" 
            size="lg" 
            onClick={onStartCall} 
            className="mt-8 w-72 h-14 text-lg font-bold shadow-xl hover:shadow-2xl transition-all duration-300 hover:scale-105 bg-gradient-to-r from-primary to-accent"
          >
            <Mic className="w-5 h-5 mr-2" />
            {startButtonText}
          </Button>
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.7, duration: 0.5 }}
          className="mt-8 flex flex-wrap gap-3 justify-center max-w-2xl"
        >
          {['🥖 Groceries', '🥛 Dairy', '🍕 Ready-to-Eat', '🍎 Fresh Produce'].map((item, i) => (
            <motion.span
              key={item}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.8 + i * 0.1 }}
              className="px-4 py-2 bg-accent/20 border border-primary/30 rounded-full text-sm font-medium text-foreground/80"
            >
              {item}
            </motion.span>
          ))}
        </motion.div>
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
