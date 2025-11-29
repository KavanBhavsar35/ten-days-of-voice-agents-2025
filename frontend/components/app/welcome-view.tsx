import { Button } from '@/components/livekit/button';
import { useState, useEffect } from 'react';
import { motion } from 'motion/react';
import { Sword, Sparkles, Mic, Shield, Flame } from 'lucide-react';

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
          scale: [1, 1.05, 1],
          rotate: [0, 3, -3, 0],
        }}
        transition={{
          duration: 4,
          repeat: Infinity,
          ease: "easeInOut"
        }}
        className="relative"
      >
        <div className="absolute inset-0 bg-linear-to-r from-amber-600/30 to-orange-600/30 blur-3xl rounded-full" />
        <div className="relative bg-linear-to-br from-amber-900/30 to-orange-900/30 p-8 rounded-3xl border-2 border-amber-600/50 shadow-2xl">
          <Sword className="w-20 h-20 text-amber-400" strokeWidth={1.5} />
          <motion.div
            animate={{
              y: [0, -10, 0],
              opacity: [0.5, 1, 0.5],
              rotate: [0, 10, -10, 0]
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: "easeInOut"
            }}
            className="absolute -top-2 -right-2"
          >
            <Sparkles className="w-8 h-8 text-yellow-400 fill-yellow-400" />
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
            <Shield className="w-6 h-6 text-amber-500" />
          </motion.div>
          <motion.div
            animate={{
              scale: [1, 1.3, 1],
              opacity: [0.4, 0.8, 0.4]
            }}
            transition={{
              duration: 1.5,
              repeat: Infinity,
              ease: "easeInOut",
            }}
            className="absolute top-0 left-0"
          >
            <Flame className="w-5 h-5 text-orange-500" />
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
  const [particles, setParticles] = useState<Array<{ x: number, y: number, targetX: number, targetY: number, duration: number }>>([]);

  useEffect(() => {
    // Generate particles only on client side
    const newParticles = [...Array(50)].map(() => ({
      x: Math.random() * window.innerWidth,
      y: Math.random() * window.innerHeight,
      targetX: Math.random() * window.innerWidth,
      targetY: Math.random() * window.innerHeight,
      duration: 20 + Math.random() * 40,
    }));
    setParticles(newParticles);
  }, []);

  return (
    <div ref={ref} className="relative h-full w-full overflow-hidden bg-black">
      {/* Cyberpunk Grid Background */}
      <div className="absolute inset-0 bg-[linear-linear(rgba(0,255,255,0.03)_1px,transparent_1px),linear-linear(90deg,rgba(0,255,255,0.03)_1px,transparent_1px)] bg-size-[50px_50px]" />

      {/* Neon Glow Orbs */}
      <div className="absolute inset-0 overflow-hidden">
        <motion.div
          className="absolute top-20 left-20 w-96 h-96 bg-cyan-500 rounded-full filter blur-[120px] opacity-20"
          animate={{
            scale: [1, 1.2, 1],
            opacity: [0.2, 0.3, 0.2],
          }}
          transition={{
            duration: 8,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
        />
        <motion.div
          className="absolute bottom-20 right-20 w-96 h-96 bg-purple-500 rounded-full filter blur-[120px] opacity-20"
          animate={{
            scale: [1, 1.3, 1],
            opacity: [0.2, 0.3, 0.2],
          }}
          transition={{
            duration: 10,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
        />
        <motion.div
          className="absolute top-1/2 left-1/2 w-96 h-96 bg-pink-500 rounded-full filter blur-[120px] opacity-15"
          animate={{
            scale: [1, 1.4, 1],
            x: [-50, 50, -50],
          }}
          transition={{
            duration: 12,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
        />
      </div>

      {/* Floating Particles */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {particles.map((particle, i) => (
          <motion.div
            key={i}
            className="absolute w-0.5 h-0.5 bg-cyan-400 rounded-full"
            style={{
              boxShadow: '0 0 2px #00ffff',
            }}
            initial={{
              x: particle.x,
              y: particle.y,
            }}
            animate={{
              y: [null, particle.targetY],
              x: [null, particle.targetX],
              opacity: [0, 0.6, 0],
            }}
            transition={{
              duration: particle.duration,
              repeat: Infinity,
              ease: 'linear',
            }}
          />
        ))}
      </div>

      <section className="relative flex flex-col items-center justify-center text-center px-4 h-full">
        {/* Cyberpunk Card */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: 'easeOut' }}
          className="cyber-card max-w-2xl w-full p-12"
        >
          {/* Icon */}
          {/* <motion.div
            initial={{ scale: 0, rotate: -180 }}
            animate={{ scale: 1, rotate: 0 }}
            transition={{ duration: 0.6, delay: 0.2, type: "spring", bounce: 0.4 }}
            className="mb-8 inline-block"
          >
            <div className="relative">
              <div className="absolute inset-0 bg-cyan-500/30 blur-xl rounded-lg" />
              <div className="relative bg-black/80 backdrop-blur-sm p-6 rounded-lg border-2 border-cyan-500/50 neon-border">
                <Sword className="w-16 h-16 text-cyan-400" strokeWidth={2} />
              </div>
            </div>
          </motion.div> */}
          <WelcomeImage />

          {/* Title */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, duration: 0.5 }}
          >
            <h1 className="text-5xl md:text-6xl font-black mb-4 text-cyan-400 neon-text tracking-tight">
              VOICE GAME MASTER
            </h1>
            <p className="text-xl text-purple-400 font-bold mb-2 tracking-wide">
              EPIC D&D-STYLE ADVENTURE
            </p>
            <p className="text-gray-400 max-w-md mx-auto text-base leading-relaxed">
              Embark on a voice-powered fantasy adventure where your choices shape the story
            </p>
          </motion.div>

          {/* Button */}
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.5, duration: 0.4 }}
            className="mt-10"
          >
            <Button
              variant="primary"
              size="lg"
              onClick={onStartCall}
              className="cyber-button w-64 h-14 text-lg font-bold transition-all duration-300 hover:scale-105"
            >
              <Mic className="w-5 h-5 mr-2" />
              {startButtonText}
            </Button>
          </motion.div>

          {/* Feature Tags */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.7, duration: 0.5 }}
            className="mt-10 flex flex-wrap gap-3 justify-center"
          >
            {[
              { icon: '⚔️', text: 'COMBAT' },
              { icon: '🗺️', text: 'EXPLORATION' },
              { icon: '🎭', text: 'ROLEPLAY' },
              { icon: '🎲', text: 'DICE ROLLS' }
            ].map((item, i) => (
              <motion.span
                key={item.text}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                whileHover={{ scale: 1.05, y: -2 }}
                transition={{ delay: 0.8 + i * 0.1 }}
                className="cyber-tag px-4 py-2 text-xs font-bold tracking-wider"
              >
                <span className="mr-1.5">{item.icon}</span>
                {item.text}
              </motion.span>
            ))}
          </motion.div>
        </motion.div>
      </section>
    </div>
  );
};
