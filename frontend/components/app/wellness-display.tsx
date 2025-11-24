'use client';

import React, { useEffect, useState } from 'react';
import { useRoomContext } from '@livekit/components-react';
import { motion, AnimatePresence } from 'motion/react';
import { cn } from '@/lib/utils';

const MotionDiv = motion.create('div');

interface WellnessCheckin {
  mood: string | null;
  energy: string | null;
  stress: string | null;
  objectives: string[];
  notes: string | null;
}

interface WellnessDisplayProps {
  className?: string;
}

export function WellnessDisplay({ className }: WellnessDisplayProps) {
  const room = useRoomContext();
  const [checkin, setCheckin] = useState<WellnessCheckin>({
    mood: null,
    energy: null,
    stress: null,
    objectives: [],
    notes: null,
  });
  const [isComplete, setIsComplete] = useState(false);

  useEffect(() => {
    if (!room) return;

    const handleData = (payload: Uint8Array, _participant: any, _kind: any, topic?: string) => {
      if (topic === 'wellness-checkin') {
        const decoder = new TextDecoder();
        const data = JSON.parse(decoder.decode(payload));

        if (data.type === 'checkin_update') {
          setCheckin(data.checkin);
          setIsComplete(false);
        } else if (data.type === 'checkin_complete') {
          setCheckin(data.checkin);
          setIsComplete(true);
        }
      }
    };

    room.on('dataReceived', handleData);

    return () => {
      room.off('dataReceived', handleData);
    };
  }, [room]);

  // Don't show anything if no data
  if (!checkin.mood && !checkin.energy && !checkin.stress && checkin.objectives.length === 0) {
    return null;
  }

  return (
    <AnimatePresence mode="wait">
      <MotionDiv
        initial={{ opacity: 0, scale: 0.8, y: -20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.8, y: -20 }}
        transition={{ duration: 0.4, type: 'spring', stiffness: 200 }}
        className={cn(
          'bg-gradient-to-br from-background via-background to-blue-50/30 dark:to-blue-950/20',
          'backdrop-blur-xl rounded-xl border border-blue-200/50 dark:border-blue-800/50',
          'p-4 shadow-xl shadow-blue-500/10 dark:shadow-blue-500/5',
          'hover:shadow-blue-500/20 dark:hover:shadow-blue-500/10 transition-all duration-300',
          'max-w-xs',
          className
        )}
      >
        <div className="flex flex-col items-center space-y-4">
          {/* Title */}
          <div className="text-center space-y-1">
            <MotionDiv
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.1, type: 'spring', stiffness: 300 }}
            >
              <h3 className="text-lg font-black bg-gradient-to-r from-blue-600 to-blue-800 dark:from-blue-400 dark:to-blue-600 bg-clip-text text-transparent drop-shadow-sm">
                {isComplete ? '✓ Check-in Complete!' : '💙 Daily Wellness Check-in'}
              </h3>
            </MotionDiv>
          </div>

          {/* Check-in Items */}
          <div className="w-full space-y-2">
            {/* Mood */}
            {checkin.mood && (
              <MotionDiv
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.2 }}
                className="flex items-center gap-2 p-2 rounded-lg bg-gradient-to-r from-blue-50/50 to-transparent dark:from-blue-900/20 border border-blue-200/30 dark:border-blue-800/30"
              >
                <div className="flex-shrink-0 w-6 h-6 rounded-full bg-blue-500 flex items-center justify-center">
                  <span className="text-white text-sm">✓</span>
                </div>
                <div className="flex-1">
                  <p className="text-xs font-medium text-muted-foreground">Mood</p>
                  <p className="font-bold capitalize text-blue-700 dark:text-blue-400">{checkin.mood}</p>
                </div>
              </MotionDiv>
            )}

            {/* Energy */}
            {checkin.energy && (
              <MotionDiv
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.3 }}
                className="flex items-center gap-2 p-2 rounded-lg bg-gradient-to-r from-blue-50/50 to-transparent dark:from-blue-900/20 border border-blue-200/30 dark:border-blue-800/30"
              >
                <div className="flex-shrink-0 w-6 h-6 rounded-full bg-blue-500 flex items-center justify-center">
                  <span className="text-white text-sm">✓</span>
                </div>
                <div className="flex-1">
                  <p className="text-xs font-medium text-muted-foreground">Energy</p>
                  <p className="font-bold capitalize text-blue-700 dark:text-blue-400">{checkin.energy}</p>
                </div>
              </MotionDiv>
            )}

            {/* Stress */}
            {checkin.stress && (
              <MotionDiv
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.4 }}
                className="flex items-center gap-2 p-2 rounded-lg bg-gradient-to-r from-blue-50/50 to-transparent dark:from-blue-900/20 border border-blue-200/30 dark:border-blue-800/30"
              >
                <div className="flex-shrink-0 w-6 h-6 rounded-full bg-blue-500 flex items-center justify-center">
                  <span className="text-white text-sm">✓</span>
                </div>
                <div className="flex-1">
                  <p className="text-xs font-medium text-muted-foreground">Stress/Concerns</p>
                  <p className="font-bold text-blue-700 dark:text-blue-400">{checkin.stress}</p>
                </div>
              </MotionDiv>
            )}

            {/* Objectives */}
            {checkin.objectives.length > 0 && (
              <MotionDiv
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.5 }}
                className="p-2 rounded-lg bg-gradient-to-r from-blue-50/50 to-transparent dark:from-blue-900/20 border border-blue-200/30 dark:border-blue-800/30"
              >
                <div className="flex items-start gap-2">
                  <div className="flex-shrink-0 w-6 h-6 rounded-full bg-blue-500 flex items-center justify-center mt-0.5">
                    <span className="text-white text-sm">✓</span>
                  </div>
                  <div className="flex-1">
                    <p className="text-xs font-medium text-muted-foreground mb-1">Today's Goals ({checkin.objectives.length})</p>
                    <div className="space-y-1">
                      {checkin.objectives.map((objective, i) => (
                        <div key={i} className="flex items-start gap-1">
                          <span className="text-blue-600 dark:text-blue-400 text-xs mt-0.5">•</span>
                          <p className="text-sm font-medium text-blue-700 dark:text-blue-400 capitalize">{objective}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </MotionDiv>
            )}

            {/* Notes */}
            {checkin.notes && (
              <MotionDiv
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.6 }}
                className="flex items-center gap-2 p-2 rounded-lg bg-gradient-to-r from-blue-50/50 to-transparent dark:from-blue-900/20 border border-blue-200/30 dark:border-blue-800/30"
              >
                <div className="flex-shrink-0 w-6 h-6 rounded-full bg-blue-500 flex items-center justify-center">
                  <span className="text-white text-sm">✓</span>
                </div>
                <div className="flex-1">
                  <p className="text-xs font-medium text-muted-foreground">Notes</p>
                  <p className="text-sm font-medium text-blue-700 dark:text-blue-400">{checkin.notes}</p>
                </div>
              </MotionDiv>
            )}
          </div>

          {/* Completion Status */}
          {isComplete && (
            <MotionDiv
              initial={{ scale: 0, rotate: -10 }}
              animate={{ scale: 1, rotate: 0 }}
              transition={{ delay: 0.7, type: 'spring', stiffness: 300 }}
              className="w-full bg-gradient-to-r from-blue-500 to-blue-600 dark:from-blue-600 dark:to-blue-700 border-2 border-blue-400 dark:border-blue-500 rounded-xl p-3 text-center shadow-lg relative overflow-hidden"
            >
              {/* Animated background shine */}
              <MotionDiv
                animate={{
                  x: ['-100%', '200%'],
                }}
                transition={{
                  duration: 2,
                  repeat: Infinity,
                  repeatDelay: 1,
                }}
                className="absolute inset-0 w-1/2 bg-gradient-to-r from-transparent via-white/20 to-transparent skew-x-12"
              />
              <p className="text-sm font-black text-white drop-shadow-md relative z-10">
                ✓ Check-in saved! Have a great day! 🌟
              </p>
            </MotionDiv>
          )}
        </div>
      </MotionDiv>
    </AnimatePresence>
  );
}
