'use client';

import React, { useEffect, useState } from 'react';
import { useRoomContext } from '@livekit/components-react';
import { motion, AnimatePresence } from 'motion/react';
import { cn } from '@/lib/utils';

const MotionDiv = motion.create('div');

interface OrderState {
  drinkType: string | null;
  size: string | null;
  milk: string | null;
  extras: string[];
  name: string | null;
}

interface OrderHistoryItem extends OrderState {
  timestamp: string;
  status: string;
}

interface BeverageDisplayProps {
  className?: string;
}

export function BeverageDisplay({ className }: BeverageDisplayProps) {
  const room = useRoomContext();
  const [currentOrder, setCurrentOrder] = useState<OrderState>({
    drinkType: null,
    size: null,
    milk: null,
    extras: [],
    name: null,
  });
  const [orderHistory, setOrderHistory] = useState<OrderHistoryItem[]>([]);
  const [isComplete, setIsComplete] = useState(false);

  useEffect(() => {
    if (!room) return;

    const handleData = (payload: Uint8Array, _participant: any, _kind: any, topic?: string) => {
      if (topic === 'coffee-order') {
        const decoder = new TextDecoder();
        const data = JSON.parse(decoder.decode(payload));

        if (data.type === 'order_update') {
          setCurrentOrder(data.order);
          if (data.history) {
            setOrderHistory(data.history);
          }
          setIsComplete(false);
        } else if (data.type === 'order_complete') {
          setCurrentOrder(data.order);
          if (data.history) {
            setOrderHistory(data.history);
          }
          setIsComplete(true);
        }
      }
    };

    room.on('dataReceived', handleData);

    return () => {
      room.off('dataReceived', handleData);
    };
  }, [room]);

  // Don't show anything if no order data
  if (!currentOrder.drinkType && !currentOrder.size && !currentOrder.milk && currentOrder.extras.length === 0) {
    return null;
  }

  // Size configurations
  const sizeConfig = {
    small: { height: 80, width: 60, fontSize: 'text-xs' },
    medium: { height: 100, width: 70, fontSize: 'text-sm' },
    large: { height: 120, width: 80, fontSize: 'text-base' },
  };

  const size = currentOrder.size || 'medium';
  const config = sizeConfig[size as keyof typeof sizeConfig] || sizeConfig.medium;

  // Drink colors based on type - Starbucks style
  const getDrinkColor = (drinkType: string | null) => {
    if (!drinkType) return 'bg-amber-700';
    const type = drinkType.toLowerCase();
    if (type.includes('latte') || type.includes('cappuccino')) return 'bg-amber-600';
    if (type.includes('mocha')) return 'bg-amber-800';
    if (type.includes('americano') || type.includes('espresso')) return 'bg-stone-800';
    if (type.includes('cold brew')) return 'bg-stone-700';
    if (type.includes('frappuccino') || type.includes('frappe')) return 'bg-emerald-100';
    return 'bg-amber-700';
  };

  const hasWhippedCream = currentOrder.extras.some((e) =>
    e.toLowerCase().includes('whipped cream')
  );

  return (
    <AnimatePresence mode="wait">
      <MotionDiv
        initial={{ opacity: 0, scale: 0.8, y: -20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.8, y: -20 }}
        transition={{ duration: 0.4, type: 'spring', stiffness: 200 }}
        className={cn(
          'bg-gradient-to-br from-background via-background to-emerald-50/30 dark:to-emerald-950/20',
          'backdrop-blur-xl rounded-xl border border-emerald-200/50 dark:border-emerald-800/50',
          'p-4 shadow-xl shadow-emerald-500/10 dark:shadow-emerald-500/5',
          'hover:shadow-emerald-500/20 dark:hover:shadow-emerald-500/10 transition-all duration-300',
          'max-w-xs',
          className
        )}
      >
        <div className="flex flex-col items-center space-y-6">
          {/* Order Title - Starbucks Style with animated gradient */}
          <div className="text-center space-y-1">
            <MotionDiv
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.1, type: 'spring', stiffness: 300 }}
            >
              <h3 className="text-lg font-black bg-gradient-to-r from-emerald-600 to-emerald-800 dark:from-emerald-400 dark:to-emerald-600 bg-clip-text text-transparent drop-shadow-sm">
                {isComplete ? '✓ Order Complete!' : '☕ Crafting...'}
              </h3>
            </MotionDiv>
            {currentOrder.name && (
              <MotionDiv
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="inline-block px-2 py-1 rounded-full bg-emerald-100 dark:bg-emerald-900/40 border border-emerald-300 dark:border-emerald-700"
              >
                <p className="text-xs font-bold text-emerald-800 dark:text-emerald-300">For: {currentOrder.name}</p>
              </MotionDiv>
            )}
          </div>

          {/* Coffee Cup Visualization with enhanced animations */}
          <div className="relative flex flex-col items-center">
            {/* Steam animation */}
            <div className="absolute -top-12 left-1/2 -translate-x-1/2 flex gap-2">
              {[0, 1, 2].map((i) => (
                <MotionDiv
                  key={i}
                  initial={{ opacity: 0, y: 0 }}
                  animate={{
                    opacity: [0, 0.5, 0],
                    y: [-10, -30],
                    scale: [0.8, 1.2],
                  }}
                  transition={{
                    duration: 2,
                    delay: i * 0.3,
                    repeat: Infinity,
                    repeatDelay: 0.5,
                  }}
                  className="w-2 h-4 bg-gradient-to-t from-emerald-300/40 to-transparent rounded-full"
                />
              ))}
            </div>
            {/* Whipped Cream on top with bounce animation */}
            {hasWhippedCream && (
              <MotionDiv
                initial={{ scale: 0, y: 20 }}
                animate={{ scale: 1, y: 0 }}
                transition={{ delay: 0.3, type: 'spring', stiffness: 400, damping: 10 }}
                className="relative z-10 -mb-4"
              >
                <div
                  className="bg-gradient-to-b from-white to-gray-100 dark:from-gray-100 dark:to-gray-300 rounded-full shadow-lg border-2 border-white/50"
                  style={{
                    width: config.width * 0.8,
                    height: config.width * 0.3,
                  }}
                >
                  {/* Whipped cream swirl detail */}
                  <div className="absolute top-1 left-1/2 -translate-x-1/2 w-3 h-3 bg-white rounded-full" />
                </div>
              </MotionDiv>
            )}

            {/* Coffee Cup with gradient and glow */}
            <MotionDiv
              initial={{ scale: 0, rotate: -10 }}
              animate={{ scale: 1, rotate: 0 }}
              transition={{ delay: 0.1, type: 'spring', stiffness: 200, damping: 15 }}
              className="relative group"
            >
              {/* Glow effect */}
              <div className="absolute inset-0 bg-gradient-to-b from-emerald-400/20 to-emerald-600/20 blur-xl rounded-full scale-110 group-hover:scale-125 transition-transform" />
              
              {/* Cup body - Starbucks style with enhanced design */}
              <div
                className={cn(
                  'rounded-b-3xl border-4 shadow-2xl relative overflow-hidden',
                  'border-emerald-700 dark:border-emerald-500',
                  'group-hover:border-emerald-600 dark:group-hover:border-emerald-400 transition-colors',
                  getDrinkColor(currentOrder.drinkType)
                )}
                style={{
                  width: config.width,
                  height: config.height,
                  borderTopLeftRadius: '20px',
                  borderTopRightRadius: '20px',
                }}
              >
                {/* Coffee liquid animation with wave effect */}
                <MotionDiv
                  initial={{ height: '0%' }}
                  animate={{ height: '100%' }}
                  transition={{ duration: 1.2, delay: 0.3, ease: 'easeOut' }}
                  className={cn('absolute bottom-0 w-full', getDrinkColor(currentOrder.drinkType))}
                >
                  {/* Wave effect on top of liquid */}
                  <div className="absolute top-0 w-full h-1 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-pulse" />
                </MotionDiv>

                {/* Foam/milk layer with texture */}
                {currentOrder.milk && (
                  <MotionDiv
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.8 }}
                    className="absolute top-0 w-full h-10 bg-gradient-to-b from-amber-100/90 via-amber-50/80 to-transparent dark:from-amber-200/70 dark:via-amber-100/60 border-b border-amber-200/50 dark:border-amber-300/30"
                  >
                    {/* Foam bubbles */}
                    <div className="absolute top-1 left-2 w-2 h-2 bg-white/40 rounded-full" />
                    <div className="absolute top-2 right-3 w-1.5 h-1.5 bg-white/30 rounded-full" />
                    <div className="absolute top-3 left-1/2 w-1 h-1 bg-white/20 rounded-full" />
                  </MotionDiv>
                )}
              </div>

              {/* Cup handle - Starbucks green with shadow */}
              <div
                className="absolute top-1/4 -right-6 w-8 h-12 border-4 border-emerald-700 dark:border-emerald-500 rounded-r-full shadow-lg group-hover:border-emerald-600 dark:group-hover:border-emerald-400 transition-colors"
                style={{ borderLeft: 'none' }}
              />

              {/* Size indicator badge - enhanced */}
              <MotionDiv
                initial={{ scale: 0, rotate: -180 }}
                animate={{ scale: 1, rotate: 0 }}
                transition={{ delay: 0.5, type: 'spring', stiffness: 300 }}
                className="absolute inset-0 flex items-center justify-center"
              >
                <span className="text-white font-black text-sm bg-gradient-to-br from-emerald-600 to-emerald-800 dark:from-emerald-500 dark:to-emerald-700 px-4 py-2 rounded-full shadow-lg border-2 border-white/30 backdrop-blur-sm">
                  {size?.toUpperCase()}
                </span>
              </MotionDiv>
            </MotionDiv>

            {/* Saucer - Starbucks green accent with shadow */}
            <MotionDiv
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: 0.4 }}
              className="bg-gradient-to-b from-emerald-700 to-emerald-900 dark:from-emerald-500 dark:to-emerald-700 rounded-full -mt-2 shadow-xl border-t-2 border-emerald-500/50"
              style={{
                width: config.width + 40,
                height: 14,
              }}
            />
          </div>

          {/* Order Details with modern cards */}
          <div className="w-full space-y-2 mt-4">
            {currentOrder.drinkType && (
              <MotionDiv
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.5 }}
                className="flex justify-between items-center p-2 rounded-lg bg-gradient-to-r from-emerald-50/50 to-transparent dark:from-emerald-900/20 border border-emerald-200/30 dark:border-emerald-800/30 hover:border-emerald-300 dark:hover:border-emerald-700 transition-colors"
              >
                <span className="text-xs font-medium text-muted-foreground">☕ Drink:</span>
                <span className="font-bold capitalize text-emerald-700 dark:text-emerald-400">{currentOrder.drinkType}</span>
              </MotionDiv>
            )}
            {currentOrder.size && (
              <MotionDiv
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.6 }}
                className="flex justify-between items-center p-2 rounded-lg bg-gradient-to-r from-emerald-50/50 to-transparent dark:from-emerald-900/20 border border-emerald-200/30 dark:border-emerald-800/30 hover:border-emerald-300 dark:hover:border-emerald-700 transition-colors"
              >
                <span className="text-xs font-medium text-muted-foreground">📏 Size:</span>
                <span className="font-bold capitalize text-emerald-700 dark:text-emerald-400">{currentOrder.size}</span>
              </MotionDiv>
            )}
            {currentOrder.milk && (
              <MotionDiv
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.7 }}
                className="flex justify-between items-center p-2 rounded-lg bg-gradient-to-r from-emerald-50/50 to-transparent dark:from-emerald-900/20 border border-emerald-200/30 dark:border-emerald-800/30 hover:border-emerald-300 dark:hover:border-emerald-700 transition-colors"
              >
                <span className="text-xs font-medium text-muted-foreground">🥛 Milk:</span>
                <span className="font-bold capitalize text-emerald-700 dark:text-emerald-400">{currentOrder.milk}</span>
              </MotionDiv>
            )}
            {currentOrder.extras.length > 0 && (
              <MotionDiv
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.8 }}
                className="flex justify-between items-start p-2 rounded-lg bg-gradient-to-r from-emerald-50/50 to-transparent dark:from-emerald-900/20 border border-emerald-200/30 dark:border-emerald-800/30 hover:border-emerald-300 dark:hover:border-emerald-700 transition-colors"
              >
                <span className="text-xs font-medium text-muted-foreground">✨ Extras:</span>
                <div className="text-right font-bold text-emerald-700 dark:text-emerald-400 space-y-1">
                  {currentOrder.extras.map((extra, i) => (
                    <div key={i} className="capitalize px-1.5 py-0.5 bg-emerald-100 dark:bg-emerald-900/40 rounded-full text-xs inline-block ml-1">
                      {extra}
                    </div>
                  ))}
                </div>
              </MotionDiv>
            )}
          </div>

          {/* Status indicator - Enhanced Starbucks style */}
          {isComplete && (
            <MotionDiv
              initial={{ scale: 0, rotate: -10 }}
              animate={{ scale: 1, rotate: 0 }}
              transition={{ delay: 0.9, type: 'spring', stiffness: 300 }}
              className="w-full bg-gradient-to-r from-emerald-500 to-emerald-600 dark:from-emerald-600 dark:to-emerald-700 border-2 border-emerald-400 dark:border-emerald-500 rounded-xl p-4 text-center shadow-lg relative overflow-hidden"
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
              <p className="text-base font-black text-white drop-shadow-md relative z-10">
                ✓ Your beverage is ready! 🎉
              </p>
            </MotionDiv>
          )}

          {/* Order History Section */}
          {orderHistory.length > 0 && (
            <MotionDiv
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1.0 }}
              className="w-full mt-4 pt-4 border-t border-emerald-200 dark:border-emerald-800"
            >
              <h4 className="text-sm font-bold text-emerald-700 dark:text-emerald-400 mb-2 flex items-center gap-1">
                <span>📋 Order History ({orderHistory.length})</span>
              </h4>
              <div className="space-y-2 max-h-40 overflow-y-auto pr-1">
                {orderHistory.map((order, index) => (
                  <MotionDiv
                    key={index}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 1.1 + index * 0.1 }}
                    className="p-2 rounded-md bg-emerald-50/50 dark:bg-emerald-900/20 border border-emerald-200/30 dark:border-emerald-800/30"
                  >
                    <div className="flex justify-between items-start">
                      <div>
                        <div className="font-bold text-emerald-800 dark:text-emerald-300 text-sm">
                          {order.size} {order.drinkType}
                        </div>
                        <div className="text-xs text-muted-foreground mt-1">
                          For: {order.name}
                        </div>
                      </div>
                      <div className="text-right text-xs text-emerald-600 dark:text-emerald-400">
                        {new Date(order.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </div>
                    </div>
                    {(order.milk || order.extras.length > 0) && (
                      <div className="mt-1 text-xs text-muted-foreground flex flex-wrap gap-1">
                        {order.milk && <span className="px-1.5 py-0.5 bg-emerald-100 dark:bg-emerald-900/30 rounded-full">{order.milk}</span>}
                        {order.extras.map((extra, i) => (
                          <span key={i} className="px-1.5 py-0.5 bg-emerald-100 dark:bg-emerald-900/30 rounded-full">
                            {extra}
                          </span>
                        ))}
                      </div>
                    )}
                  </MotionDiv>
                ))}
              </div>
            </MotionDiv>
          )}
        </div>
      </MotionDiv>
    </AnimatePresence>
  );
}
