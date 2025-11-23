// File: frontend/components/app/order-card.tsx
'use client';

import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { cn } from '@/lib/utils';

interface OrderState {
  drinkType: string | null;
  size: string | null;
  milk: string | null;
  extras: string[];
  name: string | null;
  status: 'collecting' | 'brewing' | 'ready'; // backend should set this
}

interface OrderCardProps {
  orderState: OrderState;
  className?: string;
}

const sizeMap = {
  small: { height: 'h-24', label: 'Small (8oz)' },
  medium: { height: 'h-32', label: 'Medium (12oz)' },
  large: { height: 'h-40', label: 'Large (16oz)' },
};

const drinkIcons = {
  espresso: '☕',
  cappuccino: '☕',
  latte: '🥛',
  americano: '☕',
  'cold brew': '🧊',
  mocha: '🍫',
  macchiato: '☕',
};

export function OrderCard({ orderState, className }: OrderCardProps) {
  const [isAnimating, setIsAnimating] = useState(false);
  const [completionPercentage, setCompletionPercentage] = useState(0);

  // Calculate completion percentage for visual guidance only
  useEffect(() => {
    const fields = [
      orderState.drinkType,
      orderState.size,
      orderState.milk,
      orderState.name,
    ];
    const completed = fields.filter(Boolean).length;
    const total = 4;
    setCompletionPercentage((completed / total) * 100);
  }, [orderState.drinkType, orderState.size, orderState.milk, orderState.name]);

  // Trigger animation when order moves from collecting -> brewing/ready
  useEffect(() => {
    if (orderState.status !== 'collecting' && orderState.drinkType) {
      setIsAnimating(true);
      const timeout = setTimeout(() => setIsAnimating(false), 2000);
      return () => clearTimeout(timeout);
    }
  }, [orderState.status, orderState.drinkType]);

  const currentSize = orderState.size?.toLowerCase() as keyof typeof sizeMap;
  const cupHeight = currentSize ? sizeMap[currentSize].height : 'h-32';
  const drinkEmoji = orderState.drinkType
    ? drinkIcons[orderState.drinkType.toLowerCase() as keyof typeof drinkIcons] || '☕'
    : '☕';

  const getStatus = () => {
    // Prefer explicit status from backend / agent
    if (orderState.status === 'brewing') return 'Order confirmed. Brewing now...';
    if (orderState.status === 'ready') return 'Order ready to serve.';
    // Fallback to progress messages while collecting
    if (completionPercentage >= 75) return 'Almost done with your order...';
    if (completionPercentage > 0) return 'Building your order...';
    return 'Start your order';
  };

  const isOrderConfirmed = orderState.status !== 'collecting';

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn(
        'bg-linear-to-br from-amber-50 to-orange-50 dark:from-amber-950/30 dark:to-orange-950/30',
        'border border-amber-200 dark:border-amber-800',
        'rounded-2xl p-6 shadow-lg backdrop-blur-sm',
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 bg-linear-to-br from-amber-600 to-amber-800 rounded-lg flex items-center justify-center">
            <span className="text-white text-lg">{drinkEmoji}</span>
          </div>
          <div>
            <h3 className="font-bold text-lg text-amber-900 dark:text-amber-100">
              {orderState.name ? `${orderState.name}'s Order` : 'Your Order'}
            </h3>
            <p className="text-xs text-amber-700 dark:text-amber-300 font-mono">
              {getStatus()}
            </p>
          </div>
        </div>

        {/* Progress Circle */}
        <div className="relative w-12 h-12">
          <svg className="w-12 h-12 -rotate-90">
            <circle
              cx="24"
              cy="24"
              r="20"
              stroke="currentColor"
              strokeWidth="3"
              fill="none"
              className="text-amber-200 dark:text-amber-800"
            />
            <circle
              cx="24"
              cy="24"
              r="20"
              stroke="currentColor"
              strokeWidth="3"
              fill="none"
              strokeDasharray={`${2 * Math.PI * 20}`}
              strokeDashoffset={`${2 * Math.PI * 20 * (1 - completionPercentage / 100)}`}
              className="text-amber-600 dark:text-amber-400 transition-all duration-500"
              strokeLinecap="round"
            />
          </svg>
          <span className="absolute inset-0 flex items-center justify-center text-xs font-bold text-amber-900 dark:text-amber-100">
            {Math.round(completionPercentage)}%
          </span>
        </div>
      </div>

      {/* Visual Coffee Cup */}
      <div className="flex items-end justify-center gap-6 mb-6 min-h-[180px]">
        {/* Cup Visualization */}
        <div className="relative flex flex-col items-center">
          <AnimatePresence>
            {orderState.extras.some(e => e.toLowerCase().includes('whipped cream')) && (
              <motion.div
                initial={{ scale: 0, y: 20 }}
                animate={{ scale: 1, y: 0 }}
                exit={{ scale: 0 }}
                className="absolute -top-6 w-20 h-8 bg-linear-to-b from-white to-amber-50 rounded-t-full border-2 border-amber-300"
              >
                <div className="absolute inset-0 flex items-center justify-center text-xs">
                  🍦
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          <motion.div
            animate={isAnimating ? { scale: [1, 1.05, 1] } : {}}
            transition={{ duration: 0.5 }}
            className={cn(
              'relative w-24 bg-linear-to-b from-amber-100 to-amber-200 dark:from-amber-900 dark:to-amber-800',
              'rounded-b-3xl border-4 border-amber-400 dark:border-amber-600',
              'flex flex-col justify-end overflow-hidden',
              cupHeight,
              'transition-all duration-500'
            )}
          >
            {/* Coffee Fill */}
            <AnimatePresence>
              {orderState.drinkType && (
                <motion.div
                  initial={{ height: 0 }}
                  animate={{ height: '85%' }}
                  transition={{ duration: 1, ease: 'easeOut' }}
                  className={cn(
                    'w-full',
                    orderState.drinkType.toLowerCase().includes('mocha')
                      ? 'bg-linear-to-t from-amber-900 to-amber-700'
                      : orderState.drinkType.toLowerCase().includes('cold')
                        ? 'bg-linear-to-t from-amber-800 via-amber-600 to-amber-500'
                        : 'bg-linear-to-t from-amber-800 to-amber-600'
                  )}
                >
                  {/* Steam Effect */}
                  {!orderState.drinkType.toLowerCase().includes('cold') && (
                    <div className="absolute top-0 left-1/2 -translate-x-1/2 flex gap-2">
                      {[...Array(3)].map((_, i) => (
                        <motion.div
                          key={i}
                          className="w-1 h-4 bg-white/40 rounded-full"
                          animate={{
                            y: [-10, -30],
                            opacity: [0.4, 0],
                            scaleX: [1, 0.5],
                          }}
                          transition={{
                            duration: 2,
                            repeat: Infinity,
                            delay: i * 0.3,
                          }}
                        />
                      ))}
                    </div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>

            {/* Size Label */}
            {orderState.size && (
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-center">
                <span className="text-xs font-bold text-white drop-shadow-lg">
                  {currentSize && sizeMap[currentSize].label}
                </span>
              </div>
            )}
          </motion.div>

          {/* Cup Handle */}
          <div className="absolute right-0 top-1/3 w-6 h-12 border-4 border-amber-400 dark:border-amber-600 rounded-r-full border-l-0" />
        </div>

        {/* Order Details */}
        <div className="flex-1 space-y-2">
          {/* Drink Type */}
          <div className="flex items-center gap-2">
            <div
              className={cn(
                'w-6 h-6 rounded-full flex items-center justify-center text-xs',
                orderState.drinkType
                  ? 'bg-green-500 text-white'
                  : 'bg-gray-300 dark:bg-gray-700 text-gray-500'
              )}
            >
              {orderState.drinkType ? '✓' : '1'}
            </div>
            <div className="flex-1">
              <p className="text-xs text-muted-foreground">Drink</p>
              <p className="font-semibold text-sm capitalize">
                {orderState.drinkType || 'Selecting...'}
              </p>
            </div>
          </div>

          {/* Size */}
          <div className="flex items-center gap-2">
            <div
              className={cn(
                'w-6 h-6 rounded-full flex items-center justify-center text-xs',
                orderState.size
                  ? 'bg-green-500 text-white'
                  : 'bg-gray-300 dark:bg-gray-700 text-gray-500'
              )}
            >
              {orderState.size ? '✓' : '2'}
            </div>
            <div className="flex-1">
              <p className="text-xs text-muted-foreground">Size</p>
              <p className="font-semibold text-sm capitalize">
                {orderState.size || 'Selecting...'}
              </p>
            </div>
          </div>

          {/* Milk */}
          <div className="flex items-center gap-2">
            <div
              className={cn(
                'w-6 h-6 rounded-full flex items-center justify-center text-xs',
                orderState.milk
                  ? 'bg-green-500 text-white'
                  : 'bg-gray-300 dark:bg-gray-700 text-gray-500'
              )}
            >
              {orderState.milk ? '✓' : '3'}
            </div>
            <div className="flex-1">
              <p className="text-xs text-muted-foreground">Milk</p>
              <p className="font-semibold text-sm capitalize">
                {orderState.milk || 'Selecting...'}
              </p>
            </div>
          </div>

          {/* Extras */}
          {orderState.extras.length > 0 && (
            <div className="flex items-start gap-2">
              <div className="w-6 h-6 rounded-full flex items-center justify-center text-xs bg-blue-500 text-white">
                +
              </div>
              <div className="flex-1">
                <p className="text-xs text-muted-foreground">Extras</p>
                <div className="flex flex-wrap gap-1">
                  {orderState.extras.map((extra, i) => (
                    <motion.span
                      key={i}
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      className="text-xs px-2 py-0.5 bg-blue-500/20 text-blue-700 dark:text-blue-300 rounded-full"
                    >
                      {extra}
                    </motion.span>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Status Bar */}
      <div className="mt-4 space-y-2">
        <div className="h-2 bg-amber-200 dark:bg-amber-900 rounded-full overflow-hidden">
          <motion.div
            className="h-full bg-linear-to-r from-amber-500 to-amber-600"
            initial={{ width: 0 }}
            animate={{ width: `${completionPercentage}%` }}
            transition={{ duration: 0.5 }}
          />
        </div>

        {/* Only show "Order Complete" visual once backend marks it as not collecting */}
        {isOrderConfirmed && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center"
          >
            <p className="text-sm font-bold text-green-600 dark:text-green-400 flex items-center justify-center gap-2">
              <span className="animate-bounce">🎉</span>
              Order confirmed! Preparing your {orderState.drinkType}...
              <span className="animate-bounce">🎉</span>
            </p>
          </motion.div>
        )}
      </div>
    </motion.div>
  );
}
