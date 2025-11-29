'use client';

import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Sword, Heart, Shield, Backpack, Scroll, Sparkles, Flame } from 'lucide-react';
import { cn } from '@/lib/utils';

interface CharacterSheetProps {
  className?: string;
  messages: any[];
}

interface PlayerStats {
  name: string;
  class: string;
  level: number;
  hp: number;
  maxHp: number;
  stats: {
    strength: number;
    intelligence: number;
    dexterity: number;
    charisma: number;
    luck: number;
  };
  inventory: string[];
  gold: number;
}

export function CharacterSheet({ className, messages }: CharacterSheetProps) {
  const [player, setPlayer] = useState<PlayerStats>({
    name: 'Draven Shadowblade',
    class: 'Shadow Knight',
    level: 5,
    hp: 150,
    maxHp: 150,
    stats: {
      strength: 18,
      intelligence: 14,
      dexterity: 16,
      charisma: 12,
      luck: 15,
    },
    inventory: [
      '⚔️ Shadowfang Blade',
      '🛡️ Dragon Scale Armor',
      '💎 Crystal of Light',
      '🧪 Phoenix Elixir x3',
      '📜 Scroll of Fireball',
      '🗝️ Ancient Key'
    ],
    gold: 500,
  });
  const [isOpen, setIsOpen] = useState(true);
  const [lastRoll, setLastRoll] = useState<string | null>(null);

  useEffect(() => {
    // Parse messages for stat updates
    const lastMessage = messages[messages.length - 1];
    if (!lastMessage || !lastMessage.message) return;

    const messageText = lastMessage.message;

    // Check for HP changes
    if (messageText.includes('HP') || messageText.includes('damage') || messageText.includes('heal')) {
      const hpMatch = messageText.match(/Now at (\d+)\/(\d+) HP/i);
      if (hpMatch) {
        setPlayer((prev) => ({
          ...prev,
          hp: parseInt(hpMatch[1]),
          maxHp: parseInt(hpMatch[2]),
        }));
      }
    }

    // Check for dice rolls
    if (messageText.includes('🎲') || messageText.includes('Rolled')) {
      const rollMatch = messageText.match(/🎲.*?(?:SUCCESS|FAILED)/i);
      if (rollMatch) {
        setLastRoll(rollMatch[0]);
        setTimeout(() => setLastRoll(null), 5000);
      }
    }

    // Check for inventory changes
    if (messageText.includes('Added') && messageText.includes('inventory')) {
      const itemMatch = messageText.match(/Added ([^to]+) to inventory/i);
      if (itemMatch) {
        const item = itemMatch[1].trim();
        setPlayer((prev) => ({
          ...prev,
          inventory: [...prev.inventory, item],
        }));
      }
    } else if (messageText.includes('Removed') && messageText.includes('inventory')) {
      const itemMatch = messageText.match(/Removed ([^from]+) from inventory/i);
      if (itemMatch) {
        const item = itemMatch[1].trim();
        setPlayer((prev) => ({
          ...prev,
          inventory: prev.inventory.filter((i) => i !== item),
        }));
      }
    }
  }, [messages]);

  const hpPercentage = (player.hp / player.maxHp) * 100;
  const hpColor =
    hpPercentage > 60 ? 'from-green-500 to-emerald-600' : hpPercentage > 30 ? 'from-yellow-500 to-orange-600' : 'from-red-500 to-rose-600';

  return (
    <motion.div
      initial={{ opacity: 0, x: 100 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.4 }}
      className={cn('cyber-panel rounded-lg shadow-2xl overflow-hidden w-72', className)}
    >
      {/* Compact Header */}
      <div className="relative bg-black/60 border-b border-cyan-500/30 px-4 py-3">
        <div className="flex items-center gap-2">
          <Shield className="w-6 h-6 text-cyan-400" />
          <div className="flex-1 min-w-0">
            <h3 className="font-bold text-base text-cyan-400 truncate">{player.name}</h3>
            <p className="text-xs text-gray-400">Lvl {player.level} {player.class}</p>
          </div>
          <motion.button 
            whileHover={{ scale: 1.1 }} 
            whileTap={{ scale: 0.9 }} 
            onClick={() => setIsOpen(!isOpen)} 
            className="text-cyan-400/70 hover:text-cyan-400 transition-colors flex-shrink-0"
          >
            <motion.div
              animate={{ rotate: isOpen ? 180 : 0 }}
              transition={{ duration: 0.3 }}
            >
              <Scroll className="w-5 h-5" />
            </motion.div>
          </motion.button>
        </div>
      </div>

      {/* Dice Roll Notification */}
      <AnimatePresence>
        {lastRoll && (
          <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }} className="absolute top-20 left-0 right-0 z-50 mx-4">
            <div className="bg-gradient-to-r from-purple-600 to-indigo-600 border-2 border-purple-400 rounded-lg px-4 py-2 shadow-lg">
              <p className="text-white font-bold text-center">{lastRoll}</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Content */}
      <AnimatePresence>
        {isOpen && (
          <motion.div initial={{ height: 0 }} animate={{ height: 'auto' }} exit={{ height: 0 }} transition={{ duration: 0.3 }} className="overflow-hidden">
            <div className="p-4 space-y-3">
              {/* HP Bar */}
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <div className="flex items-center gap-1.5">
                    <Heart className="w-4 h-4 text-red-400" />
                    <span className="text-xs font-bold text-cyan-400 uppercase">HP</span>
                  </div>
                  <span className="text-xs font-bold text-cyan-400">
                    {player.hp}/{player.maxHp}
                  </span>
                </div>
                <div className="h-2 bg-black/60 rounded-sm overflow-hidden border border-cyan-500/30">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${hpPercentage}%` }}
                    transition={{ duration: 0.5 }}
                    className={cn('h-full bg-gradient-to-r', hpColor)}
                    style={{
                      boxShadow: hpPercentage > 30 ? '0 0 8px currentColor' : 'none',
                    }}
                  />
                </div>
              </div>

              {/* Stats - Compact Grid */}
              <div>
                <div className="flex items-center gap-1.5 mb-1.5">
                  <Flame className="w-4 h-4 text-purple-400" />
                  <span className="text-xs font-bold text-cyan-400 uppercase">Stats</span>
                </div>
                <div className="grid grid-cols-3 gap-1.5">
                  {Object.entries(player.stats).map(([stat, value]) => (
                    <div key={stat} className="bg-black/60 border border-cyan-500/30 rounded px-2 py-1.5 text-center">
                      <div className="text-[10px] text-gray-500 uppercase font-bold">{stat.slice(0, 3)}</div>
                      <div className="text-sm font-bold text-cyan-400">{value}</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Inventory - Compact */}
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <div className="flex items-center gap-1.5">
                    <Backpack className="w-4 h-4 text-purple-400" />
                    <span className="text-xs font-bold text-cyan-400 uppercase">Items</span>
                  </div>
                  <span className="text-xs text-gray-500">({player.inventory.length})</span>
                </div>
                <div className="max-h-32 overflow-y-auto custom-scrollbar space-y-1">
                  <AnimatePresence>
                    {player.inventory.map((item, index) => (
                      <motion.div
                        key={`${item}-${index}`}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: 20 }}
                        transition={{ delay: index * 0.05 }}
                        className="bg-black/60 border border-cyan-500/30 rounded px-2 py-1.5 hover:border-cyan-500/60 transition-all"
                      >
                        <span className="text-xs text-gray-300">{item}</span>
                      </motion.div>
                    ))}
                  </AnimatePresence>
                </div>
              </div>

              {/* Gold - Compact */}
              <div className="bg-black/60 border border-cyan-500/30 rounded px-3 py-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-cyan-400 uppercase">Gold</span>
                  <span className="text-base font-bold text-yellow-400">{player.gold} 🪙</span>
                </div>
              </div>

              {/* Recent Activity - Combined */}
              <div>
                <div className="flex items-center gap-1.5 mb-1.5">
                  <Scroll className="w-4 h-4 text-purple-400" />
                  <span className="text-xs font-bold text-cyan-400 uppercase">Activity</span>
                </div>
                <div className="space-y-1 max-h-24 overflow-y-auto custom-scrollbar">
                  {messages.slice(-10).reverse().filter(m => 
                    m.message && (
                      m.message.includes('🎲') || 
                      m.message.includes('Quest') ||
                      m.message.includes('⚔️') ||
                      m.message.includes('💔') ||
                      m.message.includes('💚')
                    )
                  ).slice(0, 3).map((msg, i) => (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: i * 0.05 }}
                      className="text-[10px] text-gray-400 bg-black/40 rounded px-2 py-1 border border-cyan-500/20 leading-tight"
                    >
                      {msg.message.substring(0, 50)}...
                    </motion.div>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
