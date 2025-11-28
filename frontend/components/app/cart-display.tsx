'use client';

import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { ShoppingCart, X, Package, Sparkles, ShoppingBag, Trash2 } from 'lucide-react';
import { cn } from '@/lib/utils';

interface CartItem {
  id: string;
  name: string;
  price: number;
  unit: string;
  quantity: number;
}

interface CartDisplayProps {
  className?: string;
  messages: any[];
}

// Catalog prices mapping (from backend catalog)
const CATALOG_PRICES: Record<string, number> = {
  'milk': 60,
  'whole wheat bread': 45,
  'white bread': 35,
  'eggs': 90,
  'butter': 250,
  'cheese': 450,
  'peanut butter': 350,
  'strawberry jam': 280,
  'jam': 280,
  'tomato sauce': 120,
  'olive oil': 650,
  'spaghetti pasta': 85,
  'pasta': 85,
  'penne pasta': 85,
  'white rice': 180,
  'rice': 180,
  'potato chips': 40,
  'chips': 40,
  'chocolate cookies': 60,
  'cookies': 60,
  'granola bars': 150,
  'orange juice': 180,
  'juice': 180,
  'coffee': 450,
  'green tea': 250,
  'tea': 250,
  'pepperoni pizza': 399,
  'veggie pizza': 349,
  'pizza': 349,
  'chicken sandwich': 180,
  'sandwich': 180,
  'bananas': 50,
  'apples': 150,
  'tomatoes': 40,
  'lettuce': 60,
  'onions': 30,
  'garlic': 80,
  'flour': 50,
  'sugar': 45,
  'salt': 20,
  'ketchup': 80,
  'mayonnaise': 120,
  'soy sauce': 90,
  'honey': 180,
  'yogurt': 50,
  'paneer': 90,
  'cream': 60,
  'potatoes': 25,
  'carrots': 40,
  'cucumber': 30,
  'bell peppers': 80,
  'peppers': 80,
  'spinach': 35,
  'coriander': 20,
  'ginger': 60,
  'green chillies': 40,
  'chillies': 40,
  'biscuits': 30,
  'namkeen': 50,
  'popcorn': 40,
  'water bottle': 20,
  'water': 20,
  'soft drink': 40,
  'noodles': 35,
  'maggi': 35,
  'oats': 120,
  'poha': 60,
  'samosa': 40,
  'spring rolls': 80,
  'atta': 55,
  'wheat flour': 55,
  'besan': 70,
  'gram flour': 70,
  'dal': 120,
  'lentils': 120,
  'chickpeas': 90,
  'cooking oil': 180,
  'oil': 180,
  'ghee': 550,
  'pickle': 120,
  'chutney': 80,
  'nuts mix': 200,
  'nuts': 200,
  'dried fruits': 250,
  'lassi': 40,
  'cauliflower': 35,
  'cabbage': 30,
  'peas': 60,
  'beans': 50,
  'brinjal': 40,
  'eggplant': 40,
  'okra': 45,
  'bhindi': 45,
  'mint leaves': 20,
  'mint': 20,
  'lemon': 60,
  'mango': 120,
  'grapes': 80,
  'watermelon': 30,
  'papaya': 40,
};

function getPriceFromName(itemName: string): number {
  const nameLower = itemName.toLowerCase().trim();
  
  // Direct match
  if (CATALOG_PRICES[nameLower]) {
    return CATALOG_PRICES[nameLower];
  }
  
  // Partial match
  for (const [key, price] of Object.entries(CATALOG_PRICES)) {
    if (nameLower.includes(key) || key.includes(nameLower)) {
      return price;
    }
  }
  
  return 50; // Default fallback
}

export function CartDisplay({ className, messages }: CartDisplayProps) {
  const [cart, setCart] = useState<CartItem[]>([]);
  const [isOpen, setIsOpen] = useState(true);
  const [orderPlaced, setOrderPlaced] = useState(false);
  const [orderId, setOrderId] = useState<string>('');
  const [processedMessages, setProcessedMessages] = useState<Set<string>>(new Set());

  useEffect(() => {
    // Parse messages to extract cart information
    const lastMessage = messages[messages.length - 1];
    if (!lastMessage || !lastMessage.message) return;

    // Create unique message ID to prevent duplicate processing
    const messageId = `${lastMessage.timestamp || Date.now()}-${lastMessage.message}`;
    if (processedMessages.has(messageId)) return;

    const messageText = lastMessage.message.toLowerCase();

    // Mark message as processed
    setProcessedMessages(prev => new Set(prev).add(messageId));

    // Check if message contains cart information
    if (messageText.includes("added") && messageText.includes("cart")) {
      // Extract item info from message
      parseAndUpdateCart(lastMessage.message);
    } else if (messageText.includes("removed") && messageText.includes("cart")) {
      parseAndRemoveFromCart(lastMessage.message);
    } else if (messageText.includes("updated") && messageText.includes("quantity")) {
      parseAndUpdateQuantity(lastMessage.message);
    } else if (messageText.includes("order placed") || messageText.includes("order id")) {
      // Extract order ID
      const orderIdMatch = lastMessage.message.match(/order id[:\s]+([a-z0-9-]+)/i);
      if (orderIdMatch) {
        setOrderId(orderIdMatch[1]);
      }
      setOrderPlaced(true);
      // Don't auto-clear - let user see the success message
    }
  }, [messages, processedMessages]);

  const parseAndUpdateCart = (message: string) => {
    // Check if message contains multiple items with quantities
    // Format: "I've added 1 litre of milk, 1 dozen eggs, and 1 loaf of whole wheat bread"
    const multiItemPattern = /(\d+\s+(?:litre|liter|kg|kilogram|pack|box|jar|bottle|bunch|unit|loaf|dozen|g|gram|ml)s?\s+of\s+[^,]+)/gi;
    const multiItems = message.match(multiItemPattern);
    
    if (multiItems && multiItems.length > 1) {
      // Process each item separately
      multiItems.forEach(itemText => {
        const qtyMatch = itemText.match(/(\d+)\s+/);
        const nameMatch = itemText.match(/of\s+(.+)$/i);
        
        if (qtyMatch && nameMatch) {
          const quantity = parseInt(qtyMatch[1]);
          let name = nameMatch[1].trim()
            .replace(/\s+to your cart.*$/i, '')
            .replace(/\s+and\s+\d+.*$/i, ''); // Remove trailing "and X..."
          
          // Capitalize properly
          name = name.split(' ').map(word => 
            word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()
          ).join(' ');
          
          const price = getPriceFromName(name);
          
          setCart(prevCart => {
            const exists = prevCart.find(item => 
              item.name.toLowerCase() === name.toLowerCase()
            );
            if (!exists) {
              return [...prevCart, {
                id: `item-${Date.now()}-${Math.random()}`,
                name,
                price,
                unit: 'unit',
                quantity
              }];
            }
            return prevCart;
          });
        }
      });
      return;
    }
    
    // Check for simple list format: "adding pasta, sauce, and oil"
    if ((message.toLowerCase().includes('adding') || message.toLowerCase().includes('added')) && 
        (message.includes(',') || message.includes(' and '))) {
      const items = message.match(/(?:adding|added)[:\s]+([^.!]+)/i);
      if (items && items[1]) {
        const itemList = items[1].split(/,\s*(?:and\s+)?|\s+and\s+/).map(item => item.trim());
        itemList.forEach(itemName => {
          if (itemName && itemName.length > 2) {
            let cleanName = itemName
              .replace(/\s*\(.*?\)\s*/g, '')
              .replace(/^\d+\s+(?:litre|kg|pack|box|jar|bottle|bunch|unit|loaf|dozen|g|ml)s?\s+of\s+/i, '')
              .trim();
            
            // Capitalize
            cleanName = cleanName.split(' ').map(word => 
              word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()
            ).join(' ');
            
            setCart(prevCart => {
              const exists = prevCart.find(item => 
                item.name.toLowerCase() === cleanName.toLowerCase()
              );
              if (!exists) {
                return [...prevCart, {
                  id: `item-${Date.now()}-${Math.random()}`,
                  name: cleanName,
                  price: getPriceFromName(cleanName),
                  unit: 'unit',
                  quantity: 1
                }];
              }
              return prevCart;
            });
          }
        });
      }
      return;
    }
    
    // Single item parsing
    const quantityMatch = message.match(/(\d+)\s+(?:litre|liter|kg|kilogram|pack|box|jar|bottle|bunch|unit|loaf|dozen|g|gram|ml)/i);
    const nameMatch = message.match(/of ([^(]+?)(?:\s+to your cart|\s+are now|\s+\(|$)/i) || 
                      message.match(/(?:added|adding)\s+(?:\d+\s+)?(?:litre|kg|pack|box|jar|bottle|bunch|unit|loaf|dozen|g|ml)?\s*(?:of\s+)?([^,]+?)(?:\s+to|\s+are|\s*$)/i);
    
    const priceMatch = message.match(/₹\s*(\d+)/);

    if (nameMatch) {
      const quantity = quantityMatch ? parseInt(quantityMatch[1]) : 1;
      let name = nameMatch[1].trim()
        .replace(/\s*\(.*?\)\s*/g, '')
        .replace(/\s+each.*$/i, '')
        .replace(/\s+are now.*$/i, '')
        .replace(/\s+to your cart.*$/i, '');
      
      // Capitalize
      name = name.split(' ').map(word => 
        word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()
      ).join(' ');
      
      const messagePrice = priceMatch ? parseInt(priceMatch[1]) : null;
      const price = messagePrice || getPriceFromName(name);

      setCart(prevCart => {
        const existingItem = prevCart.find(item => 
          item.name.toLowerCase() === name.toLowerCase()
        );
        
        if (existingItem) {
          return prevCart.map(item =>
            item.name.toLowerCase() === name.toLowerCase()
              ? { ...item, quantity: quantity, price: price }
              : item
          );
        } else {
          return [...prevCart, {
            id: `item-${Date.now()}-${Math.random()}`,
            name,
            price,
            unit: 'unit',
            quantity
          }];
        }
      });
    }
  };

  const parseAndRemoveFromCart = (message: string) => {
    const nameMatch = message.match(/removed ([^from]+) from/i);
    if (nameMatch) {
      const name = nameMatch[1].trim();
      setCart(prevCart => prevCart.filter(item => 
        !item.name.toLowerCase().includes(name.toLowerCase())
      ));
    }
  };

  const parseAndUpdateQuantity = (message: string) => {
    const nameMatch = message.match(/updated ([^quantity]+) quantity to (\d+)/i);
    if (nameMatch) {
      const name = nameMatch[1].trim();
      const newQuantity = parseInt(nameMatch[2]);
      setCart(prevCart => prevCart.map(item =>
        item.name.toLowerCase().includes(name.toLowerCase())
          ? { ...item, quantity: newQuantity }
          : item
      ));
    }
  };

  const calculateTotal = () => {
    return cart.reduce((total, item) => total + (item.price * item.quantity), 0);
  };

  const total = calculateTotal();

  return (
    <motion.div
      initial={{ opacity: 0, x: 100, scale: 0.9 }}
      animate={{ opacity: 1, x: 0, scale: 1 }}
      transition={{ duration: 0.4, type: "spring", bounce: 0.3 }}
      className={cn(
        'bg-gradient-to-br from-background/98 via-background/95 to-primary/5 backdrop-blur-xl border-2 border-primary/20 rounded-2xl shadow-2xl overflow-hidden',
        className
      )}
    >
      {/* Header with Gradient */}
      <div className="relative bg-gradient-to-r from-primary/20 via-primary/15 to-accent/20 border-b-2 border-primary/30 px-5 py-4">
        <div className="absolute inset-0 bg-gradient-to-br from-primary/10 to-transparent opacity-50" />
        <div className="relative flex items-center justify-between">
          <div className="flex items-center gap-3">
            <motion.div
              animate={{ rotate: cart.length > 0 ? [0, -10, 10, -10, 0] : 0 }}
              transition={{ duration: 0.5 }}
              className="relative"
            >
              <ShoppingBag className="w-6 h-6 text-primary" />
              {cart.length > 0 && (
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  className="absolute -top-2 -right-2"
                >
                  <Sparkles className="w-3 h-3 text-yellow-500 fill-yellow-500" />
                </motion.div>
              )}
            </motion.div>
            <div>
              <h3 className="font-bold text-lg text-foreground flex items-center gap-2">
                Your Cart
                {cart.length > 0 && (
                  <motion.span
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    className="bg-gradient-to-r from-primary to-primary/80 text-primary-foreground text-xs font-bold rounded-full w-6 h-6 flex items-center justify-center shadow-lg"
                  >
                    {cart.length}
                  </motion.span>
                )}
              </h3>
              {cart.length > 0 && (
                <p className="text-xs text-muted-foreground">
                  {cart.reduce((sum, item) => sum + item.quantity, 0)} items
                </p>
              )}
            </div>
          </div>
          <motion.button
            whileHover={{ scale: 1.1, rotate: 90 }}
            whileTap={{ scale: 0.9 }}
            onClick={() => setIsOpen(!isOpen)}
            className="text-muted-foreground hover:text-foreground transition-colors p-2 rounded-full hover:bg-primary/10"
          >
            {isOpen ? <X className="w-5 h-5" /> : <Package className="w-5 h-5" />}
          </motion.button>
        </div>
      </div>

      {/* Cart Items */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ height: 0 }}
            animate={{ height: 'auto' }}
            exit={{ height: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="max-h-[450px] overflow-y-auto custom-scrollbar">
              {orderPlaced ? (
                <motion.div
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="px-6 py-12 text-center"
                >
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1, rotate: 360 }}
                    transition={{ type: "spring", bounce: 0.5 }}
                    className="w-20 h-20 mx-auto mb-4 bg-gradient-to-br from-green-500 to-emerald-600 rounded-full flex items-center justify-center"
                  >
                    <svg className="w-12 h-12 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                    </svg>
                  </motion.div>
                  <h3 className="text-xl font-bold text-foreground mb-2">Order Placed! 🎉</h3>
                  {orderId && (
                    <p className="text-sm text-muted-foreground mb-3">Order ID: {orderId}</p>
                  )}
                  <p className="text-base font-medium text-primary mb-1">Your order will arrive shortly!</p>
                  <p className="text-sm text-muted-foreground">Estimated delivery: 30-45 minutes</p>
                </motion.div>
              ) : cart.length === 0 ? (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="px-6 py-12 text-center"
                >
                  <motion.div
                    animate={{ 
                      y: [0, -10, 0],
                      rotate: [0, 5, -5, 0]
                    }}
                    transition={{ 
                      duration: 3,
                      repeat: Infinity,
                      ease: "easeInOut"
                    }}
                  >
                    <ShoppingCart className="w-16 h-16 mx-auto mb-4 text-primary/30" />
                  </motion.div>
                  <p className="text-base font-medium text-foreground mb-1">Your cart is empty</p>
                  <p className="text-sm text-muted-foreground">Start adding delicious items! 🛒</p>
                </motion.div>
              ) : (
                <div className="p-2 space-y-2">
                  <AnimatePresence mode="popLayout">
                    {cart.map((item, index) => (
                      <motion.div
                        key={item.id}
                        initial={{ opacity: 0, x: -50, scale: 0.8 }}
                        animate={{ opacity: 1, x: 0, scale: 1 }}
                        exit={{ opacity: 0, x: 100, scale: 0.8 }}
                        transition={{ 
                          duration: 0.3,
                          delay: index * 0.05,
                          type: "spring",
                          bounce: 0.4
                        }}
                        layout
                        className="relative group"
                      >
                        <div className="bg-gradient-to-r from-accent/30 via-accent/20 to-transparent hover:from-accent/50 hover:via-accent/30 border border-border/50 rounded-xl p-4 transition-all duration-300 hover:shadow-lg hover:scale-[1.02]">
                          <div className="flex justify-between items-start gap-3">
                            <div className="flex-1 min-w-0">
                              <div className="flex items-start gap-2">
                                <motion.div
                                  initial={{ scale: 0 }}
                                  animate={{ scale: 1 }}
                                  className="mt-0.5 w-2 h-2 rounded-full bg-gradient-to-r from-primary to-accent flex-shrink-0"
                                />
                                <div className="flex-1">
                                  <p className="font-semibold text-sm text-foreground leading-tight">
                                    {item.name}
                                  </p>
                                  <div className="flex items-center gap-2 mt-1.5">
                                    <span className="text-xs px-2 py-0.5 rounded-full bg-primary/10 text-primary font-medium">
                                      ₹{item.price}
                                    </span>
                                    <span className="text-xs text-muted-foreground">
                                      × {item.quantity}
                                    </span>
                                  </div>
                                </div>
                              </div>
                            </div>
                            <div className="text-right">
                              <motion.p
                                key={item.price * item.quantity}
                                initial={{ scale: 1.2, color: "#10b981" }}
                                animate={{ scale: 1, color: "inherit" }}
                                className="font-bold text-base text-foreground"
                              >
                                ₹{item.price * item.quantity}
                              </motion.p>
                            </div>
                          </div>
                        </div>
                      </motion.div>
                    ))}
                  </AnimatePresence>
                </div>
              )}
            </div>

            {/* Total with Gradient */}
            {cart.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="relative border-t-2 border-primary/30 bg-gradient-to-br from-primary/20 via-accent/20 to-primary/10 px-5 py-4"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-primary/5 to-accent/5 opacity-50" />
                <div className="relative">
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-bold text-base text-foreground flex items-center gap-2">
                      <Sparkles className="w-4 h-4 text-yellow-500" />
                      Total Amount
                    </span>
                    <motion.span
                      key={total}
                      initial={{ scale: 1.3, color: "#10b981" }}
                      animate={{ scale: 1, color: "inherit" }}
                      transition={{ type: "spring", bounce: 0.5 }}
                      className="font-black text-2xl bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent"
                    >
                      ₹{total}
                    </motion.span>
                  </div>
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-muted-foreground">
                      {cart.length} item{cart.length !== 1 ? 's' : ''} • {cart.reduce((sum, item) => sum + item.quantity, 0)} units
                    </span>
                    <span className="text-primary font-medium">
                      Ready to order! 🎉
                    </span>
                  </div>
                </div>
              </motion.div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
