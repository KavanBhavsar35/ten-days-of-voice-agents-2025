import sqlite3
import logging
import os
from datetime import datetime
from typing import Optional, Dict, List

logger = logging.getLogger("fraud-db")

class FraudDatabase:
    def __init__(self, db_path: str = "fraud_db.sqlite"):
        """Initialize SQLite database connection and create tables if needed."""
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Create tables if they don't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Cases table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS cases (
                        customer_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        card_last4 TEXT NOT NULL,
                        security_question TEXT NOT NULL,
                        security_answer TEXT NOT NULL,
                        status TEXT DEFAULT 'pending',
                        notes TEXT DEFAULT '',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Transactions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS transactions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        customer_id TEXT NOT NULL,
                        merchant TEXT NOT NULL,
                        amount TEXT NOT NULL,
                        location TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        FOREIGN KEY (customer_id) REFERENCES cases (customer_id)
                    )
                """)
                
                conn.commit()
                logger.info(f"Database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            raise
    
    def get_case_by_name(self, name: str) -> Optional[Dict]:
        """Retrieve a case by customer name."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Get case details
                cursor.execute("""
                    SELECT * FROM cases WHERE LOWER(name) = LOWER(?)
                """, (name,))
                
                case_row = cursor.fetchone()
                if not case_row:
                    return None
                
                case = dict(case_row)
                
                # Get associated transaction
                cursor.execute("""
                    SELECT merchant, amount, location, timestamp 
                    FROM transactions 
                    WHERE customer_id = ?
                    ORDER BY id DESC LIMIT 1
                """, (case['customer_id'],))
                
                tx_row = cursor.fetchone()
                if tx_row:
                    case['transaction'] = dict(tx_row)
                else:
                    case['transaction'] = {}
                
                return case
        except Exception as e:
            logger.error(f"Error fetching case by name '{name}': {e}")
            return None
    
    def get_case_by_id(self, customer_id: str) -> Optional[Dict]:
        """Retrieve a case by customer ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("SELECT * FROM cases WHERE customer_id = ?", (customer_id,))
                case_row = cursor.fetchone()
                
                if not case_row:
                    return None
                
                case = dict(case_row)
                
                # Get transaction
                cursor.execute("""
                    SELECT merchant, amount, location, timestamp 
                    FROM transactions 
                    WHERE customer_id = ?
                    ORDER BY id DESC LIMIT 1
                """, (customer_id,))
                
                tx_row = cursor.fetchone()
                case['transaction'] = dict(tx_row) if tx_row else {}
                
                return case
        except Exception as e:
            logger.error(f"Error fetching case by ID '{customer_id}': {e}")
            return None
    
    def update_case_status(self, customer_id: str, status: str, note: str) -> bool:
        """Update case status and append notes."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get existing notes
                cursor.execute("SELECT notes FROM cases WHERE customer_id = ?", (customer_id,))
                row = cursor.fetchone()
                
                if not row:
                    logger.warning(f"Customer ID {customer_id} not found")
                    return False
                
                existing_notes = row[0] or ""
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                new_notes = f"{existing_notes} | [{timestamp}] {note}".strip()
                
                # Update status and notes
                cursor.execute("""
                    UPDATE cases 
                    SET status = ?, notes = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE customer_id = ?
                """, (status, new_notes, customer_id))
                
                conn.commit()
                logger.info(f"Updated case {customer_id}: status={status}")
                return True
        except Exception as e:
            logger.error(f"Error updating case status: {e}")
            return False
    
    def add_case(self, customer_id: str, name: str, card_last4: str, 
                 security_question: str, security_answer: str,
                 merchant: str, amount: str, location: str, timestamp: str) -> bool:
        """Add a new fraud case with transaction."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Insert case
                cursor.execute("""
                    INSERT INTO cases (customer_id, name, card_last4, security_question, security_answer)
                    VALUES (?, ?, ?, ?, ?)
                """, (customer_id, name, card_last4, security_question, security_answer))
                
                # Insert transaction
                cursor.execute("""
                    INSERT INTO transactions (customer_id, merchant, amount, location, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (customer_id, merchant, amount, location, timestamp))
                
                conn.commit()
                logger.info(f"Added new case: {customer_id}")
                return True
        except sqlite3.IntegrityError:
            logger.warning(f"Case {customer_id} already exists")
            return False
        except Exception as e:
            logger.error(f"Error adding case: {e}")
            return False
    
    def list_all_cases(self) -> List[Dict]:
        """Get all cases with their basic info."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT customer_id, name, card_last4, status, updated_at 
                    FROM cases 
                    ORDER BY updated_at DESC
                """)
                
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error listing cases: {e}")
            return []